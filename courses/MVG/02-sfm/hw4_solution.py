import os
import cv2
import numpy as np

import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature_solution import BuildFeatureTrack
from camera_pose_solution import EstimateCameraPose
from camera_pose_solution import Triangulation
from camera_pose_solution import EvaluateCheirality
from pnp_solution import PnP_RANSAC
from pnp_solution import PnP_nl
from reconstruction_solution import FindMissingReconstruction
from reconstruction_solution import Triangulation_nl
from reconstruction_solution import RunBundleAdjustment


if __name__ == '__main__':
    K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1]
    ])
    num_images = 6
    h_im = 540
    w_im = 960

    # Load input images
    Im = np.empty((num_images, h_im, w_im, 3), dtype=np.uint8)
    for i in range(num_images):
        im_file = 'im/image{:07d}.jpg'.format(i + 1)
        im = cv2.imread(im_file)
        if im is None:
            print(f"Error loading {im_file}")
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im[i,:,:,:] = im

    # Build feature track
    print("Building feature tracks...")
    track = BuildFeatureTrack(Im, K)
    print(f"Track shape: {track.shape}")

    track1 = track[0,:,:]
    track2 = track[1,:,:]
    
    # Filter valid matches for initialization
    # track entries are -1 if invalid
    valid_ind = (track1[:, 0] != -1) & (track2[:, 0] != -1)
    t1_init = track1[valid_ind]
    t2_init = track2[valid_ind]

    # Estimate ﬁrst two camera poses
    print("Estimating initial pose...")
    R, C, X_init = EstimateCameraPose(t1_init, t2_init)

    output_dir = 'output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set of camera poses
    P = np.zeros((num_images, 3, 4))
    
    # Set first two camera poses
    # P1 = [I | 0]
    P[0] = np.eye(3, 4)
    
    # P2 = [R | t], t = -R C
    t = -R @ C
    P[1] = np.hstack((R, t.reshape(3,1)))
    
    # Initialize 3D points
    # X needs to hold all points. Initialize with -1
    num_features = track.shape[1]
    X = np.full((num_features, 3), -1.0)
    
    # Fill in points reconstructed from 0-1
    # We need to map X_init back to 'valid_ind' locations
    X[valid_ind] = X_init

    ransac_n_iter = 200
    ransac_thr = 0.01
    
    for i in range(2, num_images):
        print(f"Processing image {i+1}...")
        
        # Estimate new camera pose
        # Find points in X that are visible in current image i
        # i.e., track[i] is valid AND X is valid
        
        # valid_3d = (X[:, 0] != -1) # Simple check if point is reconstructed
        # Actually X might be initialized with -1.
        valid_3d = (X[:, 2] != -1) # Checking Z or any coord
        valid_2d = (track[i, :, 0] != -1)
        
        common = valid_3d & valid_2d
        
        if np.sum(common) < 6:
            print(f"Not enough points to register image {i+1}")
            continue
            
        X_pnp = X[common]
        x_pnp = track[i, common]
        
        print(f"  PnP with {len(X_pnp)} points")

        R_new, C_new, inliers = PnP_RANSAC(X_pnp, x_pnp, ransac_n_iter, ransac_thr)
        
        # Refine PnP
        # Filter inliers for refinement
        if np.sum(inliers) > 4:
            inlier_mask = inliers.astype(bool)
            R_new, C_new = PnP_nl(R_new, C_new, X_pnp[inlier_mask], x_pnp[inlier_mask])

        # Add new camera pose to the set
        t_new = -R_new @ C_new
        P[i] = np.hstack((R_new, t_new.reshape(3,1)))

        # Triangulate new points
        for j in range(i):
            # Find new points to reconstruct
            # Points seen in i and j, BUT NOT yet in X
            seen_i = (track[i, :, 0] != -1)
            seen_j = (track[j, :, 0] != -1)
            not_in_X = (X[:, 0] == -1) # Assuming -1 means invalid
            
            candidates = seen_i & seen_j & not_in_X
            
            if np.sum(candidates) == 0:
                continue
                
            idx_cand = np.where(candidates)[0]
            
            # Triangulate points
            pts_i = track[i, idx_cand]
            pts_j = track[j, idx_cand]
            
            X_new_cand = Triangulation(P[j], P[i], pts_j, pts_i)

            # Filter out points based on cheirality
            valid_cheir = EvaluateCheirality(P[j], P[i], X_new_cand)
            
            # Update 3D points
            # Only update valid ones
            start_update_idx = idx_cand[valid_cheir]
            X[start_update_idx] = X_new_cand[valid_cheir]
        
        # Run bundle adjustment
        print("  Running Bundle Adjustment...")
        valid_ind = X[:, 0] != -1
        X_ba = X[valid_ind, :]
        track_ba = track[:i + 1, valid_ind, :]
        
        # P input: only up to i
        P_ba_full, X_ba_new = RunBundleAdjustment(P[:i + 1, :, :], X_ba, track_ba)
        
        P[:i + 1, :, :] = P_ba_full
        X[valid_ind, :] = X_ba_new

        ###############################################################
        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i+1):
            R_d = P[j, :, :3]
            C_d = -R_d.T @ P[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh('{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_current = X[valid_ind]
        if len(X_current) > 0:
            X_new_h = np.hstack([X_current, np.ones((X_current.shape[0],1))])
            colors = np.zeros_like(X_current)
            for j in range(i, -1, -1):
                x = X_new_h @ P[j,:,:].T
                x = x / x[:, 2, np.newaxis]
                mask_valid = (x[:,0] >= -1) * (x[:,0] <= 1) * (x[:,1] >= -1) * (x[:,1] <= 1)
                
                # Only color points that are valid in this view
                # And haven't been colored yet (optional strategy, here we just overwrite)
                # But X indices need to map back to original X via valid_ind?
                # Ah, 'X_current' matches 'colors'.
                
                uv = x[mask_valid,:] @ K.T
                
                # Careful: 'mask_valid' is for X_current subset
                # We need to extract colors for these valid points
                
                # Image color lookup
                # Only if image exists
                # image indices
                
                # Vectorized interpolation is tricky with masking.
                # Loop is provided in template, let's trust it works if logic holds.
                pass # Template code logic for coloring was correct
                
                # Re-implementing color logic:
                im_curr = Im[j].astype(float)/255.0
                for k in range(3):
                    interp_fun = RectBivariateSpline(np.arange(h_im), np.arange(w_im), im_curr[:,:,k], kx=1, ky=1)
                    # uv coords are (u, v) -> (x, y) = (col, row)
                    # RectBivariateSpline expects (x, y) = (row, col)
                    # uv[:, 1] is y (row), uv[:, 0] is x (col)
                    interpolated = interp_fun(uv[:,1], uv[:,0], grid=False)
                    colors[mask_valid, k] = interpolated

            ind = np.sqrt(np.sum(X_current ** 2, axis=1)) < 200 # Clipping far points
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_current[ind]))
            pcd.colors = o3d.utility.Vector3dVector(colors[ind])
            o3d.io.write_point_cloud('{}/points_{}.ply'.format(output_dir, i+1), pcd)
    
    print("SfM pipeline finished.")