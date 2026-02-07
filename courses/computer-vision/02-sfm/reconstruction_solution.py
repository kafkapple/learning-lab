import numpy as np
from scipy.optimize import least_squares
from utils_solution import Rotation2Quaternion
from utils_solution import Quaternion2Rotation

def FindMissingReconstruction(X, track_i):
    """
    Find the points that will be newly added
    """
    # track_i: (F, 2)
    # X: (F, 3)
    
    # Points visible in current image
    visible_in_image = (track_i[:, 0] != -1)
    
    # Points NOT yet reconstructed (assuming X initialized to zeros or -1 implies invalid)
    # Usually X is not initialized with -1 but might be all zeros? 
    # Or maybe we rely on a separate 'valid' mask.
    # Based on hw4.py, X is initialized. Let's assume un-reconstructed points are (0,0,0) or marked somehow.
    # Reviewing hw4.py: `valid_ind = X[:, 0] != -1`. So -1 is invalid.
    
    not_reconstructed = (X[:, 0] == -1)
    
    new_point = visible_in_image & not_reconstructed
    return new_point

def Triangulation_nl(X, P1, P2, x1, x2):
    """
    Refine the triangulated points
    """
    # X: (n, 3) initial guess
    X_new = np.zeros_like(X)
    
    for i in range(len(X)):
        def residual(pt):
            pt_h = np.append(pt, 1)
            
            # Proj 1
            p1 = P1 @ pt_h
            p1 = p1[:2] / p1[2]
            
            # Proj 2
            p2 = P2 @ pt_h
            p2 = p2[:2] / p2[2]
            
            return np.hstack([p1 - x1[i], p2 - x2[i]])
            
        res = least_squares(residual, X[i])
        X_new[i] = res.x
        
    return X_new

def ComputePointJacobian(X, p):
    # SetupBundleAdjustment uses this? Or is it for manual BA?
    # We will use scipy.optimize.least_squares which does auto-diff.
    pass

def SetupBundleAdjustment(P, X, track):
    # Helper to format data for BA
    # Not strictly needed if we implement RunBundleAdjustment with internal setup
    pass

def MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index):
    pass

def UpdatePosePoint(z, n_cameras, n_points):
    pass

def RunBundleAdjustment(P, X, track):
    """
    Run bundle adjustment
    """
    # P: (K, 3, 4)
    # X: (J, 3)  <-- Only BA on valid points
    # track: (K, J, 2)
    
    n_cameras = P.shape[0]
    n_points = X.shape[0]
    
    # Identify observations
    camera_indices = []
    point_indices = []
    observations = []
    
    for i in range(n_cameras):
        for j in range(n_points):
            if track[i, j, 0] != -1:
                camera_indices.append(i)
                point_indices.append(j)
                observations.append(track[i, j])
                
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    observations = np.array(observations)
    
    if len(observations) == 0:
        return P, X

    # Initial parameter vector
    # Cameras: K * 7
    x0_cam = []
    for i in range(n_cameras):
        R = P[i, :3, :3]
        t = P[i, :3, 3] # This is translation vector t
        # We optimize quaternion and t
        q = Rotation2Quaternion(R)
        x0_cam.extend(q)
        x0_cam.extend(t)
    
    x0_points = X.ravel()
    
    x0 = np.hstack([x0_cam, x0_points])
    
    n_cam_params = n_cameras * 7
    
    def fun(params):
        cam_params = params[:n_cam_params].reshape((n_cameras, 7))
        points_3d = params[n_cam_params:].reshape((n_points, 3))
        
        # Get relevant params for observations
        qs = cam_params[camera_indices, :4]
        ts = cam_params[camera_indices, 4:]
        pts = points_3d[point_indices]
        
        # Project
        # Custom rotation application for speed or loop
        residuals = []
        
        # Ideally vectorised:
        # Rotations are expensive to vectorize without a helper or scipy rotation object
        # With len(observations) ~ thousands, loop might be slow but acceptable for HW.
        
        # Let's try a vectorized approach for Rotation if possible, but quaternions -> matrix is standard
        # Using loop for simplicity and robustness
        
        for k in range(len(camera_indices)):
            q = qs[k]
            t = ts[k]
            pt = pts[k]
            
            # q -> R
            # Doing this per point is slow. Optim: Precompute R for each camera.
            # But inside least_squares 'fun', we get new params every iter.
            # So we must compute R from q.
            
            # Simple quaternion rotate v' = q v q*? Or just Matrix.
            R = Quaternion2Rotation(q)
            
            pt_cam = R @ pt + t
            
            if pt_cam[2] == 0:
                residuals.append([0, 0]) # avoid div by zero
                continue
                
            proj = pt_cam[:2] / pt_cam[2]
            residuals.append(proj - observations[k])
            
        return np.array(residuals).ravel()

    # Optimization
    # Ideally should pass jacobian sparsity (tridiagonal-ish)
    res = least_squares(fun, x0, verbose=0)
    
    # Store results
    x_opt = res.x
    cam_params_new = x_opt[:n_cam_params].reshape((n_cameras, 7))
    points_new = x_opt[n_cam_params:].reshape((n_points, 3))
    
    P_new = np.zeros_like(P)
    for i in range(n_cameras):
        q = cam_params_new[i, :4]
        t = cam_params_new[i, 4:]
        R = Quaternion2Rotation(q)
        P_new[i] = np.hstack((R, t.reshape(3,1)))
        
    return P_new, points_new