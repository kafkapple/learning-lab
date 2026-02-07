import numpy as np
import cv2

def computeH(x1, x2):
    # Q3.6
    # Compute the homography between two sets of points
    A = []
    for i in range(x1.shape[0]):
        p1 = x1[i]
        p2 = x2[i]
        A.append([-p2[0], -p2[1], -1, 0, 0, 0, p1[0]*p2[0], p1[0]*p2[1], p1[0]])
        A.append([0, 0, 0, -p2[0], -p2[1], -1, p1[1]*p2[0], p1[1]*p2[1], p1[1]])
    
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    H2to1 = Vh[-1, :].reshape(3, 3)
    return H2to1



def computeH_norm(x1, x2):
    # Q3.7
    # Compute the centroid of the points
    x1_centroid = np.mean(x1, axis=0)
    x2_centroid = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    x1_shifted = x1 - x1_centroid
    x2_shifted = x2 - x2_centroid

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    avg_dist1 = np.mean(np.sqrt(np.sum(x1_shifted**2, axis=1)))
    avg_dist2 = np.mean(np.sqrt(np.sum(x2_shifted**2, axis=1)))

    s1 = np.sqrt(2) / avg_dist1
    s2 = np.sqrt(2) / avg_dist2

    # Similarity transform 1
    T1 = np.array([[s1, 0, -s1*x1_centroid[0]],
                   [0, s1, -s1*x1_centroid[1]],
                   [0, 0, 1]])

    # Similarity transform 2
    T2 = np.array([[s2, 0, -s2*x2_centroid[0]],
                   [0, s2, -s2*x2_centroid[1]],
                   [0, 0, 1]])
    
    # Homogeneous coordinates
    x1_homo = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x2_homo = np.hstack((x2, np.ones((x2.shape[0], 1))))

    x1_norm = (T1 @ x1_homo.T).T
    x2_norm = (T2 @ x2_homo.T).T

    # Compute homography
    H_norm = computeH(x1_norm[:, :2], x2_norm[:, :2])
    
    # Denormalization
    H2to1 = np.linalg.inv(T1) @ H_norm @ T2
    return H2to1

def computeH_ransac(x1, x2):
    # Q3.8
    # Compute the best fitting homography given a list of matching points
    max_iters = 1000
    inlier_tol = 2.5
    bestH2to1 = None
    inliers = None
    max_inliers = 0

    num_points = x1.shape[0]

    for _ in range(max_iters):
        # Randomly sample 4 points
        indices = np.random.choice(num_points, 4, replace=False)
        p1 = x1[indices]
        p2 = x2[indices]
        
        # Compute homography
        H = computeH_norm(p1, p2)

        # Apply homography to all points
        x2_homo = np.hstack((x2, np.ones((num_points, 1))))
        x1_proj_homo = (H @ x2_homo.T).T
        x1_proj = x1_proj_homo[:, :2] / x1_proj_homo[:, 2, np.newaxis]
        
        # Calculate distances and count inliers
        dist = np.sqrt(np.sum((x1 - x1_proj)**2, axis=1))
        current_inliers = np.where(dist < inlier_tol)[0]
        
        if len(current_inliers) > max_inliers:
            max_inliers = len(current_inliers)
            inliers = current_inliers
            bestH2to1 = H
            
    return bestH2to1, inliers

def compositeH(H2to1, template, img):
    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.
    H_inv = np.linalg.inv(H2to1)

    # Create mask of same size as template
    mask = np.ones(template.shape, dtype=np.uint8) * 255

    # Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H_inv, (img.shape[1], img.shape[0]))

    # Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H_inv, (img.shape[1], img.shape[0]))

    # Use mask to combine the warped template and the image
    composite_img = img.copy()
    composite_img[warped_mask > 0] = 0
    composite_img += warped_template
    
    return composite_img
