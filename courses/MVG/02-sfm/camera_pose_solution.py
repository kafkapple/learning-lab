import numpy as np
import cv2

def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E
    """
    R1, R2, t = cv2.decomposeEssentialMat(E)
    
    # 4 Configurations: (R1, t), (R1, -t), (R2, t), (R2, -t)
    # The output of decomposeEssentialMat: 
    # R1, R2: 3x3
    # t: 3x1
    
    R_set = np.zeros((4, 3, 3))
    C_set = np.zeros((4, 3))
    
    configurations = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]
    
    for i, (R, T_vec) in enumerate(configurations):
        R_set[i] = R
        # Camera center C = -R^T * t
        C = -R.T @ T_vec
        C_set[i] = C.ravel()
        
    return R_set, C_set

def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point
    """
    # cv2.triangulatePoints requires 2xN arrays
    pts4D = cv2.triangulatePoints(P1, P2, track1.T, track2.T)
    
    # Convert to 3D (homogeneous division)
    pts3D = pts4D[:3] / pts4D[3]
    
    return pts3D.T

def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points
    """
    # Points must be in front of both cameras (Z > 0)
    
    # Homogeneous 3D points
    X_h = np.hstack([X, np.ones((X.shape[0], 1))])
    
    # Project to camera 1
    x1_cam = (P1 @ X_h.T).T # (N, 3)
    
    # Project to camera 2
    x2_cam = (P2 @ X_h.T).T # (N, 3)
    
    # Check Z > 0
    # Note: In standard camera frame, Z>0 is in front.
    # However, P = K[R|t] or just [R|t] for normalized.
    # If P includes K, then the 3rd component w usually corresponds to Z * const.
    # Since we are using Extrinsics P (normalized), w = z.
    
    valid_index = (x1_cam[:, 2] > 0) & (x2_cam[:, 2] > 0)
    return valid_index

def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration
    """
    # 1. Estimate E using RANSAC
    E, mask = cv2.findEssentialMat(track1, track2, focal=1.0, pp=(0,0), 
                                   method=cv2.RANSAC, prob=0.999, threshold=0.001)
    
    if E is None or E.shape != (3,3):
        # Fallback or error handling
        # Sometimes RANSAC fails or returns multiple matrices. Take the first one.
        if E is not None and E.shape[0] > 3:
            E = E[:3, :3]
    
    # 2. Decompose E to 4 potential poses
    R_set, C_set = GetCameraPoseFromE(E)
    
    # 3. Triangulate and check Cheirality for each configuration
    best_num_valid = -1
    best_idx = 0
    best_X = None
    
    # P1 is Identity [I | 0]
    P1 = np.eye(3, 4)
    
    for i in range(4):
        R = R_set[i]
        C = C_set[i]
        t = -R @ C
        P2 = np.hstack((R, t.reshape(3,1)))
        
        # Triangulate
        try:
            X = Triangulation(P1, P2, track1, track2)
            
            # Check Cheirality
            valid = EvaluateCheirality(P1, P2, X)
            num_valid = np.sum(valid)
            
            if num_valid > best_num_valid:
                best_num_valid = num_valid
                best_idx = i
                best_X = X
        except Exception:
            continue
            
    R_best = R_set[best_idx]
    C_best = C_set[best_idx]
    
    # Return best X? The signature says X: (F, 3).
    # But usually we only keep inliers or valid ones? 
    # For initial pair, we should probably keep valid points effectively.
    # But for the assignment, sticking to returning the triangulated set corresponding to best pose.
    
    return R_best, C_best, best_X