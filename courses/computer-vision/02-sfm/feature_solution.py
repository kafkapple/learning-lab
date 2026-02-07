import cv2
import numpy as np

def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    """
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
            
    x1 = np.float32([loc1[m.queryIdx] for m in good])
    x2 = np.float32([loc2[m.trainIdx] for m in good])
    ind1 = np.array([m.queryIdx for m in good])
    
    # We also need to return the indices in loc2 to build tracks!
    # The template signature: Returns x1, x2, ind1.
    # But for BuildFeatureTrack we definitely need the indices in image 2 as well.
    # Let's stick to the template for this function but helper functions might need more.
    
    return x1, x2, ind1

def EstimateE(x1, x2):
    """
    Estimate the essential matrix
    """
    # Assuming x1, x2 are normalized coordinates as per the project structure usually,
    # or if pixel coords, we'd need K.
    # findEssentialMat can take pixel coords and K. 
    # If inputs are normalized, focal=1.0, pp=(0,0) is correct.
    E, mask = cv2.findEssentialMat(x1, x2, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.001)
    return E

def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC
    """
    E, mask = cv2.findEssentialMat(x1, x2, focal=1.0, pp=(0, 0), method=cv2.RANSAC, 
                                   prob=0.999, threshold=ransac_thr)
    if mask is None:
        return E, np.array([])
    inlier = mask.ravel().astype(bool)
    return E, inlier

def BuildFeatureTrack(Im, K):
    """
    Build feature track
    """
    num_images = Im.shape[0]
    sift = cv2.SIFT_create()
    
    keypoints_list = []
    descriptors_list = []
    
    # 1. Detect Features
    for i in range(num_images):
        gray = cv2.cvtColor(Im[i], cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        
        # Convert to normalized coordinates
        kp_pt = cv2.KeyPoint_convert(kp) # (N, 2)
        kp_h = np.hstack([kp_pt, np.ones((kp_pt.shape[0], 1))])
        kp_norm = (np.linalg.inv(K) @ kp_h.T).T[:, :2]
        
        keypoints_list.append(kp_norm)
        descriptors_list.append(des)
        
    # 2. Build Tracks
    # We will match adjacent pairs (0-1, 1-2, ...) and chain them.
    # This is a simplified SfM tracking.
    
    matcher = cv2.BFMatcher()
    
    # Track structure: dict mapping track_id -> {img_idx: (x, y)}
    # But output format is (N, F, 2).
    # Let's use Union-Find to group observations into tracks.
    
    # Maps (img_idx, feature_idx) -> parent_node
    parent = {}
    
    def find(node):
        if parent.setdefault(node, node) == node:
            return node
        parent[node] = find(parent[node])
        return parent[node]
    
    def union(n1, n2):
        root1 = find(n1)
        root2 = find(n2)
        if root1 != root2:
            parent[root2] = root1
            
    for i in range(num_images - 1):
        des1 = descriptors_list[i]
        des2 = descriptors_list[i+1]
        
        matches = matcher.knnMatch(des1, des2, k=2)
        
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                # Valid match
                idx1 = m.queryIdx
                idx2 = m.trainIdx
                
                node1 = (i, idx1)
                node2 = (i+1, idx2)
                
                union(node1, node2)
                
    # Group by root
    groups = {}
    for i in range(num_images):
        n_feats = len(keypoints_list[i])
        for j in range(n_feats):
            node = (i, j)
            # Only consider nodes that were part of at least one match?
            # Or all nodes? 
            # If a node was never matched, it's a track of length 1.
            # Usually we filter short tracks (len < 2).
            if node in parent:
                root = find(node)
                groups.setdefault(root, []).append(node)
                
    # Create final track array
    # Filter tracks with length >= 2
    valid_groups = [nodes for nodes in groups.values() if len(nodes) >= 2]
    
    F = len(valid_groups)
    track = np.full((num_images, F, 2), -1.0)
    
    for f_idx, nodes in enumerate(valid_groups):
        for (img_idx, k_idx) in nodes:
            track[img_idx, f_idx, :] = keypoints_list[img_idx][k_idx]
            
    # Output requires (N, F, 2)
    return track