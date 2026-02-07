import numpy as np
import cv2
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac

#Write script for Q4.2x

# Read the images
pano_left = cv2.imread('../data/pano_left.jpg')
pano_right = cv2.imread('../data/pano_right.jpg')

# Match features between the two images
matches, locs1, locs2 = matchPics(pano_left, pano_right)

# Arrange points for RANSAC
# We want to warp the right image to the left image's perspective
# So, x1 are points in left, x2 are points in right
x1 = locs1[matches[:, 0], 0:2][:, [1, 0]]
x2 = locs2[matches[:, 1], 0:2][:, [1, 0]]

# Compute homography
H, _ = computeH_ransac(x1, x2)

# Set the size of the panorama
pano_width = pano_left.shape[1] + pano_right.shape[1]
pano_height = pano_left.shape[0]

# Warp the right image
warped_right = cv2.warpPerspective(pano_right, H, (pano_width, pano_height))

# Create the panorama
panorama = warped_right.copy()
# Place the left image on the left side of the panorama
panorama[0:pano_left.shape[0], 0:pano_left.shape[1]] = pano_left

# A simple blending can be done by taking the non-black pixels from the left image
# This is a bit more robust than just pasting
# Create a mask for the left image
mask = np.zeros((warped_right.shape[0], warped_right.shape[1]), dtype=np.uint8)
mask[0:pano_left.shape[0], 0:pano_left.shape[1]] = 255
# Invert the mask to get the area where the left image is
inv_mask = cv2.bitwise_not(mask)
# Black-out the area of the left image in the warped image
warped_right_masked = cv2.bitwise_and(warped_right, warped_right, mask=inv_mask)
# Take only the left image area
pano_left_masked = cv2.bitwise_and(panorama, panorama, mask=mask)
# Add the two images
panorama_final = cv2.add(warped_right_masked, pano_left_masked)

# Display the final panorama
cv2.imshow('Panorama', panorama_final)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the resulting image
cv2.imwrite('../result/panorama.png', panorama_final)
