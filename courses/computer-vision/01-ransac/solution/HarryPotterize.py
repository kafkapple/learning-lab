import numpy as np
import cv2
import skimage.io 
import skimage.color

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from helper import plotMatches

#Write script for Q3.9

#Read the images
cv_desk = cv2.imread('../data/cv_desk.png')
cv_cover = cv2.imread('../data/cv_cover.jpg')
hp_cover = cv2.imread('../data/hp_cover.jpg')

#Match features between the cover and the desk
matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

# Re-arrange matched points for RANSAC
x1 = locs1[matches[:, 0], 0:2]
x2 = locs2[matches[:, 1], 0:2]
# The points are in (row, col) format, which is (y, x). We need (x, y) for RANSAC
x1 = x1[:, [1, 0]]
x2 = x2[:, [1, 0]]

#Compute homography using RANSAC
bestH, _ = computeH_ransac(x1, x2)

# Warp the hp_cover to the size of the desk
# To do this, we need a homography from hp_cover to cv_desk.
# The homography we computed is from cv_cover to cv_desk.
# We can find the homography from cv_cover to hp_cover (since they are the same size)
# and then chain the homographies.
# H_hp_desk = H_cv_desk * H_hp_cv
# However, a simpler approach for this specific case is to just resize the hp_cover
# to be the same size as cv_cover.
hp_cover_resized = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

#Composite hp_cover onto cv_desk
composite_img = compositeH(bestH, hp_cover_resized, cv_desk)

#Display composite image
cv2.imshow('Composite Image', composite_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
