import numpy as np
import cv2
from matchPics import matchPics
from matplotlib import pyplot as plt
import scipy.ndimage

#Read the image and convert to grayscale, if necessary
img = cv2.imread('../data/cv_cover.jpg')

#is_gray = len(img.shape) == 2
#if is_gray == False:
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


matches_per_rotation = []
#Loop through rotations
for i in range(36):
    angle = i*10
    #Rotate image
    rot_img = scipy.ndimage.rotate(img, angle, reshape=False)
    #Compute features, descriptors and matches
    matches, locs1, locs2 = matchPics(img, rot_img)
    matches_per_rotation.append(len(matches))
    #Update histogram
    print("Rotation: %d, Matches: %d" % (angle, len(matches)))

#Display histogram
degrees = [i*10 for i in range(36)]
plt.bar(degrees, matches_per_rotation)
plt.xlabel("Rotation (degrees)")
plt.ylabel("Number of Matches")
plt.title("BRIEF descriptor rotation variance")
plt.show()

