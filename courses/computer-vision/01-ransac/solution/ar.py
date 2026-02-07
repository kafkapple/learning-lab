import numpy as np
import cv2
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

#Write script for Q4.1

# Open video streams
book_vid = cv2.VideoCapture('../data/book.mov')
ar_source_vid = cv2.VideoCapture('../data/ar_source.mov')
# Load reference image
cv_cover = cv2.imread('../data/cv_cover.jpg')

# Get dimensions of the reference image for aspect ratio correction
h_ref, w_ref, _ = cv_cover.shape

# Get total number of frames
# Not all opencv versions support this, but we can try
try:
    total_frames = int(ar_source_vid.get(cv2.CAP_PROP_FRAME_COUNT))
except:
    total_frames = 1000 # Assume a large number if property is not available


frame_idx = 0

while(book_vid.isOpened()):
    ret_book, frame_book = book_vid.read()
    ret_ar, frame_ar = ar_source_vid.read()

    if not ret_book:
        break

    if not ret_ar:
        # If the AR source video ends, loop it
        ar_source_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_ar, frame_ar = ar_source_vid.read()

    # --- Aspect Ratio Correction ---
    h_ar, w_ar, _ = frame_ar.shape
    
    # Calculate aspect ratios
    ar_ref = w_ref / h_ref
    ar_vid = w_ar / h_ar

    if ar_vid > ar_ref:
        # Video is wider than reference, crop width
        new_w = int(h_ar * ar_ref)
        start_w = (w_ar - new_w) // 2
        template = frame_ar[:, start_w:start_w + new_w]
    else:
        # Video is taller than reference, crop height
        new_h = int(w_ar / ar_ref)
        start_h = (h_ar - new_h) // 2
        template = frame_ar[start_h:start_h + new_h, :]

    # Resize template to match reference size for consistency
    template = cv2.resize(template, (w_ref, h_ref))

    # Match features between the reference cover and the book in the video
    matches, locs1, locs2 = matchPics(cv_cover, frame_book)

    # Check if we have enough matches
    if matches.shape[0] < 10: # Threshold for minimum matches
        cv2.imshow('Augmented Reality', frame_book)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Arrange points for RANSAC
    x1 = locs1[matches[:, 0], 0:2][:, [1, 0]]
    x2 = locs2[matches[:, 1], 0:2][:, [1, 0]]

    # Compute homography
    bestH, _ = computeH_ransac(x1, x2)

    # Composite the AR source frame onto the book frame
    if bestH is not None:
        composite_img = compositeH(bestH, template, frame_book)
        cv2.imshow('Augmented Reality', composite_img)
    else:
        cv2.imshow('Augmented Reality', frame_book)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1
    print(f"Processing frame {frame_idx}")


# Release everything when done
book_vid.release()
ar_source_vid.release()
cv2.destroyAllWindows()
