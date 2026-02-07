# Assignment 2: Augmented Reality with Planar Homographies (16-385 Computer Vision)

This document provides a systematic summary of the "Augmented Reality with Planar Homographies" assignment for the 16-385 Computer Vision course (Spring 2024).

---

## 1. Assignment Overview and Objectives

The goal of this assignment is to create an **Augmented Reality (AR)** application that tracks a planar object (a book cover) in images and videos and replaces it with another image or video, using the principles of **Planar Homography**.

### Core Concepts

- **Planar Homography**: A 3x3 transformation matrix that describes the projection of one plane onto another. It is used for correcting image distortion and warping.
- **Homogeneous Coordinates**: A coordinate system that represents 2D points (x, y) as (x, y, 1). It is used to handle complex transformations like translation, rotation, and scaling as a single matrix multiplication.
- **Feature Detection & Matching**: Techniques to automatically find and link corresponding points (e.g., corners) between two images (FAST, BRIEF, Hamming Distance).
- **RANSAC (Random Sample Consensus)**: A robust algorithm for estimating an optimal Homography matrix from data that contains mismatched points (outliers).

---

## 2. Detailed Tasks

### 2.1 Theory Questions

- **Q2.1 Correspondences (10 points)**:
    - Calculate the Degrees of Freedom (DoF) of a Homography matrix.
    - Determine the minimum number of point pairs required to estimate H.
    - Derive the `Ah=0` matrix and analyze its rank.

- **Q2.2 Fun with Homogeneous Coordinates (15 points)**:
    - Least-squares estimation of a conic equation and the minimum number of points required.
    - Matrix representation using a symmetric matrix `Q` and derivation of its transformation under a projection.

- **Q3.1~3.3 Feature Point Analysis (15 points)**:
    - Compare the FAST and Harris Corner Detectors.
    - Compare the BRIEF and Filter-bank based descriptors.
    - Explain the matching principle using Hamming distance and Nearest Neighbors.

### 2.2 Programming Tasks

#### Part 1: Feature Matching and Rotation Test
- **Q3.4 `matchPics.py`**: Find feature points (FAST), compute their descriptors (BRIEF), and visualize the matches between two images.
- **Q3.5 `briefRotTest.py`**: Analyze the rotation invariance limitations of the BRIEF descriptor by plotting a histogram of the number of matches as an image is rotated.

#### Part 2: Homography Calculation and RANSAC
- **Q3.6 `computeH`**: Calculate the Homography matrix `H` from a set of point pairs using the Direct Linear Transform (DLT).
- **Q3.7 `computeH_norm`**: Calculate the `H` matrix after normalizing the points for numerical stability.
- **Q3.8 `computeH_ransac`**: Implement the RANSAC algorithm to find the optimal `H` matrix from noisy data.

#### Part 3: AR Application Implementation
- **Q3.9 `HarryPotterize.py`**: Composite the `hp_cover.jpg` image onto the book cover area in the `cv_desk.png` image using a homography.
- **Q4.1 `ar.py`**: Track the book cover in every frame of a video (`book.mov`) and superimpose frames from another video (`ar_source.mov`) in real-time. (Requires handling aspect ratio).

### 2.3 Extra Credit
- **Q5.1x Panorama**: Stitch two images taken from the same viewpoint but with different rotations (`pano_left.jpg`, `pano_right.jpg`) into a single panoramic image using a homography.

---

## 3. Administrative Details

- **Deadline**: Wednesday, February 21, 2024, 23:59.
- **Submission**: A single zip file containing the complete code and a write-up report.
- **Constraints**: No absolute paths; adhere to the provided function prototypes.
- **File Structure**:
    - `python/`: `ar.py`, `briefRotTest.py`, `HarryPotterize.py`, `matchPics.py`, `planarH.py`
    - `result/`: `ar.avi`
    - `ec/`: (Optional) `panorama.py`, etc.
    - `<AndrewId>.pdf`: Report file.
