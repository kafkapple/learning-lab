# 📘 MVG Assignment 2: 완벽 가이드 (Code & Theory)

이 문서는 **Computer Vision Assignment 2**의 모든 코드를 **한 줄 한 줄** 뜯어보며, 이론과 구현을 동시에 마스터할 수 있도록 작성된 "All-in-One 교육용 노트북"입니다. 

`/Users/joon/dev/MVG/assgn2/solution/` 디렉토리의 실제 코드를 기반으로 작성되었습니다.

---

## 1. Feature Matching (특징점 매칭)

가장 먼저 할 일은 두 이미지 사이의 **연결고리**를 찾는 것입니다. 
`matchPics.py`는 두 이미지에서 특징점(Corner)을 찾고, 서로 닮은 점들을 매칭해줍니다.

### 📝 코드 분석: `matchPics.py`

```python
import numpy as np
import cv2
import skimage.color
from helper import briefMatch, computeBrief, corner_detection

def matchPics(I1, I2):
    """
    두 이미지 I1, I2에서 특징점을 검출하고 매칭 결과를 반환합니다.
    """
    # 1. Grayscale 변환
    # 컬러 이미지는 정보량이 많아 특징점 검출에 방해가 될 수 있으므로 흑백으로 변환합니다.
    # cv2.COLOR_BGR2GRAY: Blue-Green-Red 순서의 이미지를 Gray로 변환
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # 2. 특징점(Corner) 검출
    # corner_detection 함수(FAST 알고리즘 등 사용)는 코너 점들의 좌표를 찾아줍니다.
    # locs1, locs2는 (N, 2) 형태의 배열로, [x, y] 좌표를 담고 있습니다.
    locs1 = corner_detection(I1_gray)
    locs2 = corner_detection(I2_gray)

    # 3. Descriptor(기술자) 추출
    # 단순히 점의 위치만으로는 누가 누구랑 짝인지 알 수 없습니다.
    # computeBrief는 각 점 주변의 픽셀 패턴을 0과 1의 이진 벡터(Descriptor)로 요약해줍니다.
    # desc1: I1 특징점들의 특징 벡터들
    # locs1: 유효한 Descriptor를 가진 특징점들의 위치 (경계선 근처 점들은 버려질 수 있음)
    desc1, locs1 = computeBrief(I1_gray, locs1)
    desc2, locs2 = computeBrief(I2_gray, locs2)

    # 4. 특징점 매칭
    # briefMatch는 두 이미지의 Descriptor들을 비교(Hamming Distance)하여
    # 가장 유사한 쌍을 찾아줍니다.
    # matches: 매칭된 인덱스 쌍 (M, 2). 예: [ [0, 5], [1, 2], ... ]
    #          0번 특징점(I1)이 5번 특징점(I2)과 매칭되었다는 뜻.
    matches = briefMatch(desc1, desc2)

    return matches, locs1, locs2
```

---

## 2. Homography Estimation (호모그래피 계산)

매칭된 점들이 주어졌을 때, 한 평면을 다른 평면으로 변환하는 **Homography 행렬 ($3 \times 3$)**을 구해야 합니다. 이는 `planarH.py`에 구현되어 있습니다.

### 2.1 Basic Homography via DLT (`computeH`)

**이론**: 점 $x_1$과 $x_2$가 매칭되었다면, $x_1 \equiv H x_2$ 관계가 성립합니다. 이를 $Ax=0$ 형태의 선형 방정식으로 만들고, SVD(특이값 분해)를 통해 $H$를 구합니다. 이것이 **DLT(Direct Linear Transform)**입니다.

```python
def computeH(x1, x2):
    # Q3.6
    # x1, x2: 매칭된 점들의 좌표 (N, 2)
    # 목표: x1 ~ H * x2 를 만족하는 H 구하기

    A = []
    # 모든 점 쌍에 대해 방정식 행렬 A를 구성합니다.
    for i in range(x1.shape[0]):
        p1 = x1[i] # Target Image의 점 (u, v)
        p2 = x2[i] # Source Image의 점 (x, y)
        
        # DLT 방정식 구조:
        # [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        # [0, 0, 0, -x, -y, -1, v*x, v*y, v]
        A.append([-p2[0], -p2[1], -1, 0, 0, 0, p1[0]*p2[0], p1[0]*p2[1], p1[0]])
        A.append([0, 0, 0, -p2[0], -p2[1], -1, p1[1]*p2[0], p1[1]*p2[1], p1[1]])
    
    A = np.array(A)
    
    # SVD 수행 (Singular Value Decomposition)
    # A * h = 0 의 해는 A의 가장 작은 singular value에 대응하는 right singular vector입니다.
    _, _, Vh = np.linalg.svd(A)
    
    # Vh의 마지막 행이 바로 우리가 찾는 H의 요소들입니다.
    H2to1 = Vh[-1, :].reshape(3, 3)
    return H2to1
```

### 2.2 Normalized Homography (`computeH_norm`)

**이론**: 픽셀 좌표값(예: 1920, 1080)을 그대로 DLT에 넣으면 숫자 단위가 너무 커서 계산 오차가 발생합니다. 따라서 점들의 중심을 (0,0)으로 옮기고, 평균 거리가 $\sqrt{2}$가 되도록 **정규화(Normalization)**한 뒤 $H$를 계산해야 합니다.

```python
def computeH_norm(x1, x2):
    # Q3.7
    
    # 1. 중심점(Centroid) 계산
    x1_centroid = np.mean(x1, axis=0)
    x2_centroid = np.mean(x2, axis=0)

    # 2. 중심을 원점으로 이동 (Shift)
    x1_shifted = x1 - x1_centroid
    x2_shifted = x2 - x2_centroid

    # 3. 스케일 계산 (Average Distance가 sqrt(2)가 되도록)
    avg_dist1 = np.mean(np.sqrt(np.sum(x1_shifted**2, axis=1)))
    avg_dist2 = np.mean(np.sqrt(np.sum(x2_shifted**2, axis=1)))

    s1 = np.sqrt(2) / avg_dist1
    s2 = np.sqrt(2) / avg_dist2

    # 4. 변환 행렬 T 구성 (Similarity Transform)
    # T = Scale Matrix * Shift Matrix
    T1 = np.array([[s1, 0, -s1*x1_centroid[0]],
                   [0, s1, -s1*x1_centroid[1]],
                   [0, 0, 1]])

    T2 = np.array([[s2, 0, -s2*x2_centroid[0]],
                   [0, s2, -s2*x2_centroid[1]],
                   [0, 0, 1]])
    
    # 5. 점들에 T 적용 (Homogeneous 좌표계로 변환 후 적용)
    x1_homo = np.hstack((x1, np.ones((x1.shape[0], 1)))) # (x, y, 1) 만들기
    x2_homo = np.hstack((x2, np.ones((x2.shape[0], 1))))

    # 정규화된 좌표 x_norm = T * x
    x1_norm = (T1 @ x1_homo.T).T
    x2_norm = (T2 @ x2_homo.T).T

    # 6. 정규화된 점들로 Homography 계산
    H_norm = computeH(x1_norm[:, :2], x2_norm[:, :2])
    
    # 7. Denormalization (원래 좌표계의 H로 복원)
    # H = inv(T1) * H_norm * T2
    H2to1 = np.linalg.inv(T1) @ H_norm @ T2
    return H2to1
```

### 2.3 RANSAC (`computeH_ransac`)

**이론**: 매칭 결과에는 오류(Outlier)가 반드시 섞여 있습니다. **RANSAC**은 무작위로 소수의 데이터를 뽑아 모델을 만들고, 다수결로 검증하여 최적의 모델을 찾는 알고리즘입니다.

```python
def computeH_ransac(x1, x2):
    # Q3.8
    max_iters = 1000  # 반복 횟수
    inlier_tol = 2.5  # Inlier로 인정할 최대 거리 오차 (픽셀 단위)
    bestH2to1 = None  # 최고의 H를 저장할 변수
    inliers = None    # 최고의 Inlier 인덱스 저장
    max_inliers = 0   # 발견된 최대 Inlier 개수

    num_points = x1.shape[0]

    for _ in range(max_iters):
        # 1. 4개의 점을 무작위로 선택 (Homography 계산 최소 조건)
        indices = np.random.choice(num_points, 4, replace=False)
        p1 = x1[indices]
        p2 = x2[indices]
        
        # 2. 선택된 4개 점으로 H 계산
        H = computeH_norm(p1, p2)

        # 3. 모든 점을 변환해보고 에러 측정
        x2_homo = np.hstack((x2, np.ones((num_points, 1))))
        
        # x2를 H로 변환 -> 예측된 x1 위치 (x1_proj)
        x1_proj_homo = (H @ x2_homo.T).T
        # Homogeneous 좌표 (x, y, w)를 (x/w, y/w)로 변환
        x1_proj = x1_proj_homo[:, :2] / x1_proj_homo[:, 2, np.newaxis]
        
        # 4. 실제 x1 좌표와의 거리(에러) 계산
        dist = np.sqrt(np.sum((x1 - x1_proj)**2, axis=1))
        
        # 5. Inlier 개수 세기 (에러가 허용치보다 작은 점들)
        current_inliers = np.where(dist < inlier_tol)[0]
        
        # 6. 신기록 갱신?
        if len(current_inliers) > max_inliers:
            max_inliers = len(current_inliers) # 기록 경신
            inliers = current_inliers          # Inlier 인덱스 저장
            bestH2to1 = H                      # 최고의 H 저장
            
    return bestH2to1, inliers
```

---

## 3. Image Compositing (이미지 합성)

구해진 $H$를 이용해 이미지를 변형하고 합성하는 함수입니다.

### 📝 코드 분석: `compositeH` (`planarH.py`)

```python
def compositeH(H2to1, template, img):
    # template: 덮어씌울 이미지 (예: 해리포터 표지)
    # img: 배경 이미지 (예: 책상 위 책 사진)
    
    # Warping에는 역행렬이 필요합니다. 
    # (이미지 A를 B로 보낼 때, B의 각 픽셀이 A의 어디에서 왔는지 찾아야 색칠할 수 있기 때문)
    # H2to1은 img -> template 방향이므로, template -> img 방향인 역행렬을 구합니다.
    H_inv = np.linalg.inv(H2to1)

    # 1. 마스크 생성
    # template 크기와 똑같은 흰색(255) 마스크를 만듭니다.
    mask = np.ones(template.shape, dtype=np.uint8) * 255

    # 2. 마스크 변형 (Warp)
    # 마스크를 H_inv를 이용해 배경 이미지(img) 시점으로 변형시킵니다.
    # 이렇게 하면 img 상에서 template이 들어갈 영역만 흰색이 됩니다.
    warped_mask = cv2.warpPerspective(mask, H_inv, (img.shape[1], img.shape[0]))

    # 3. 템플릿 변형 (Warp)
    # 실제 template 이미지도 똑같이 변형시킵니다.
    warped_template = cv2.warpPerspective(template, H_inv, (img.shape[1], img.shape[0]))

    # 4. 합성 (Compositing)
    composite_img = img.copy()
    
    # 배경 이미지에서 템플릿이 들어갈 자리를 구멍 냅니다 (검은색 0으로 채움)
    composite_img[warped_mask > 0] = 0
    
    # 구멍 난 자리에 변형된 템플릿을 채워 넣습니다.
    composite_img += warped_template
    
    return composite_img
```

---

## 4. 응용 프로그램 (Application)

위의 함수들을 조립하여 실제 프로그램을 만듭니다.

### 4.1 HarryPotterize (`HarryPotterize.py`)

책상 위의 책(`cv_desk`)을 해리포터 책(`hp_cover`)으로 바꿉니다. 기준이 되는 책 표지 이미지(`cv_cover`)를 이용해 좌표를 찾습니다.

```python
# HarryPotterize.py 주요 로직 설명

# 1. 이미지 읽기
cv_desk = cv2.imread('../data/cv_desk.png')   # 배경 (Target)
cv_cover = cv2.imread('../data/cv_cover.jpg') # 기준 표지 (Source 1)
hp_cover = cv2.imread('../data/hp_cover.jpg') # 바꿀 표지 (Source 2)

# 2. 특징점 매칭 (기준 표지 <-> 책상 위 책)
matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

# 3. RANSAC을 위한 좌표 정리
# matchPics는 (row, col) = (y, x) 순서로 주지만, 
# Homography 계산은 (x, y) 좌표계를 쓰므로 순서를 바꿔줍니다 ([1, 0]).
x1 = locs1[matches[:, 0], 0:2] # 표지 좌표
x2 = locs2[matches[:, 1], 0:2] # 책상 좌표
x1 = x1[:, [1, 0]] 
x2 = x2[:, [1, 0]]

# 4. Homography 계산
# cv_cover(x1)를 cv_desk(x2)로 보내는 행렬 H를 구합니다.
bestH, _ = computeH_ransac(x1, x2)

# 5. hp_cover 리사이즈
# hp_cover를 cv_cover와 똑같은 크기로 만듭니다.
# 이렇게 하면 cv_cover -> cv_desk 로 가는 H를 hp_cover에도 그대로 쓸 수 있습니다.
hp_cover_resized = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

# 6. 합성
composite_img = compositeH(bestH, hp_cover_resized, cv_desk)
```

### 4.2 Panorama Sticthing (`panaroma.py`)

두 장의 사진을 이어 붙이는 코드입니다. (유저 분이 수정 요청하셨던 그 부분!)

```python
# panaroma.py 주요 로직 설명

# 1. 이미지 읽기
pano_left = cv2.imread('../data/pano_left.jpg')
pano_right = cv2.imread('../data/pano_right.jpg')

# 2. 매칭
matches, locs1, locs2 = matchPics(pano_left, pano_right)

# 3. 좌표 정리 및 H 계산
# 오른쪽(x2)을 왼쪽(x1) 시점으로 보낼 것이므로, H는 Right -> Left 변환입니다.
x1 = locs1[matches[:, 0], 0:2][:, [1, 0]]
x2 = locs2[matches[:, 1], 0:2][:, [1, 0]]
H, _ = computeH_ransac(x1, x2)

# 4. 캔버스 크기 설정
pano_width = pano_left.shape[1] + pano_right.shape[1]
pano_height = pano_left.shape[0]

# 5. 오른쪽 이미지 변형 (Warping)
warped_right = cv2.warpPerspective(pano_right, H, (pano_width, pano_height))

# 6. 합성 (Bug Fix 반영)
# 왼쪽 이미지가 들어갈 영역의 마스크를 만듭니다 (1채널!!).
mask = np.zeros((warped_right.shape[0], warped_right.shape[1]), dtype=np.uint8)
mask[0:pano_left.shape[0], 0:pano_left.shape[1]] = 255

# 마스크 반전 (오른쪽 이미지가 보일 영역)
inv_mask = cv2.bitwise_not(mask)

# 오른쪽 이미지에서 왼쪽 이미지가 덮일 부분을 지웁니다.
warped_right_masked = cv2.bitwise_and(warped_right, warped_right, mask=inv_mask)

# 왼쪽 이미지는 그대로 가져옵니다. (캔버스 크기에 맞춰진 상태가 아니므로 복사해서 넣거나 해야 함)
# 아래는 간단히 panorama 캔버스를 만들어 합치는 방식입니다.
panorama = warped_right.copy()
# 왼쪽 자리는 비워두고
panorama[mask > 0] = 0 
# 왼쪽 이미지 투하
panorama[0:pano_left.shape[0], 0:pano_left.shape[1]] = pano_left

# 최종 결과는 warped_right_masked + panorama 식으로 합쳐집니다.
# (본 코드에서는 더 정교한 블렌딩을 위해 bitwise 연산을 사용했습니다)
```

---

이 문서를 통해 각 함수의 내부 동작 원리와 실제 코드가 어떻게 연결되는지 이해하실 수 있기를 바랍니다! 🚀
