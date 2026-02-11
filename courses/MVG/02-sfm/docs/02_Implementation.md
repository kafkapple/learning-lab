# Structure from Motion (SfM) 구현 가이드 - 완벽 분석 & 튜토리얼

이 문서는 과제 코드를 직접 구현하거나 분석할 때 참고할 수 있도록, **이론적 배경**과 **코드 구현의 핵심**을 단계별로 아주 상세하게 설명하는 가이드입니다. `sfm_implementation_walkthrough_kr`와 `sfm_coding_tutorial_kr`를 통합하여 이론과 실전을 한 번에 볼 수 있게 구성했습니다.

---

## 1. Feature Matching & Tracking (특징점 매칭)

**파일**: `feature.py`

### 1.1 SIFT 매칭 (`MatchSIFT`)
**이론**:
두 이미지의 관계를 알기 위해서는 같은 물체가 어디에 있는지 알아야 합니다. **SIFT (Scale-Invariant Feature Transform)**는 이미지의 크기나 회전이 변해도 잘 찾을 수 있는 강력한 특징점입니다.
매칭할 때 **Lowe's Ratio Test**를 사용합니다. 가장 비슷한 특징점(1순위)과 두 번째로 비슷한 특징점(2순위)의 거리 차이가 크지 않다면, 애매한 매칭이므로 버리는 방식입니다.

**구현 코드**:
```python
def MatchSIFT(loc1, des1, loc2, des2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2) # 가장 가까운 2개 찾기
    
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance: # 1순위 거리가 2순위의 70% 미만일 때만 인정
            good.append(m)
            
    # 매칭된 점들의 좌표만 추출해서 반환
    x1 = np.float32([loc1[m.queryIdx] for m in good])
    x2 = np.float32([loc2[m.trainIdx] for m in good])
    ind1 = np.array([m.queryIdx for m in good])
    return x1, x2, ind1
```

### 1.2 Essential Matrix 추정 (`EstimateE_RANSAC`)
**이론**:
**Essential Matrix ($E$)**는 두 카메라 간의 회전($R$)과 이동($t$) 정보를 담고 있는 $3 \times 3$ 행렬입니다. $x'^T E x = 0$ (Epipolar constraint)식을 만족해야 합니다. 노이즈가 많으므로 **RANSAC**을 사용해 이상치(Outlier)를 제거하며 추정합니다.

**구현 코드**:
```python
def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    # RANSAC을 켜고 E를 찾습니다.
    # focal=1.0, pp=(0,0)인 이유는 이미 입력 좌표가 정규화되어 들어오기 때문입니다.
    E, mask = cv2.findEssentialMat(x1, x2, focal=1.0, pp=(0, 0), 
                                   method=cv2.RANSAC, 
                                   prob=0.999, threshold=ransac_thr)
    return E, mask.ravel().astype(bool)
```

---

## 2. Initialization (초기화)

**파일**: `camera_pose.py`

### 2.1 포즈 분해 (`GetCameraPoseFromE`)
**이론**:
$E$ 행렬을 SVD 분해하면 수학적으로 가능한 4가지의 카메라 포즈 조합 $(R, t)$가 나옵니다.
1. $(R_1, t)$
2. $(R_1, -t)$
3. $(R_2, t)$
4. $(R_2, -t)$

**구현 코드**:
```python
def GetCameraPoseFromE(E):
    R1, R2, t = cv2.decomposeEssentialMat(E)
    # 4가지 경우의 수 조합
    R_set = np.array([R1, R1, R2, R2])
    C_set = []
    # t는 이동 벡터고, 실제 카메라 위치(Center) C = -R^T * t 입니다.
    for i, R, T_vec in zip(range(4), R_set, [t, -t, t, -t]):
        C_set.append(-R.T @ T_vec)
    return R_set, np.array(C_set)
```

### 2.2 삼각측량 (`Triangulation`)
**이론**:
두 카메라의 위치($P_1, P_2$)와 2D 매칭점($x_1, x_2$)을 알 때, 두 시선이 만나는 3D 교점 $X$를 계산합니다 (DLT 방법).

**구현 코드**:
```python
def Triangulation(P1, P2, track1, track2):
    # OpenCV 함수 이용 (결과는 4D 호모지니어스 좌표)
    pts4D = cv2.triangulatePoints(P1, P2, track1.T, track2.T)
    # 3D로 변환: (X,Y,Z,W) -> (X/W, Y/W, Z/W)
    pts3D = pts4D[:3] / pts4D[3]
    return pts3D.T
```

### 2.3 Cheirality Check (`EvaluateCheirality`)
**이론**:
4가지 포즈 중 진짜 정답은 **"복원된 3D 점이 두 카메라 모두의 앞($Z > 0$)에 있는 경우"**입니다.

**구현 코드**:
```python
def EvaluateCheirality(P1, P2, X):
    # 점들을 동차좌표로 변환 [X, Y, Z, 1]
    X_h = np.hstack([X, np.ones((X.shape[0], 1))])
    # 카메라 좌표계로 투영
    x1_cam = (P1 @ X_h.T).T
    x2_cam = (P2 @ X_h.T).T
    # Z값이 0보다 커야 카메라 앞입니다.
    valid = (x1_cam[:, 2] > 0) & (x2_cam[:, 2] > 0)
    return valid
```

---

## 3. Registering New Images (확장)

**파일**: `pnp.py`

### 3.1 PnP (Perspective-n-Point) (`PnP_RANSAC`)
**이론**:
이미 3D 좌표를 알고 있는 점들($X$)과, 새로운 이미지에서의 2D 좌표($x$)가 주어졌을 때, 새로운 카메라의 포즈($R, t$)를 구하는 문제입니다.

**구현 코드**:
```python
def PnP_RANSAC(X, x, ransac_n_iter, ransac_thr):
    # solvePnPRansac 함수 하나면 끝납니다.
    # Intrinsic K는 Identity 행렬을 넣습니다 (이미 정규화됨).
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        X, x, np.eye(3), None, 
        iterationsCount=ransac_n_iter, reprojectionError=ransac_thr
    )
    # rvec(회전벡터)를 R(행렬)로 변환: cv2.Rodrigues
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec # 카메라 중심 좌표 계산
    return R, C.ravel(), inliers
```

---

## 4. Reconstruction & Optimization (재구성 및 최적화)

**파일**: `reconstruction.py`

### 4.1 Bundle Adjustment (`RunBundleAdjustment`)
**이론**:
SfM의 꽃입니다. 편차를 줄이기 위해 **모든 카메라 포즈($P_1...P_N$)**와 **모든 3D 점($X_1...X_M$)**을 한꺼번에 최적화합니다.

**구현 포인트**:
1.  **파라미터 묶기 (`x0`)**: 최적화 함수(`least_squares`) 입력을 위해 [카메라1, 카메라2..., 점1, 점2...] 순서로 1차원 배열 생성.
2.  **오차 함수 정의 (`fun`)**:
    -   배열 `x0`를 다시 카메라와 점으로 분해.
    -   각 카메라에서 점들을 투영(Project).
    -   `투영된 점 - 실제 관측 점` (Residual) 계산.
3.  **실행**: `scipy.optimize.least_squares(fun, x0)`

---

## 5. Main Loop Implementation

**파일**: `hw4.py`

이제 위 함수들을 조립하여 전체 SfM 파이프라인을 돌립니다.

**작성 순서**:
1.  **초기화**: `EstimateCameraPose` 호출해서 0번, 1번 이미지 처리.
2.  **반복문 (for i in 2..N)**:
    -   **PnP**: 현재 이미지($i$)에서 보이는 3D 점들을 모아서 `PnP_RANSAC` 호출 -> 카메라 $P_i$ 구함.
    -   **확장(Triangulation)**: 카메라 $P_i$가 생겼으니, 예전 카메라들($P_0...P_{i-1}$)과 협력해서 예전엔 못 만들었던 3D 점들을 추가로 복원(`Triangulation`).
    -   **최적화(BA)**: `RunBundleAdjustment` 호출해서 지금까지 만든 거 전체 보정.
    -   **저장**: 결과를 `.ply`로 저장.
