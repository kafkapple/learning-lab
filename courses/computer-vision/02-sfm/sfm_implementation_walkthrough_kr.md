# Structure from Motion (SfM) 단계별 상세 구현 가이드

이 문서는 과제 코드를 직접 구현(또는 분석)할 때 참고할 수 있도록, **이론적 배경**과 **코드 구현의 핵심**을 단계별로 아주 상세하게 설명하는 가이드입니다.

---

## 1. Feature Matching & Tracking (특징점 매칭)

**파일**: `feature.py`

### 1.1 SIFT 매칭 (`MatchSIFT`)
**이론**:
두 이미지의 관계를 알기 위해서는 같은 물체가 어디에 있는지 알아야 합니다. **SIFT (Scale-Invariant Feature Transform)**는 이미지의 크기나 회전이 변해도 잘 찾을 수 있는 강력한 특징점입니다.
매칭할 때 **Lowe's Ratio Test**를 사용합니다. 가장 비슷한 특징점(1순위)과 두 번째로 비슷한 특징점(2순위)의 거리 차이가 크지 않다면, 애매한 매칭이므로 버리는 방식입니다.

**구현 포인트**:
```python
def MatchSIFT(loc1, des1, loc2, des2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2) # 가장 가까운 2개 찾기
    
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance: # 1순위 거리가 2순위의 70% 미만일 때만 인정
            good.append(m)
    # ... 좌표 추출 및 반환 ...
```

### 1.2 Essential Matrix 추정 (`EstimateE_RANSAC`)
**이론**:
**Essential Matrix ($E$)**는 두 카메라 간의 회전($R$)과 이동($t$) 정보를 담고 있는 $3 \times 3$ 행렬입니다. $x'^T E x = 0$ (Epipolar constraint)식을 만족해야 합니다. 노이즈가 많으므로 **RANSAC**을 사용해 이상치(Outlier)를 제거하며 추정합니다.

**구현 포인트**:
```python
def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    # cv2.findEssentialMat 함수가 내부적으로 RANSAC을 수행합니다.
    # focal=1.0, pp=(0,0)인 이유는 입력 좌표(x1, x2)가 이미 정규화(normalized)된 좌표이기 때문입니다.
    E, mask = cv2.findEssentialMat(x1, x2, focal=1.0, pp=(0, 0), 
                                   method=cv2.RANSAC, 
                                   prob=0.999, threshold=ransac_thr)
    return E, mask.ravel().astype(bool)
```

### 1.3 Feature Track 구성 (`BuildFeatureTrack`)
**이론**:
SfM은 여러 장의 이미지를 다룹니다. 이미지 1의 점 A가 이미지 2의 점 B고, 이미지 2의 점 B가 이미지 3의 점 C라면, (A-B-C)는 하나의 **3D 점**을 나타내는 **Track**이 됩니다.

**구현 포인트**:
-   모든 이미지 쌍(또는 인접 쌍)에 대해 매칭을 수행합니다.
-   **Union-Find** 알고리즘이나 그래프 탐색을 통해 연결된 점들을 하나의 그룹(Track)으로 묶습니다.
-   결과 행렬 `track`은 `(N_images, N_features, 2)` 크기이며, 해당 이미지에서 보이지 않는 특징점은 `-1`로 채웁니다.

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

**구현 포인트**:
```python
def GetCameraPoseFromE(E):
    R1, R2, t = cv2.decomposeEssentialMat(E)
    # 4가지 조합을 리스트로 만들어 반환합니다.
    # 주의: t는 translation vector이고, 카메라 중심 C = -R^T * t 입니다.
```

### 2.2 삼각측량 (`Triangulation`)
**이론**:
두 카메라의 위치($P_1, P_2$)와 2D 매칭점($x_1, x_2$)을 알 때, 두 시선이 만나는 3D 교점 $X$를 계산합니다. 이를 **Linear Triangulation** (DLT 방법)이라 합니다.

**구현 포인트**:
```python
def Triangulation(P1, P2, track1, track2):
    # cv2.triangulatePoints는 (4, N) 크기의 homogeneous 좌표를 반환합니다.
    pts4D = cv2.triangulatePoints(P1, P2, track1.T, track2.T)
    # 마지막 성분(w)으로 나누어 (x, y, z, 1) 형태로 만듭니다.
    pts3D = pts4D[:3] / pts4D[3]
    return pts3D.T
```

### 2.3 Cheirality Check (`EvaluateCheirality`)
**이론**:
앞서 구한 4가지 포즈 중 진짜 정답은 **"복원된 3D 점이 두 카메라 모두의 앞($Z > 0$)에 있는 경우"**입니다. 이를 Cheirality 조건이라 합니다.

**구현 포인트**:
```python
def EvaluateCheirality(P1, P2, X):
    # 3D 점을 각 카메라 좌표계로 투영 (P @ X)
    # 투영된 점의 Z좌표(3번째 성분)가 양수인지 확인
    valid = (x1_cam[:, 2] > 0) & (x2_cam[:, 2] > 0)
    return valid
```

---

## 3. Registering New Images (확장)

**파일**: `pnp.py`

### 3.1 PnP (Perspective-n-Point) (`PnP_RANSAC`)
**이론**:
이미 3D 좌표를 알고 있는 점들($X$)과, 새로운 이미지에서의 2D 좌표($x$)가 주어졌을 때, 새로운 카메라의 포즈($R, t$)를 구하는 문제입니다.

**구현 포인트**:
```python
def PnP_RANSAC(X, x, ransac_n_iter, ransac_thr):
    # 이미 정규화된 좌표(x)를 사용하므로 Intrinsic K는 Identity 행렬을 넣습니다.
    # cv2.solvePnPRansac 이용
    success, rvec, tvec, inliers = cv2.solvePnPRansac(X, x, np.eye(3), None, ...)
    # rvec(회전벡터)를 R(행렬)로 변환: cv2.Rodrigues
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec
    return R, C.ravel(), inliers
```

### 3.2 Non-linear Refinement (`PnP_nl`)
**이론**:
PnP로 구한 초기값은 오차가 있을 수 있습니다. 재투영 오차(Reprojection Error)를 최소화하도록 $R, t$를 미세 조정합니다.

**구현 포인트**:
-   `scipy.optimize.least_squares`를 사용하여 `(투영된 x - 실제 x)`를 최소화하는 최적화를 수행합니다.

---

## 4. Reconstruction & Optimization (재구성 및 최적화)

**파일**: `reconstruction.py`

### 4.1 새로운 점 추가 (`Triangulation_nl`)
새로운 카메라가 추가되면, 기존에는 2장 이하에서만 보여서 3D로 만들지 못했던 점들이 이제 3D로 복원 가능해질 수 있습니다. 이를 찾아 삼각측량을 수행합니다. 또한 비선형 최적화(Non-linear Triangulation)로 위치를 다듬습니다.

### 4.2 Bundle Adjustment (`RunBundleAdjustment`)
**이론**:
SfM의 꽃입니다. 지금까지 구한 **모든 카메라 포즈($P_1...P_N$)**와 **모든 3D 점($X_1...X_M$)**을 한꺼번에 최적화합니다.
목적 함수: $\sum \text{reprojection\_error}^2$

**구현 포인트**:
```python
def RunBundleAdjustment(P, X, track):
    # 1. 최적화 변수 벡터(x0) 구성: [카메라파라미터들..., 3D점좌표들...]
    #    카메라는 (Quaternion + Translation) 7개 파라미터로 표현
    # 2. Residual 함수 정의: 모든 관측에 대해 (투영점 - 관측점) 계산
    # 3. least_squares(residual_func, x0) 실행
    # 4. 결과 벡터를 다시 P, X로 분해하여 반환
```

---

## 5. Main Pipeline (전체 흐름)

**파일**: `hw4.py`

1.  **Feature Extraction**: 모든 이미지에서 특징점 추출 및 매칭 (`track` 생성)
2.  **Initialization**:
    -   이미지 1, 2 선택
    -   `EstimateCameraPose`: E 계산 -> 포즈 분해 -> 삼각측량 -> Cheirality 검사 -> 초기 3D 점 생성
3.  **Incremental Loop** (이미지 3부터 끝까지):
    -   **PnP**: 현재 이미지에서 보이는 3D 점들을 이용해 카메라 포즈($P_i$) 추정
    -   **Triangulation**: 새로운 카메라 덕분에 3D로 만들 수 있게 된 점들 추가 복원
    -   **Bundle Adjustment**: 현재까지의 모든 카메라와 점들을 최적화
4.  **Visualization**: 결과를 `.ply` 파일로 저장하여 시각화

이 가이드와 함께 `_solution.py` 코드들을 비교해서 보시면 전체 구조가 명확히 이해되실 것입니다.
