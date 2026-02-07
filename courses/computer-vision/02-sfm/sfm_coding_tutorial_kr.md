# SfM 코드 구현 핸즈온 튜토리얼 (Step-by-Step)

이 문서는 빈 껍데기 파일(`TODO`가 있는 원본 파일)을 하나씩 열어서, 실제 코드를 어떻게 작성해야 하는지 단계별로 설명합니다. 단순히 정답을 붙여넣는 것이 아니라, **왜 이 코드가 필요한지** 이해하면서 하나씩 완성해 봅시다.

---

## 🏗️ 1단계: 기초 도구 만들기 (`utils.py`)

가장 먼저 수학적 도구를 만듭니다. SfM에서는 회전 행렬(Rotation Matrix, $3 \times 3$)과 쿼터니언(Quaternion, $4 \times 1$)을 왔다 갔다 해야 할 일이 많습니다. (최적화할 때 파라미터 개수를 줄이기 위해 쿼터니언을 주로 씁니다.)

**할 일**: `Rotation2Quaternion`과 `Quaternion2Rotation` 함수 채우기.

**작성 코드 예시**:
```python
from scipy.spatial.transform import Rotation

def Rotation2Quaternion(R):
    # R을 쿼터니언으로 변환
    r = Rotation.from_matrix(R)
    return r.as_quat()

def Quaternion2Rotation(q):
    # 쿼터니언을 R로 변환
    r = Rotation.from_quat(q)
    return r.as_matrix()
```
*설명: 복잡한 수식을 직접 짜는 것보다 `scipy` 라이브러리를 쓰는 것이 훨씬 정확하고 안전합니다.*

---

## 🔍 2단계: 특징점 편 (`feature.py`)

이제 사진을 보고 "같은 점"을 찾는 기능을 만듭니다.

### 2-1. `MatchSIFT` 구현
**목표**: 두 이미지에서 뽑은 SIFT 특징점 `des1`, `des2`를 비교해서 짝을 찾습니다.

**작성 코드**:
```python
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2) # 가장 가까운 2개 검색
    
    good = []
    # Lowe's Ratio Test: 1등이 2등보다 압도적으로(0.7배) 가깝지 않으면 버림
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
            
    # 매칭된 점들의 좌표만 추출해서 반환
    x1 = np.float32([loc1[m.queryIdx] for m in good])
    x2 = np.float32([loc2[m.trainIdx] for m in good])
    ind1 = np.array([m.queryIdx for m in good])
    return x1, x2, ind1
```

### 2-2. `EstimateE_RANSAC` 구현
**목표**: 매칭된 점들(`x1`, `x2`)을 보고 두 카메라 사이의 관계(Essential Matrix)를 구합니다.

**작성 코드**:
```python
    # RANSAC을 켜고 E를 찾습니다.
    # focal=1.0, pp=(0,0)인 이유는 이미 입력 좌표가 정규화되어 들어오기 때문입니다.
    E, mask = cv2.findEssentialMat(x1, x2, focal=1.0, pp=(0, 0), 
                                   method=cv2.RANSAC, 
                                   prob=0.999, threshold=ransac_thr)
    # mask: 어떤 점이 정상(inlier)이고 어떤 점이 노이즈인지 알려줍니다.
    inlier = mask.ravel().astype(bool)
    return E, inlier
```

---

## 📸 3단계: 초기 포즈 잡기 (`camera_pose.py`)

첫 두 장의 사진 위치를 결정하는 아주 중요한 단계입니다.

### 3-1. `GetCameraPoseFromE` 구현
**목표**: `E` 행렬에서 가능한 4가지 $(R, t)$ 조합을 뽑아냅니다. 수학적으로 $E$를 분해하면 4가지 경우가 나옵니다.

**작성 코드**:
```python
    R1, R2, t = cv2.decomposeEssentialMat(E)
    # 4가지 경우의 수 조합
    R_set = np.array([R1, R1, R2, R2])
    C_set = []
    # t는 이동 벡터고, 실제 카메라 위치(Center) C = -R^T * t 입니다.
    for i, R, T_vec in zip(range(4), R_set, [t, -t, t, -t]):
        C_set.append(-R.T @ T_vec)
    return R_set, np.array(C_set)
```

### 3-2. `Triangulation` 구현
**목표**: 두 카메라($P1, P2$)에서 쏜 광선이 만나는 3D 점 찾기.

**작성 코드**:
```python
    # OpenCV 함수 이용 (결과는 4D 호모지니어스 좌표)
    pts4D = cv2.triangulatePoints(P1, P2, track1.T, track2.T)
    # 3D로 변환: (X,Y,Z,W) -> (X/W, Y/W, Z/W)
    pts3D = pts4D[:3] / pts4D[3]
    return pts3D.T
```

### 3-3. `EvaluateCheirality` 구현
**목표**: 4가지 포즈 중, 점들이 "카메라 뒤"가 아니라 "카메라 앞"에 있는 진짜 포즈 찾기.

**작성 코드**:
```python
    # 점들을 동차좌표로 변환 [X, Y, Z, 1]
    X_h = np.hstack([X, np.ones((X.shape[0], 1))])
    # 카메라 좌표계로 투영
    x1_cam = (P1 @ X_h.T).T
    x2_cam = (P2 @ X_h.T).T
    # Z값이 0보다 커야 카메라 앞입니다.
    valid_index = (x1_cam[:, 2] > 0) & (x2_cam[:, 2] > 0)
    return valid_index
```

### 3-4. `EstimateCameraPose` 최종 조립
**할 일**: 위 함수들을 순서대로 호출합니다.
1. `matches`로 `E` 구하기
2. `E` 분해해서 4개 후보 만들기
3. 각 후보마다 `Triangulation` 해보고 `Cheirality` 검사해서 점들이 가장 많이 살아남는 후보 선택.

---

## 📍 4단계: 카메라 추가하기 (`pnp.py`)

이미 3D 점들이 만들어져 있을 때, 새로운 카메라 위치를 알아내는 과정입니다.

### 4-1. `PnP_RANSAC` 구현
**목표**: 3D 점 `X`와 2D 점 `x`를 매칭해서 카메라 $(R, C)$ 찾기.

**작성 코드**:
```python
    # solvePnPRansac 함수 하나면 끝납니다.
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        X, x, np.eye(3), None, 
        iterationsCount=ransac_n_iter, reprojectionError=ransac_thr
    )
    # rvec(벡터) -> R(행렬) 변환
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec # 카메라 중심 좌표 계산
    return R, C.ravel(), inliers
```

---

## 🔧 5단계: 최적화 (`reconstruction.py`)

### 5-1. `RunBundleAdjustment` 구현
**목표**: 모든 카메라와 점들을 미세 조정해서 오차 줄이기. 가장 어렵지만 핵심인 부분입니다.

**작성 가이드**:
1.  **파라미터 묶기 (`x0`)**: 최적화 함수(`least_squares`)는 입력 변수가 1차원 배열이어야 합니다. 그래서 [카메라1, 카메라2..., 점1, 점2...] 순서로 아주 긴 배열을 만듭니다.
2.  **오차 함수 정의 (`fun`)**:
    -   긴 배열 `x0`를 다시 카메라와 점으로 뜯어냅니다.
    -   각 카메라에서 점들을 투영해 봅니다. (Project)
    -   `투영된 점 - 실제 관측 점` (Residual)을 계산해서 리턴합니다.
3.  **실행**: `scipy.optimize.least_squares(fun, x0)`

---

## 🚀 6단계: 메인 루프 돌리기 (`hw4.py`)

이제 `hw4.py`의 `TODO`를 채워서 전체를 돌립니다.

**작성 순서**:
1.  **초기화**: `EstimateCameraPose` 호출해서 0번, 1번 이미지 처리.
2.  **반복문 (for i in 2..N)**:
    -   **PnP**: 현재 이미지($i$)에서 보이는 3D 점들을 모아서 `PnP_RANSAC` 호출 -> 카메라 $P_i$ 구함.
    -   **확장(Triangulation)**: 카메라 $P_i$가 생겼으니, 예전 카메라들($P_0...P_{i-1}$)과 협력해서 예전엔 못 만들었던 3D 점들을 추가로 복원(`Triangulation`).
    -   **최적화(BA)**: `RunBundleAdjustment` 호출해서 지금까지 만든 거 전체 보정.
    -   **저장**: 결과를 `.ply`로 저장.

---

이제 이 가이드를 따라 `hw4.py`를 실행해 보세요!
`python hw4.py`
