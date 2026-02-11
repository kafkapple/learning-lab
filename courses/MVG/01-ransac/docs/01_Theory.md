# 과제 2: 이론적 배경 (Theory) - 심화 분석

이 문서는 과제의 핵심 이론인 **Homography**, **DLT(Direct Linear Transform)**, **Normalization**, 그리고 **RANSAC**에 대해 수학적 배경과 함께 상세히 설명합니다.

---

## 1. Planar Homography (평면 호모그래피)

### 1.1 정의
**Homography**는 3D 공간 상의 한 평면(Plane)을 다른 시점(Viewpoint)에서 보았을 때 어떻게 변하는지를 설명하는 $3 \times 3$ 변환 행렬입니다.
수학적으로는 **Projective Transformation(사영 변환)**의 일종으로, 평행선이 보존되지 않고(원근감 효과), 직선은 직선으로 보존됩니다.

$$ \mathbf{x'} \sim H \mathbf{x} $$

여기서 $\mathbf{x} = [x, y, 1]^T$와 $\mathbf{x'} = [x', y', 1]^T$는 동차 좌표(Homogeneous Coordinates)이며, $\sim$는 스케일(scale)을 무시하고 방향이 같다는 의미입니다(Up to scale).

### 1.2 자유도 (Degrees of Freedom)
$H$ 행렬은 $3 \times 3$ 크기로 총 9개의 원소를 가지지만, 전체 스케일이 바뀌어도 같은 변환을 의미하므로 ($kH \sim H$), 실제 **자유도는 8**입니다.
따라서 $H$를 구하기 위해서는 최소 8개의 식, 즉 **4개의 점 쌍(Point Pairs)**이 필요합니다 ($2 \times 4 = 8$).

---

## 2. DLT (Direct Linear Transformation)

점 대응 쌍 $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$ 가 주어졌을 때 $H$를 구하는 가장 표준적인 방법입니다.

### 2.1 수식 유도
관계식 $\mathbf{x}' \sim H \mathbf{x}$ 는 벡터의 외적(Cross Product)을 이용해 다음과 같이 쓸 수 있습니다. 두 벡터가 평행하다면 외적은 0이 됩니다.

$$ \mathbf{x}' \times (H \mathbf{x}) = \mathbf{0} $$

$\mathbf{x}' = [x', y', 1]^T$, $H$의 행 벡터를 $\mathbf{h}_1^T, \mathbf{h}_2^T, \mathbf{h}_3^T$ 라고 할 때, 이를 전개하면 다음과 같은 선형 방정식이 나옵니다.

$$
\begin{pmatrix}
\mathbf{0}^T & -w'_i \mathbf{x}_i^T & y'_i \mathbf{x}_i^T \\
w'_i \mathbf{x}_i^T & \mathbf{0}^T & -x'_i \mathbf{x}_i^T
\end{pmatrix}
\begin{pmatrix}
\mathbf{h}_1 \\
\mathbf{h}_2 \\
\mathbf{h}_3
\end{pmatrix}
= \mathbf{0}
$$

여기서 $w'_i = 1$로 두면, 하나의 점 쌍마다 $2 \times 9$ 크기의 행렬 $A_i$가 만들어집니다.
$N$개의 점이 있다면 $2N \times 9$ 크기의 행렬 $A$를 만들 수 있고, 최종적으로 **$A \mathbf{h} = 0$** 꼴의 문제가 됩니다.

### 2.2 해 구하기 (SVD)
$A \mathbf{h} = 0$의 해는 자명해($\mathbf{h}=0$)를 제외하고, $\|\mathbf{h}\|=1$ 제약 조건 하에서 $\|A\mathbf{h}\|$를 최소화하는 문제입니다.
이는 **SVD (Singular Value Decomposition)**를 통해 풀 수 있습니다.
$A = U \Sigma V^T$ 로 분해했을 때, **가장 작은 특이값(Singular Value)에 대응하는 $V^T$의 마지막 행(Right Singular Vector)**이 바로 우리가 찾는 해 $\mathbf{h}$ 입니다.

---

## 3. Data Normalization (정규화)

DLT를 수행하기 전에 데이터 정규화는 **선택이 아니라 필수**입니다.

### 3.1 왜 필요한가요?
이미지 좌표는 보통 $(0, 0)$에서 $(1920, 1080)$ 같은 큰 값을 가집니다.
$A$ 행렬을 구성할 때 $x, y$ ($10^3$ 단위)와 $xy$ ($10^6$ 단위) 항이 섞이게 되는데, 이렇게 스케일 차이가 큰 값들이 하나의 행렬에 들어가면 **Condition Number**가 매우 나빠져서, 작은 노이즈에도 해가 크게 흔들리는 수치적 불안정성(Numerical Instability)이 발생합니다.

### 3.2 정규화 과정
1.  **Shift**: 점들의 중심(Centroid)이 원점 $(0, 0)$에 오도록 이동시킵니다.
2.  **Scale**: 점들의 원점으로부터의 평균 거리가 $\sqrt{2}$가 되도록 스케일링합니다.

변환 행렬 $T$를 구해서 $\mathbf{\tilde{x}} = T \mathbf{x}$ 로 변환한 뒤 DLT를 수행하여 $\tilde{H}$를 구합니다.
최종 $H$는 다시 원래 좌표계로 돌려놓아야 하므로 다음 식을 사용합니다:

$$ H = T'^{-1} \tilde{H} T $$

---

## 4. RANSAC (Random Sample Consensus)

실제 데이터에는 잘못 매칭된 점(**Outlier**)이 반드시 포함되어 있습니다. DLT는 아웃라이어에 매우 취약합니다 (Least Squares 방식이기 때문). RANSAC은 이를 극복하기 위한 반복적 추정 알고리즘입니다.

### 4.1 알고리즘 순서
1.  **Sample**: 전체 데이터에서 최소 개수(4개)의 점 쌍을 무작위로 뽑습니다.
2.  **Model**: 뽑힌 점들로 $H$를 계산합니다 (DLT).
3.  **Verify**: 계산된 $H$를 나머지 모든 점들에 적용해 봅니다.
    *   변환된 점과 실제 매칭점 사이의 거리(Reprojection Error)를 구합니다.
    *   거리가 임계값(`threshold`)보다 작은 점들을 **Inlier**로 분류합니다.
4.  **Update**: 현재 Inlier 개수가 지금까지의 최대 기록보다 많으면, 현재 모델을 'Best Model'로 저장합니다.
5.  **Repeat**: 위 과정을 `max_iters` 번 반복합니다.
6.  **Refine**: 최종적으로 선택된 Best Inlier 점들만 가지고 다시 DLT를 수행하여 $H$를 정밀하게 다듬습니다.

이 과정을 통해 노이즈가 많은 상황에서도 매우 강인(Robust)하게 호모그래피를 찾아낼 수 있습니다.
