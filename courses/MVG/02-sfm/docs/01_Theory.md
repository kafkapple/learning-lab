# Structure from Motion (SfM) - 이론적 배경 심화

이 문서는 SfM의 핵심 알고리즘인 Epipolar Geometry, Triangulation, PnP, 그리고 Bundle Adjustment에 대해 깊이 있게 다룹니다.

---

## 1. Epipolar Geometry (에피폴라 기하학)

두 카메라 사이의 기하학적 관계를 다루는 이론입니다. 한 이미지의 점 $x$는 다른 이미지의 대응되는 에피폴라 선(Epipolar Line) 위에 존재해야 한다는 제약 조건을 가집니다.

### 1.1 Essential Matrix ($E$)
정규화된 이미지 좌표계(Normalized Image Coordinates)에서 두 카메라 간의 회전($R$)과 이동($t$)을 표현하는 $3 \times 3$ 행렬입니다.
본질저으로 $E$는 외적(Cross Product)을 행렬로 표현한 것과 회전 행렬의 결합입니다.

$$ E = [t]_{\times} R $$

**Epipolar Constraint**:
$$ \mathbf{x}'^T E \mathbf{x} = 0 $$
이 식은 한 점과 그 대응점, 그리고 두 카메라 원점이 하나의 평면(Epipolar Plane) 위에 있음을 의미합니다.

### 1.2 Pose Recovery (포즈 복원)
$E$ 행렬을 SVD($U \Sigma V^T$)하면 수학적으로 가능한 4가지의 $(R, t)$ 조합이 나옵니다.
*   $(U W V^T, +u_3)$
*   $(U W V^T, -u_3)$
*   $(U W^T V^T, +u_3)$
*   $(U W^T V^T, -u_3)$

이 중 올바른 해를 찾기 위해 **Cheirality Check**를 수행합니다. 복원된 3D 점이 **두 카메라 모두의 앞쪽(Positive Depth)**에 위치하는 경우가 유일한 정답입니다.

---

## 2. Triangulation (삼각측량)

카메라 포즈($P, P'$)와 매칭된 2D 점($x, x'$)을 알 때, 3D 점 $X$를 찾는 과정입니다.

### 2.1 Linear Triangulation (DLT)
$$ x = PX \implies x \times (PX) = 0 $$
이 식을 이용해 $AX=0$ 형태의 선형 방정식을 만들고, SVD를 이용해 $X$를 구합니다. 이는 빠르지만 재투영 오차를 기하학적으로 최소화하지는 않습니다.

---

## 3. PnP (Perspective-n-Point) 및 Refinement

### 3.1 PnP Problem
이미 알고 있는 3D 점들($X_i$)과 새로운 2D 이미지 점들($x_i$) 사이의 매칭을 이용해, 새로운 카메라의 포즈($R, t$)를 구하는 문제입니다. 주로 **P3P**나 **EPnP** 알고리즘이 사용되며, RANSAC과 결합하여 Outlier를 제거합니다.

### 3.2 Non-linear PnP (PnP_nl)
초기 PnP 해는 대수적 오차(Algebraic Error)를 최소화한 것이라 부정확할 수 있습니다.
따라서 **Reprojection Error**를 최소화하는 비선형 최적화를 수행해야 합니다.

$$ \min_{R, t} \sum_i \| x_i - \text{proj}(K[R|t]X_i) \|^2 $$

Levenberg-Marquardt와 같은 최적화 기법을 사용하여 $R, t$를 미세 조정(Refine)합니다. 이 과정이 문서에서 언급되는 `PnP_nl` 입니다.

---

## 4. Bundle Adjustment (BA, 번들 조정)

SfM의 마지막 단계이자 가장 중요한 단계입니다. 지금까지 구한 **모든** 카메라 포즈와 **모든** 3D 점들을 한꺼번에 최적화하여 전체적인 오차를 줄입니다.

### 4.1 목적 함수 (Cost Function)
전체 재투영 오차(Total Reprojection Error)의 합을 최소화합니다.

$$ \min_{\{P_j\}, \{X_i\}} \sum_{j} \sum_{i} v_{ij} \| x_{ij} - \text{proj}(P_j X_i) \|^2 $$

*   $P_j$: $j$번째 카메라 포즈
*   $X_i$: $i$번째 3D 점
*   $x_{ij}$: $j$번째 카메라에서 관측된 $i$번째 점의 2D 좌표
*   $v_{ij}$: 관측 여부 (보이면 1, 안 보이면 0)

이 문제는 변수의 개수가 매우 많은 대규모 비선형 최소제곱 문제(Large-scale Non-linear Least Squares)이며, 보통 희소성(Sparsity)을 활용한 Levenberg-Marquardt 알고리즘(`scipy.optimize.least_squares`)으로 풉니다.
