# MVG Assignment 2: Augmented Reality with Planar Homography

이 문서는 MVG Assignment 2의 과제 내용, 해결 솔루션(알고리즘), 코드 구현 분석, 그리고 실행 방법을 상세히 설명합니다.

## 1. 과제 개요
이 과제의 목표는 **평면 호모그래피(Planar Homography)** 를 이용하여 이미지와 비디오에 증강 현실(AR) 효과를 구현하는 것입니다.
주요 내용은 다음과 같습니다:
1.  **특징점 매칭 (Feature Matching)**: 두 이미지 간의 대응점 찾기 (FAST Corner Detector, BRIEF Descriptor).
2.  **호모그래피 계산 (Homography Estimation)**: 매칭된 점들을 이용하여 한 평면을 다른 평면으로 매핑하는 변환 행렬 $H$ 계산.
    *   **Direct Linear Transform (DLT)**: 기본적인 $H$ 계산.
    *   **Normalization**: 수치적 안정성을 위한 좌표 정규화.
    *   **RANSAC**: 노이즈(Outlier)에 강인한 $H$ 추정.
3.  **이미지 합성 (Warping & Compositing)**: 계산된 $H$를 이용해 이미지를 변형하고 합성.
4.  **응용 (Application)**:
    *   **HarryPotterize**: 책 표지를 해리포터 표지로 교체.
    *   **AR Video**: 동영상 속 책 표지에 다른 비디오를 오버레이.
    *   **Panorama (Extra Credit)**: 두 이미지를 이어 붙여 파노라마 생성.

---

## 2. 문제 풀이 및 솔루션 상세 (Solution Logic)

### 2.1 Feature Matching (`matchPics.py`)
두 이미지 $I_1, I_2$ 사이의 관계를 알기 위해서는 서로 대응되는 점들을 찾아야 합니다.
*   **Corner Detection**: `helper.corner_detection`을 사용해 코너(특징점)를 찾습니다.
*   **Descriptor Extraction**: `helper.computeBrief`를 사용해 각 특징점의 지역적 특징을 기술하는 BRIEF 벡터를 뽑습니다.
*   **Matching**: `helper.briefMatch`를 사용해 두 이미지의 descriptor들을 비교하여 가장 유사한 쌍을 찾습니다.

### 2.2 Homography Computation (`planarH.py`)
매칭된 점들의 집합 $x_1, x_2$가 주어졌을 때, $x_1 \equiv H x_2$ 관계를 만족하는 $3 \times 3$ 행렬 $H$를 구합니다.

1.  **DLT (`computeH`)**:
    *   $x_1 \times H x_2 = 0$ 식을 이용해 선형 방정식 $Ah=0$ 꼴로 만듭니다.
    *   SVD(Singular Value Decomposition)를 이용해 이 방정식의 해(최소 자승해)를 구합니다. $H$는 $A$의 가장 작은 singular value에 대응하는 singular vector입니다.

2.  **Normalization (`computeH_norm`)**:
    *   좌표값의 스케일 차이로 인한 수치적 불안정을 막기 위해, 점들의 중심을 원점으로 옮기고 평균 거리가 $\sqrt{2}$가 되도록 정규화합니다.
    *   정규화된 좌표로 $H_{norm}$을 구한 뒤, 다시 원래 좌표계로 돌려놓는 역변환(Denormalization) 과정을 거쳐 최종 $H$를 얻습니다. $H = T_1^{-1} H_{norm} T_2$.

3.  **RANSAC (`computeH_ransac`)**:
    *   매칭 결과에는 잘못된 매칭(Outlier)이 섞여 있을 수 있습니다.
    *   무작위로 4개의 점 쌍을 뽑아 $H$를 계산하고, 이 $H$가 나머지 점들을 얼마나 잘 설명그하는지(Inlier 개수) 확인합니다.
    *   이 과정을 반복하여 가장 많은 Inlier를 가지는 최적의 $H$를 선택합니다.

### 2.3 Image Compositing (`compositeH`)
*   Template 이미지(덮어씌울 이미지)를 $H$를 이용해 Target 이미지(배경)의 시점으로 변형(Warping)합니다.
*   이때 $H$는 Target -> Template 방향이라면, Warping에는 그 역행렬인 $H^{-1}$가 필요합니다 (Backward Warping).
*   마스크를 이용해 덮어씌울 영역만 Template 이미지로 교체합니다.

---

## 3. 코드 구현 분석 (Code Implementation)

### 3.1 `matchPics.py`
```python
def matchPics(I1, I2):
    # 1. 이미지를 흑백으로 변환 (GrayScale)
    # 2. 특징점 검출 (corner_detection)
    # 3. Descriptor 추출 (computeBrief) - 특징점 주변의 패턴 정보
    # 4. 특징점 매칭 (briefMatch) - 서로 비슷한 특징점끼리 연결
    # 결과: 매칭된 인덱스 쌍(matches)과 각 이미지의 특징점 좌표들(locs1, locs2) 반환
```

### 3.2 `planarH.py`
*   `computeH(x1, x2)`: DLT 알고리즘 구현. $2N \times 9$ 행렬 $A$를 구성하고 `np.linalg.svd`로 풀이.
*   `computeH_norm(x1, x2)`: 좌표 정규화 로직 적용 후 `computeH` 호출.
    *   Centroid 계산 -> Shift -> Scale -> Similarity Transform Matrix $T$ 생성 (순서: Scale * Shift). (**주의**: 코드에서는 $T$ 행렬을 직접 구성함)
*   `computeH_ransac(x1, x2)`:
    *   Loop `max_iters`:
        *   Random 4 points sampling.
        *   `computeH_norm`으로 $H$ 계산.
        *   모든 $p_2$를 $H$로 변환하여 $p_1$과의 거리(에러) 계산.
        *   `dist < inlier_tol`인 점들을 Inlier로 카운트.
        *   최대 Inlier 갱신 시 $H$ 저장.
*   `compositeH(H2to1, template, img)`:
    *   `cv2.warpPerspective`를 사용해 `template`을 `img` 위에 합성.
    *   `H_inv`를 사용하여 `template`을 `img` 좌표계로 변형.

### 3.3 `HarryPotterize.py` (Q3.9)
1.  이지 로드: `cv_desk` (배경), `cv_cover` (찾을 표지), `hp_cover` (바꿀 표지).
2.  `matchPics(cv_cover, cv_desk)`: 책상 위 사진에서 표지 특징점 찾기.
3.  `computeH_ransac`: 표지 -> 책상 변환 행렬 $H$ 계산.
4.  `hp_cover`를 `cv_cover` 크기로 리사이즈 (간단한 합성을 위함).
5.  `compositeH`: $H$를 이용해 `hp_cover`를 책상 이미지 위에 합성.

### 3.4 `ar.py` (Q4.1)
동영상 처리를 위한 확장입니다.
1.  비디오 로드: `book.mov` (배경 영상), `ar_source.mov` (재생할 AR 영상).
2.  프레임 루프:
    *   `ar_source` 프레임을 `cv_cover` 비율에 맞게 크롭(Crop) 및 리사이즈.
    *   `matchPics(cv_cover, frame_book)`: 현재 비디오 프레임에서 책 표지 찾기.
    *   `computeH_ransac`: 호모그래피 계산. (매칭 점이 너무 적으면 건너뜀).
    *   `compositeH`: AR 영상을 책 위에 합성.
3.  결과: 책 표지 위에서 쿵푸 팬더 영상(ar_source)이 재생되는 효과.

### 3.5 `panaroma.py` (Extra Credit)
1.  `pano_left`, `pano_right` 로드.
2.  `matchPics`로 두 이미지 간 공통 특징점 찾기.
3.  `computeH_ransac`: Right -> Left 변환 $H$ 계산.
4.  `cv2.warpPerspective`로 Right 이미지를 Left 좌표계로 변환 (캔버스 크기는 두 이미지 합친 너비).
5.  Left 이미지를 캔버스 왼쪽에 배치하고, 변형된 Right 이미지와 합성.

---

## 4. 실행 방법 (Detailed Execution Instructions)

터미널에서 `/Users/joon/dev/MVG/assgn2` 디렉토리로 이동한 후 실행합니다. (또는 `solution` 폴더 내에서 실행)

**준비 사항:**
*   필요 라이브러리: `numpy`, `opencv-python` (`cv2`), `scikit-image` (`skimage`).
*   데이터 파일: `data/` 폴더에 `cv_desk.png`, `cv_cover.jpg`, `hp_cover.jpg`, `book.mov`, `ar_source.mov` 등이 있어야 함.

### 4.1 HarryPotterize (이미지 합성)
책상 이미지의 책 표지를 해리포터로 바꿉니다.
```bash
cd /Users/joon/dev/MVG/assgn2/solution
python HarryPotterize.py
```
*   **결과**: 'Composite Image' 창이 뜨고, 책 표지가 해리포터로 바뀐 이미지가 보여야 합니다. 아무 키나 누르면 종료됩니다.

### 4.2 AR Application (동영상 AR)
책상 위 움직이는 책 표지에 영상을 띄웁니다.
```bash
cd /Users/joon/dev/MVG/assgn2/solution
python ar.py
```
*   **결과**: 비디오가 재생되면서 책 표지 영역에 쿵푸 팬더 영상이 오버레이 됩니다. 책이 움직여도 영상이 따라다녀야 합니다. 'q'를 누르면 종료됩니다.
*   **참고**: 초기 `generate_image` 기능이 없는 환경이므로 `cv2.imshow`가 작동하지 않는 원격 환경이라면 실행 결과를 볼 수 없습니다. (로컬 환경에서 실행 권장)

### 4.3 Panorama (파노라마 스티칭)
```bash
cd /Users/joon/dev/MVG/assgn2/solution
python panaroma.py
```
*   **결과**: 'Panorama' 창에 두 이미지가 자연스럽게 연결된 파노라마 사진이 뜹니다. 또한 `../result/panorama.png`로 저장됩니다.

---

## 5. 결론
이 코드는 컴퓨터 비전의 핵심 개념인 **특징점 매칭**과 **호모그래피**를 구현하여 실용적인 AR 애플리케이션을 완성하는 모범적인 솔루션입니다.
*   `computeH_ransac`이 가장 중요한 핵심 함수이며, 노이즈가 많은 실제 환경에서 강인하게 모델을 추정하는 방법을 보여줍니다.
*   `matchPics`의 성능(특징점 검출 품질)이 전체 AR 품질에 큰 영향을 미칩니다.
