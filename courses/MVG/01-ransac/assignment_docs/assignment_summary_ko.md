# 과제 2: 평면 호모그래피를 이용한 증강현실 (16-385 Computer Vision)

이 문서는 16-385 Computer Vision (Spring 2024) 과제 2, "Augmented Reality with Planar Homographies"의 내용을 체계적으로 정리한 것입니다.

---

## 1. 과제 개요 및 목표

본 과제는 **Planar Homography(평면 호모그래피)**를 이용하여 사진 및 영상 속 평면 객체(책 표지)를 추적하고, 이를 다른 이미지나 영상으로 대체하는 **Augmented Reality(AR, 증강 현실)** 애플리케이션 제작을 목표로 합니다.

### 핵심 개념

- **Planar Homography**: 한 평면을 다른 평면으로 투영(Project)했을 때 발생하는 3x3 변환 행렬. 이미지 왜곡 및 보정에 사용됩니다.
- **Homogeneous Coordinates (동차 좌표계)**: 2D 좌표 (x, y)를 (x, y, 1)로 표현하여 이동, 회전, 스케일링 등 복잡한 변환을 단일 행렬 곱으로 처리하기 위한 좌표계입니다.
- **Feature Detection & Matching**: 두 이미지에서 서로 상응하는 특징점(코너 등)을 자동으로 찾아 연결하는 기술입니다 (FAST, BRIEF, Hamming Distance).
- **RANSAC (Random Sample Consensus)**: 잘못 매칭된 점(Outlier)들을 제거하고, 가장 많은 지지를 받는 최적의 Homography 행렬을 강인하게(robust) 추정하는 알고리즘입니다.

---

## 2. 세부 과제 내용 (Tasks)

### 2.1 이론 문제 (Theory Questions)

- **Q2.1 Correspondences (10점)**:
    - Homography 행렬의 자유도(Degrees of Freedom) 계산.
    - H 행렬 추정에 필요한 최소 점 쌍의 개수.
    - `Ah=0` 행렬 유도 및 Rank 분석.

- **Q2.2 Fun with Homogeneous Coordinates (15점)**:
    - 원추곡선(Conic) 방정식의 최소제곱 추정과 필요한 점의 개수.
    - 대칭 행렬 `Q`를 이용한 행렬 형태 표현 및 투영 변환 관계 유도.

- **Q3.1~3.3 특징점 분석 (15점)**:
    - FAST와 Harris Corner Detector 비교.
    - BRIEF와 Filterbank 기반 디스크립터 비교.
    - Hamming distance와 Nearest Neighbor를 이용한 매칭 원리 설명.

### 2.2 프로그래밍 과제 (Programming Tasks)

#### Part 1: 특징점 매칭 및 회전 테스트
- **Q3.4 `matchPics.py`**: 두 이미지 간의 특징점을 찾고(FAST), 기술자(descriptor)를 계산하여(BRIEF), 매칭 결과를 시각화합니다.
- **Q3.5 `briefRotTest.py`**: 이미지를 회전시키며 매칭되는 특징점 수의 변화를 히스토그램으로 분석하여 BRIEF의 회전 불변성 한계를 확인합니다.

#### Part 2: Homography 계산 및 RANSAC
- **Q3.6 `computeH`**: DLT(Direct Linear Transform)를 이용해 점 쌍으로부터 Homography 행렬 `H`를 계산합니다.
- **Q3.7 `computeH_norm`**: 수치적 안정성을 위해 점들을 정규화(Normalization)한 후 `H` 행렬을 계산합니다.
- **Q3.8 `computeH_ransac`**: RANSAC 알고리즘을 구현하여 노이즈가 섞인 데이터에서 최적의 `H` 행렬을 찾습니다.

#### Part 3: AR 애플리케이션 구현
- **Q3.9 `HarryPotterize.py`**: Homography를 이용해 `hp_cover.jpg` 이미지를 `cv_desk.png` 이미지의 책 표지 영역에 합성(composite)합니다.
- **Q4.1 `ar.py`**: 비디오(`book.mov`)의 매 프레임마다 책 표지를 추적하여, 다른 비디오(`ar_source.mov`)의 프레임을 실시간으로 덮어씌웁니다. (Aspect Ratio 처리 필요)

### 2.3 추가 과제 (Extra Credit)
- **Q5.1x Panorama**: 동일한 지점에서 회전하며 찍은 두 장의 사진(`pano_left.jpg`, `pano_right.jpg`)을 Homography를 이용해 연결하여 파노라마 영상을 생성합니다.

---

## 3. 행정 사항

- **마감 기한**: 2024년 2월 21일 수요일 23:59.
- **제출물**: 전체 코드와 리포트(write-up)를 포함한 단일 zip 파일.
- **제약 사항**: 절대 경로 사용 금지, 제공된 함수 프로토타입 준수.
- **파일 구조**:
    - `python/`: `ar.py`, `briefRotTest.py`, `HarryPotterize.py`, `matchPics.py`, `planarH.py`
    - `result/`: `ar.avi`
    - `ec/`: (선택) `panorama.py` 등
    - `<AndrewId>.pdf`: 리포트 파일
