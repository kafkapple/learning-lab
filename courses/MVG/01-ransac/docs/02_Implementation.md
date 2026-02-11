# 과제 2: 증강 현실(AR) 구현 가이드

안녕하세요! 이 문서는 컴퓨터 비전 과제 2를 해결하기 위한 단계별 가이드입니다. 각 파일과 함수의 역할을 설명하고, 어떤 순서로 무엇을 구현해야 하는지 안내합니다.

## 1. 프로젝트 구조 이해

먼저 우리에게 주어진 파일들이 각각 어떤 역할을 하는지 알아봅시다.

- **`helper.py`**: FAST 코너 검출, BRIEF 디스크립터 계산 등 복잡한 기본 기능들이 이미 구현된 **도구 상자**입니다. 우리는 이 파일의 함수들을 잘 가져다 쓰기만 하면 됩니다.
- **`matchPics.py`**: 두 이미지 사이에서 특징점을 찾아 연결(매칭)하는 `matchPics` 함수를 구현해야 하는 파일입니다. `helper.py`의 도구들을 사용하는 방법을 여기서 익힙니다.
- **`planarH.py`**: Homography 행렬 `H`를 계산하는 수학적 핵심 로직(DLT, RANSAC)과, 이를 이용해 이미지를 변형하고 합성하는 함수들을 구현하는 파일입니다.
- **`HarryPotterize.py`**: `matchPics`와 `planarH`에서 만든 함수들을 이용해 책상 위 책 표지를 해리포터 표지로 바꾸는 이미지 합성 프로그램을 완성하는 파일입니다.
- **`ar.py`**: 위 기능들을 동영상에 적용하여, 책 표지 위에서 다른 영상이 재생되는 최종 AR 애플리케이션을 구현하는 파일입니다.
- **`briefRotTest.py`**: BRIEF 디스크립터가 회전에 얼마나 취약한지 테스트하는 스크립트입니다.

## 2. 단계별 구현 가이드

(이전 단계 생략)

---

### 단계 4: 애플리케이션 완성 (`HarryPotterize.py` & `ar.py`)

(이전 단계 생략)

---

이 가이드를 따라 모든 단계를 완료하면 과제가 완성됩니다. 행운을 빕니다!

---

## 3. 추가 과제 (Extra Credit)

### `panaroma.py`: 파노라마 이미지 스티칭

카메라를 한 자리에서 회전하며 찍은 여러 장의 사진은 Homography 관계를 이용해 하나로 합칠 수 있습니다. 여기서는 `pano_left.jpg`와 `pano_right.jpg` 두 장의 이미지를 이어 붙여 하나의 넓은 파노라마 사진을 만듭니다.

**구현 아이디어:**
`left` 이미지를 기준으로, `right` 이미지를 변형(`warp`)시켜 `left` 이미지 옆에 자연스럽게 갖다 붙입니다.

**`panaroma.py` 스크립트 구현 지침:**

1.  **이미지 불러오기**: `cv2.imread()`로 `data/pano_left.jpg`와 `data/pano_right.jpg`를 불러옵니다.
2.  **특징점 매칭**: `matchPics(pano_left, pano_right)`를 호출합니다. 여기서 `pano_left`가 기준(I1)이 되고 `pano_right`가 변형될 대상(I2)이 됩니다.
3.  **Homography 계산**:
    *   `matches` 정보를 이용해 매칭된 점들의 좌표 `x1` (left), `x2` (right)를 정리합니다.
    *   `computeH_ransac(x1, x2)`를 호출하여 Homography `H`를 계산합니다. 이 `H`는 `pano_right`의 점을 `pano_left`의 좌표계로 보내는 변환입니다 (`x_left ~ H * x_right`).

4.  **이미지 변형 (Warping)**:
    *   `cv2.warpPerspective` 함수를 사용하여 `pano_right` 이미지를 `H` 행렬에 따라 변형합니다.
    *   **캔버스 크기(dsize)**: 결과물은 두 이미지를 합친 것보다 커야 합니다. 두 이미지의 가로 길이를 더한 정도로 넉넉한 크기의 캔버스를 지정합니다. 예를 들어, `(left.shape[1] + right.shape[1], left.shape[0])`.
    *   `warped_right = cv2.warpPerspective(pano_right, H, (width, height))` 와 같이 호출합니다.

5.  **이미지 합성 (Stitching)**:
    *   `warped_right` 이미지에는 변형된 `pano_right`의 모습이 담겨 있습니다.
    *   이 `warped_right` 이미지의 왼쪽 부분(`0`부터 `pano_left.shape[1]`까지)에 원본 `pano_left` 이미지를 그대로 복사해 덮어씌웁니다.
    *   이렇게 하면 왼쪽은 원본, 오른쪽은 변형되어 연결된 파노라마 이미지가 완성됩니다.

6.  **결과 출력**: 완성된 파노라마 이미지를 화면에 띄웁니다. 경계선이 자연스럽게 연결되는지 확인해보세요.
