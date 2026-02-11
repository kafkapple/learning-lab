# 과제 2: 평면 호모그래피를 이용한 증강현실 (AR) - 개요 및 문서 목차

## 0. 문서 인덱스 (Map of Content)
이 폴더(`docs/`)에는 과제 수행에 필요한 모든 문서가 체계적으로 정리되어 있습니다.

- **[📖 02_Implementation.md (구현 가이드)](02_Implementation.md)**: 실제 코드 구현을 위한 상세 튜토리얼입니다.
- **[🧠 01_Theory.md (이론 및 배경)](01_Theory.md)**: Homography, RANSAC 등 핵심 이론 설명입니다.
- **[📄 원본 과제 PDF](../assgn2.pdf)**: 과제 요구사항이 담긴 원본 문서 (상위 폴더).
- **[💻 소스 코드 폴더](../solution/)**: 완성된 코드가 저장된 폴더 (상위 폴더).
- **[데이터 폴더](../data/)**: 테스트용 이미지와 영상 (상위 폴더).

---

## 1. 과제 개요
본 과제는 **Planar Homography(평면 호모그래피)**를 이용하여 사진 및 영상 속 평면 객체(책 표지)를 추적하고, 이를 다른 이미지나 영상으로 대체하는 **Augmented Reality(AR, 증강 현실)** 애플리케이션 제작을 목표로 합니다.

### 핵심 목표
1.  **Feature Matching**: 두 이미지 간의 대응점 찾기 (FAST, BRIEF).
2.  **Homography Estimation**: 대응점을 이용한 변환 행렬 $H$ 계산 (DLT, RANSAC).
3.  **Image Compositing**: $H$를 이용한 이미지 합성 (Warping).
