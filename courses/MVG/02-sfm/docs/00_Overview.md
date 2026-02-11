# Structure from Motion (SfM) - 개요 및 문서 목차

## 0. 문서 인덱스 (Map of Content)
이 폴더(`docs/`)에는 단계별 구현 가이드와 이론적 배경이 정리되어 있습니다.

- **[📖 02_Implementation.md (구현 가이드)](02_Implementation.md)**: 실제 코드 구현을 위한 상세 튜토리얼 (Walkthrough).
- **[🧠 01_Theory.md (이론 및 PnP)](01_Theory.md)**: SfM 핵심 개념과 PnP(PnP_nl) 심화 학습.
- **[📄 원본 과제 PDF](../CSCI%205563_%20Assignment%20%234%20Structure%20from%20Motion.pdf)**: 과제 요구사항이 담긴 원본 문서 (상위 폴더).
- **[💻 메인 실행 코드](../hw4.py)**: 전체 파이프라인을 실행하는 메인 스크립트 (상위 폴더).
- **[📂 구 버전 문서 (English)](archive/sfm_assignment_guide_en.md)**: 원본 영문 가이드.

---

## 1. 과제 개요
본 과제는 여러 장의 2D 사진을 이용해 3D 공간(Structure)과 카메라 위치(Motion)를 동시에 복원하는 **SfM (Structure from Motion)** 시스템을 구현합니다.

### 핵심 단계
1.  **Feature Matching**: 특징점 검출 및 매칭 (SIFT).
2.  **Initialization**: 초기 2장의 이미지로 3D 구조 생성 (Epipolar Geometry).
3.  **PnP (Perspective-n-Point)**: 새로운 카메라의 위치 추정.
4.  **Triangulation**: 새로운 3D 점 복원.
5.  **Bundle Adjustment**: 전체적인 오차 최소화 (Non-linear Optimization).
