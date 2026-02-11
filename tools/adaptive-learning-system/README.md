# 몰입 최적화 학습 시스템 (Focus-Optimized Learning System)

> 새로운 지식을 체계적으로, 빠르고, 쉽고, 정확하게 습득하기 위한 인지과학 기반 학습 시스템

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 개요

이 시스템은 현대인의 짧은 주의 지속 시간을 극복하고, 몰입(Flow) 상태를 유지하며 효율적으로 지식을 관리하기 위해 설계된 자동화 도구입니다.

### 핵심 철학

```
지식 입력 → 청킹 & 구조화 → 우선순위화 → 간격 반복 → 게이미피케이션 → 장기 기억
```

### 왜 이 시스템인가?

| 일반 학습 도구 | 몰입 최적화 시스템 |
|--------------|-------------------|
| 긴 학습 세션 | 15-25분 마이크로 학습 (Micro-learning) |
| 지연된 피드백 | 즉각적 XP/레벨 보상 (Gamified Feedback) |
| 고정된 스케줄 | 인지 에너지 기반 스케줄링 (Adaptive Scheduling) |
| 단조로운 반복 | 다양성 보장 (Context Switching 최소화) |
| 과몰입/탈진 | 45분 Deep Work 제한 및 휴식 유도 |

---

## 주요 기능

### 1. FSRS 간격 반복 알고리즘

[FSRS (Free Spaced Repetition Scheduler)](https://github.com/open-spaced-repetition/fsrs4anki)는 기존 SM-2 대비 **20-30% 적은 복습**으로 동일한 기억률을 달성합니다.

```python
from src.core.fsrs import FSRS, Card, Rating

fsrs = FSRS()
card = Card(
    card_id="python_001",
    content="Python의 GIL이란?",
    answer="Global Interpreter Lock - 한 번에 하나의 스레드만 실행"
)

# 복습 후 다음 스케줄 자동 계산
card = fsrs.repeat(card, datetime.now(), Rating.GOOD)
print(f"다음 복습: {card.scheduled_days}일 후")
```

**알고리즘 특징:**
- 3가지 메모리 변수: 검색가능성(R), 안정성(S), 난이도(D)
- 개인화된 망각 곡선 학습
- 지연된 복습에도 최적 스케줄링

### 2. 지식 청킹 & 구조화

큰 개념을 원자적 단위(Atomic Notes)로 분해하고 자동으로 연결합니다.

#### 수동 모드

```python
from src.core.knowledge import KnowledgeProcessor, KnowledgeType

processor = KnowledgeProcessor()

chunk = processor.create_chunk(
    title="Python GIL",
    content="""GIL은 Python 인터프리터가 한 번에 하나의 스레드만
    Python 바이트코드를 실행하도록 하는 뮤텍스입니다.""",
    knowledge_type=KnowledgeType.CONCEPT,
    tags=["python", "concurrency"]
)

# 자동 생성:
# - 우선순위: MEDIUM
# - 난이도: 6/10
# - 정교화 질문: "왜 GIL이 필요한가?"
```

#### LLM 기반 자동 분해 (Gemini/OpenAI)

긴 텍스트를 LLM이 자동으로 원자적 단위로 분해합니다:

```python
from src.core.knowledge import SmartKnowledgeProcessor

processor = SmartKnowledgeProcessor(llm_provider="auto")

# 긴 텍스트 입력 → 자동으로 여러 청크로 분해
chunks = processor.process_large_text(
    text="""...""",
    topic="Python 동시성",
    source="Python 공식 문서"
)

# 결과: 원자적 청크 자동 생성 + 스마트 플래시카드 생성
```

**자동화 기능:**
- **LLM 기반 분해**: 긴 텍스트를 원자적 단위로 자동 분리
- **스마트 플래시카드**: 하나의 청크에서 여러 유형의 카드 생성 (정의형, 비교형, 적용형, Why형)
- 키워드 기반 자동 태깅
- 연결 수 기반 우선순위 산정
- 정교화 질문 자동 생성 (Why? How? What if?)
- 유사 지식 자동 연결

### 3. 적응형 스케줄러 (Adaptive Scheduler)

에너지 레벨과 시간대에 맞춘 최적의 학습 스케줄을 생성합니다.

```python
from src.adaptive.scheduler import AdaptiveScheduler

scheduler = AdaptiveScheduler()

schedule = scheduler.create_daily_schedule(
    date=datetime.now(),
    available_hours=[(9, 12), (14, 17)],  # 학습 가능 시간
    cards_due=50,
    new_cards=15,
    energy_pattern={
        "morning": "high",
        "afternoon": "medium",
        "evening": "low"
    }
)

# 결과:
# 09:00-09:25 [새 지식 학습] (25분, high)
# 09:25-09:30 [휴식] (5분)
# 09:30-09:55 [복습] (25분, high)
# ...
```

**몰입 특화 기능:**
- 포모도로: 25분 작업 + 5분 휴식
- Deep Work 제한: 최대 45분 연속 작업 (탈진 방지)
- 다양성 보장: 같은 유형 3회 이상 연속 방지
- 에너지 매칭: 고에너지 시간에 어려운 작업

### 4. 게이미피케이션 엔진

도파민 기반 보상 시스템으로 학습 동기를 유지합니다.

```python
from src.gamification.engine import GamificationEngine

engine = GamificationEngine()

# 복습 기록
result = engine.record_review("user_001", correct=True, card_difficulty=7)
print(f"+{result['final_xp']}XP!")
```

**보상 시스템:**
- XP & 레벨 시스템 (성장 체감)
- 스트릭 (연속 학습) 보너스
- 뱃지 & 업적
- 일일 퀘스트

### 5. 자동화 & 알림

학습 리마인더와 루틴을 자동화합니다.

```python
from src.scheduler.automation import AutomationRunner

automation = AutomationRunner()
automation.setup({
    "daily": {
        "morning_reminder": "09:00",
        "evening_warning": "20:00"
    }
})
automation.start()
```

---

## 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/adaptive-learning-system.git
cd adaptive-learning-system

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 기본 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (.env)

프로젝트 루트에 `.env` 파일을 생성하여 API 키를 안전하게 관리합니다:

```bash
# .env 파일 생성
touch .env
```

```env
# .env 파일 내용 (예시)
GOOGLE_API_KEY=your-gemini-api-key
```

### 3. 시스템 시작

```bash
# 웹 대시보드 실행
python src/web/server.py

# 브라우저에서 http://localhost:5000 접속
```

또는 CLI 사용:
```bash
python src/main.py
```

### 첫 번째 학습

```bash
$ python src/main.py

🧠 적응형 학습 시스템에 오신 것을 환영합니다!

> add
제목: Python 리스트 컴프리헨션
...
```

---

## 프로젝트 구조

```
adaptive-learning-system/
├── config/
│   ├── settings.yaml        # 전체 설정
│   └── ROUTINE_GUIDE.md     # 루틴 가이드
├── src/
│   ├── core/
│   │   ├── fsrs.py          # FSRS 간격반복 알고리즘
│   │   ├── knowledge.py     # 지식 청킹 & LLM 자동 분해
│   │   └── database.py      # SQLite 저장소
│   ├── adaptive/
│   │   └── scheduler.py     # 적응형 스케줄러
│   ├── gamification/
│   │   └── engine.py        # XP/레벨 시스템
│   ├── scheduler/
│   │   └── automation.py    # 자동 알림
│   ├── web/                 # 웹 대시보드
│   └── main.py              # CLI 인터페이스
├── data/                    # 데이터 저장소
└── tests/                   # 테스트
```

---

## 설정

`config/settings.yaml`에서 시스템을 개인화할 수 있습니다:

```yaml
# 적응형 학습 설정
adaptive:
  pomodoro:
    work_duration: 25      # 작업 시간 (분)
    short_break: 5         # 짧은 휴식
    max_work_duration: 45  # Deep Work 제한
  
  energy_pattern:
    morning: "high"
    afternoon: "medium"
    evening: "low"
```

---

## 라이선스

MIT License
