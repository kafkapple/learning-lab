# ADHD 최적화 학습 자동화 시스템

> 새로운 지식을 체계적으로, 빠르고, 쉽고, 정확하게 습득하기 위한 ADHD 친화적 학습 시스템

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 개요

이 시스템은 ADHD를 가진 학습자를 위해 최적화된 지식 관리 및 학습 자동화 도구입니다.

### 핵심 철학

```
지식 입력 → 청킹 & 구조화 → 우선순위화 → 간격 반복 → 게이미피케이션 → 장기 기억
```

### 왜 이 시스템인가?

| 일반 학습 도구 | ADHD 최적화 시스템 |
|--------------|-------------------|
| 긴 학습 세션 | 15-25분 포모도로 |
| 지연된 피드백 | 즉각적 XP/레벨 보상 |
| 고정된 스케줄 | 에너지 기반 스케줄링 |
| 단조로운 반복 | 다양성 보장 (3회 연속 제한) |
| 과몰입 방치 | 45분 하이퍼포커스 제한 |

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
    text="""
    Python의 GIL(Global Interpreter Lock)은 인터프리터가 한 번에
    하나의 스레드만 실행하도록 하는 뮤텍스입니다. 이로 인해
    CPU-bound 작업에서 멀티스레딩의 이점을 얻기 어렵습니다.

    해결책으로 multiprocessing 모듈을 사용하면 각 프로세스가
    독립적인 인터프리터를 가져 진정한 병렬 처리가 가능합니다.

    또한 I/O-bound 작업에서는 asyncio를 활용하여 비동기 처리를
    할 수 있습니다.
    """,
    topic="Python 동시성",
    source="Python 공식 문서"
)

# 결과: 3개의 원자적 청크 자동 생성
# - Python GIL 개념
# - multiprocessing 사용법
# - asyncio 활용
# + 자동 연결 및 다양한 플래시카드 생성
```

**자동화 기능:**
- **LLM 기반 분해**: 긴 텍스트를 원자적 단위로 자동 분리
- **스마트 플래시카드**: 하나의 청크에서 여러 유형의 카드 생성 (정의형, 비교형, 적용형, Why형)
- 키워드 기반 자동 태깅
- 연결 수 기반 우선순위 산정
- 정교화 질문 자동 생성 (Why? How? What if?)
- 유사 지식 자동 연결

### 3. ADHD 최적화 스케줄러

에너지 레벨과 시간대에 맞춘 최적의 학습 스케줄을 생성합니다.

```python
from src.adhd.scheduler import ADHDScheduler

scheduler = ADHDScheduler()

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

**ADHD 특화 기능:**
- 포모도로: 25분 작업 + 5분 휴식
- 하이퍼포커스 제한: 최대 45분 연속 작업
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

# 스트릭 보너스
# 2일: 5%, 3일: 10%, 5일: 15%, 7일: 25%, 14일: 50%, 30일: 100%

# 대시보드
dashboard = engine.get_dashboard_data("user_001")
# {
#   "level": 5,
#   "total_xp": 2450,
#   "current_streak": 12,
#   "badges_earned": 5,
#   "motivational_message": "🔥 12일 연속 학습 중!"
# }
```

**보상 시스템:**
- XP & 레벨 시스템 (초반 레벨업 빠르게, 점진적 상승)
  - 레벨 1→2: 30XP (카드 3-5장)
  - 레벨 2→3: 60XP
  - 이후 점진적 증가
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

# 09:00 - "🧠 학습 시간! 오늘의 복습 카드가 기다리고 있어요."
# 20:00 - "🔥 스트릭 위험! 오늘 학습하지 않으면 12일 스트릭이 끊깁니다!"
```

---

## 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/adhd-learning-system.git
cd adhd-learning-system

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
# .env 파일 내용

# === LLM API (자동 지식 분해용, 택1) ===
# Google Gemini 사용시 (무료 티어 있음, 권장)
GOOGLE_API_KEY=your-gemini-api-key

# OpenAI 사용시
# OPENAI_API_KEY=your-openai-api-key

# === 모바일 알림 (선택) ===
# Pushover
# PUSHOVER_USER_KEY=your-user-key
# PUSHOVER_API_TOKEN=your-api-token

# Telegram
# TELEGRAM_BOT_TOKEN=your-bot-token
# TELEGRAM_CHAT_ID=your-chat-id

# Email (Gmail 예시)
# EMAIL_SMTP_HOST=smtp.gmail.com
# EMAIL_USERNAME=your-email@gmail.com
# EMAIL_PASSWORD=your-app-password
# EMAIL_TO=recipient@example.com
```

**API 키 발급:**
- **Gemini**: [Google AI Studio](https://aistudio.google.com/apikey)에서 무료 발급
- **OpenAI**: [OpenAI Platform](https://platform.openai.com/api-keys)에서 발급

### 3. LLM 기능 활성화 (선택)

자동 지식 분해 기능을 사용하려면 추가 패키지 설치:

```bash
# Gemini 사용시 (권장 - 무료 티어)
pip install google-generativeai python-dotenv

# 또는 OpenAI 사용시
pip install openai python-dotenv
```

### 4. 시스템 시작

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

🧠 ADHD 학습 시스템에 오신 것을 환영합니다!

> add
제목: Python 리스트 컴프리헨션
내용: [expression for item in iterable if condition] 형태로 리스트를 간결하게 생성
태그: python, syntax

✅ 지식 추가 완료: Python 리스트 컴프리헨션
   우선순위: MEDIUM, 난이도: 5/10
   태그: python, syntax

> start

📊 대시보드
   레벨: 1 (0 XP)
   스트릭: 1일 🔥

📝 오늘의 학습
   복습 대기: 3장
   새 카드: 1장

📝 질문: Python 리스트 컴프리헨션
   (엔터를 눌러 답변 확인)
   답변: [expression for item in iterable if condition] 형태로 리스트를 간결하게 생성

   평가: 1=다시 2=어려움 3=좋음 4=쉬움
   > 3
   ✅ 정답! +7XP (다음 복습: 2일 후)
```

---

## 프로젝트 구조

```
adhd-learning-system/
├── .env.example             # 환경변수 템플릿
├── .env                     # 환경변수 (생성 필요, git 무시됨)
├── .gitignore
├── requirements.txt
├── README.md
│
├── config/
│   ├── settings.yaml        # 전체 설정
│   └── ROUTINE_GUIDE.md     # 루틴 가이드
│
├── src/
│   ├── core/
│   │   ├── fsrs.py          # FSRS 간격반복 알고리즘
│   │   ├── knowledge.py     # 지식 청킹 & LLM 자동 분해
│   │   └── database.py      # SQLite 저장소 (Export/Import)
│   ├── adhd/
│   │   └── scheduler.py     # ADHD 최적화 스케줄러
│   ├── gamification/
│   │   └── engine.py        # XP/레벨/뱃지 시스템
│   ├── scheduler/
│   │   └── automation.py    # 자동 알림 & 루틴
│   ├── integrations/
│   │   ├── notifications.py # 모바일 알림 (Pushover, Telegram 등)
│   │   └── obsidian.py      # Obsidian 연동
│   ├── web/                  # 웹 대시보드
│   │   ├── server.py        # Flask REST API + LLM 엔드포인트
│   │   ├── templates/
│   │   │   ├── dashboard.html  # 메인 대시보드
│   │   │   ├── review.html     # 복습 세션
│   │   │   ├── knowledge.html  # 지식 추가
│   │   │   ├── library.html    # 지식 라이브러리
│   │   │   └── schedule.html   # 스케줄
│   │   └── static/
│   │       ├── css/style.css
│   │       └── js/dashboard.js
│   └── main.py              # CLI 인터페이스
│
├── data/                     # 데이터 (git 무시됨)
│   └── learning.db          # SQLite 데이터베이스
│
└── tests/                    # 테스트
```

### 데이터 저장 위치

모든 학습 데이터는 `data/learning.db` SQLite 파일에 저장됩니다:
- 지식 청크
- 플래시카드
- 복습 기록
- 사용자 통계
- 카테고리

**Export/Import**로 다른 환경으로 쉽게 이전 가능:
```bash
# 웹 UI에서: 라이브러리 > Export 버튼
# 또는 API: GET /api/export
```

---

## 설정

`config/settings.yaml`에서 시스템을 개인화할 수 있습니다:

```yaml
# ADHD 최적화 설정
adhd:
  pomodoro:
    work_duration: 25      # 작업 시간 (분)
    short_break: 5         # 짧은 휴식
    max_work_duration: 45  # 하이퍼포커스 제한

  energy_pattern:
    morning: "high"        # 오전: 고에너지
    afternoon: "medium"    # 오후: 중에너지
    evening: "low"         # 저녁: 저에너지

# 게이미피케이션
gamification:
  streak_multipliers:
    3: 1.1    # 3일: 10% 보너스
    7: 1.25   # 7일: 25% 보너스
    30: 2.0   # 30일: 100% 보너스
```

---

## 일일 루틴 예시

### 아침 (5분)
```
1. 시스템 시작 → 대시보드 확인
2. 오늘 복습 카드 수 확인
3. 에너지 레벨 자가 평가
```

### 학습 세션 (포모도로)
```
[25분] 새 지식 학습 (고에너지 시간)
[5분] 휴식 - 화면에서 떠나기
[25분] 복습
[5분] 휴식
... (4세션 후 15분 긴 휴식)
```

### 저녁 (5분)
```
1. 오늘 학습 기록 확인
2. 스트릭 유지 확인
3. 내일 계획
```

자세한 루틴은 [ROUTINE_GUIDE.md](config/ROUTINE_GUIDE.md)를 참조하세요.

---

## 웹 대시보드

웹 기반 인터랙티브 대시보드를 제공합니다:

```bash
# 웹 서버 시작
python src/web/server.py

# 브라우저에서 http://localhost:5000 접속
```

### 주요 기능

- **대시보드**: 레벨, XP, 스트릭, 일일 퀘스트, GitHub 스타일 히트맵 캘린더
- **복습 세션**:
  - 스페이스바로 카드 뒤집기
  - 화살표 키로 카드 간 이동
  - 카드 수 설정 (10/20/30/50/100/전체)
  - 카드 스킵 기능
  - 마크다운 렌더링 지원
- **지식 추가**: 제목, 내용, 유형, 태그로 지식 등록
- **지식 라이브러리**:
  - 유형/토픽/우선순위별 필터링
  - 지식 검색, 수정, 삭제
  - 카테고리 관리
  - JSON Export/Import (다른 환경으로 이전 가능)
- **포모도로 타이머**: 25분 집중 + 5분 휴식

### 키보드 단축키 (복습)

| 키 | 기능 |
|---|------|
| `Space` | 카드 뒤집기 |
| `1-4` | 평가 (다시/어려움/좋음/쉬움) |
| `←/→` | 이전/다음 카드로 이동 |
| `S` | 카드 스킵 |

---

## 다음 단계 (로드맵)

### Phase 1: 웹 대시보드 - 완료!
- [x] 인터랙티브 HTML 대시보드
- [x] GitHub 스타일 히트맵 캘린더
- [x] 실시간 통계 차트
- [x] 반응형 모바일 지원
- [x] 지식 라이브러리 (CRUD)
- [x] 카테고리 관리
- [x] Export/Import

### Phase 2: 통합
- [ ] Obsidian 연동 (마크다운 내보내기/가져오기)
- [ ] Anki 동기화
- [ ] Notion API 연동

### Phase 3: 모바일
- [ ] PWA (Progressive Web App)
- [ ] 푸시 알림
- [ ] 위젯

### Phase 4: AI 강화
- [ ] LLM 기반 자동 플래시카드 생성
- [ ] 개인화된 학습 경로 추천
- [ ] 지능형 복습 스케줄링

---

## 기여

기여를 환영합니다! 다음 방법으로 참여해주세요:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 참고 자료

### 학습 과학
- [FSRS Algorithm](https://github.com/open-spaced-repetition/fsrs4anki) - 최신 간격반복 알고리즘
- [Elaborative Interrogation](https://en.wikipedia.org/wiki/Elaborative_interrogation) - 정교화 질문 기법
- [Zettelkasten Method](https://zettelkasten.de/) - 지식 관리 방법론

### ADHD 전략
- [ADHD Timeboxing](https://llamalife.co/blog/what-is-timeboxing-and-why-is-it-crucial-for-adhd-time-management-and-productivity-clgn7muyw66122zpfl0gosa7c)
- [Hyperfocus Management](https://add.org/adhd-hyperfocus/)
- [Gamification for ADHD](https://www.tiimoapp.com/resource-hub/how-turning-chores-into-quests-can-make-your-neurodivergent-brain-happy)

### 도구
- [Habitica](https://habitica.com/) - 게이미피케이션 할일 앱
- [Forest](https://www.forestapp.cc/) - 집중 타이머
- [Obsidian](https://obsidian.md/) - 지식 관리

---

## 라이선스

MIT License - 자유롭게 사용, 수정, 배포할 수 있습니다.

---

*"완벽함보다 꾸준함이 중요합니다. 오늘 할 수 있는 만큼만 하세요."*
