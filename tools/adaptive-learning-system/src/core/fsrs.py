"""
FSRS (Free Spaced Repetition Scheduler) Algorithm Implementation
Based on: https://github.com/open-spaced-repetition/fsrs4anki

ADHD 최적화:
- 짧은 간격 옵션
- 난이도 자동 조정
- 에너지 레벨 기반 스케줄링
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Optional
import math


class Rating(IntEnum):
    """복습 응답 등급"""
    AGAIN = 1   # 완전히 잊음
    HARD = 2    # 어렵게 기억
    GOOD = 3    # 적절히 기억
    EASY = 4    # 쉽게 기억


class State(IntEnum):
    """카드 상태"""
    NEW = 0
    LEARNING = 1
    REVIEW = 2
    RELEARNING = 3


@dataclass
class ReviewLog:
    """복습 기록"""
    rating: Rating
    scheduled_days: int
    elapsed_days: int
    review_time: datetime
    state: State


@dataclass
class Card:
    """학습 카드 모델"""
    card_id: str
    content: str
    answer: str

    # FSRS 파라미터
    due: datetime = field(default_factory=datetime.now)
    stability: float = 0.0
    difficulty: float = 0.0
    elapsed_days: int = 0
    scheduled_days: int = 0
    reps: int = 0
    lapses: int = 0
    state: State = State.NEW
    last_review: Optional[datetime] = None

    # 메타데이터
    tags: list = field(default_factory=list)
    priority: int = 5  # 1-10, ADHD용 우선순위
    energy_required: str = "medium"  # low, medium, high
    created_at: datetime = field(default_factory=datetime.now)

    # 복습 기록
    review_logs: list = field(default_factory=list)


@dataclass
class FSRSParameters:
    """FSRS 알고리즘 파라미터 (FSRS-4.5 기반)"""
    # 초기 안정성 값 (등급별)
    w: list = field(default_factory=lambda: [
        0.4,    # w[0]: Again 초기 안정성
        0.6,    # w[1]: Hard 초기 안정성
        2.4,    # w[2]: Good 초기 안정성
        5.8,    # w[3]: Easy 초기 안정성
        4.93,   # w[4]: 난이도 기본값
        0.94,   # w[5]: 난이도 변화율
        0.86,   # w[6]: 난이도 평균회귀
        0.01,   # w[7]: 난이도 변동
        1.49,   # w[8]: 안정성 증가 (성공 시)
        0.14,   # w[9]: 안정성 감소 인자
        0.94,   # w[10]: 검색가능성 감쇠
        2.18,   # w[11]: 안정성 하한
        0.05,   # w[12]:
        0.34,   # w[13]:
        1.26,   # w[14]:
        0.29,   # w[15]:
        2.61,   # w[16]:
    ])

    # ADHD 최적화 파라미터
    request_retention: float = 0.9  # 목표 기억률 90%
    maximum_interval: int = 365
    easy_bonus: float = 1.3
    hard_factor: float = 1.2

    # ADHD 특화: 짧은 학습 간격
    learning_steps: list = field(default_factory=lambda: [1, 10])  # 분 단위
    relearning_steps: list = field(default_factory=lambda: [10])


class FSRS:
    """FSRS 스케줄러"""

    def __init__(self, params: Optional[FSRSParameters] = None):
        self.params = params or FSRSParameters()
        self.DECAY = -0.5
        self.FACTOR = 19/81

    def repeat(self, card: Card, now: datetime, rating: Rating) -> Card:
        """카드 복습 처리 및 다음 스케줄 계산"""
        card = self._copy_card(card)

        elapsed_days = 0 if card.state == State.NEW else (now - card.last_review).days
        card.elapsed_days = elapsed_days
        card.last_review = now
        card.reps += 1

        # 상태별 처리
        if card.state == State.NEW:
            card = self._handle_new_card(card, rating)
        elif card.state == State.LEARNING or card.state == State.RELEARNING:
            card = self._handle_learning_card(card, rating)
        else:  # REVIEW
            card = self._handle_review_card(card, rating, elapsed_days)

        # 복습 기록 추가
        card.review_logs.append(ReviewLog(
            rating=rating,
            scheduled_days=card.scheduled_days,
            elapsed_days=elapsed_days,
            review_time=now,
            state=card.state
        ))

        return card

    def _handle_new_card(self, card: Card, rating: Rating) -> Card:
        """새 카드 처리"""
        card.difficulty = self._init_difficulty(rating)
        card.stability = self._init_stability(rating)

        if rating == Rating.AGAIN:
            card.state = State.LEARNING
            card.scheduled_days = 0
            card.due = card.last_review + timedelta(minutes=self.params.learning_steps[0])
        elif rating == Rating.HARD:
            card.state = State.LEARNING
            card.scheduled_days = 0
            card.due = card.last_review + timedelta(minutes=self.params.learning_steps[-1])
        elif rating == Rating.GOOD:
            card.state = State.REVIEW
            card.scheduled_days = self._next_interval(card.stability)
            card.due = card.last_review + timedelta(days=card.scheduled_days)
        else:  # EASY
            card.state = State.REVIEW
            easy_interval = self._next_interval(card.stability) * self.params.easy_bonus
            card.scheduled_days = int(easy_interval)
            card.due = card.last_review + timedelta(days=card.scheduled_days)

        return card

    def _handle_learning_card(self, card: Card, rating: Rating) -> Card:
        """학습 중 카드 처리"""
        if rating == Rating.AGAIN:
            card.scheduled_days = 0
            card.due = card.last_review + timedelta(minutes=self.params.learning_steps[0])
        elif rating == Rating.HARD:
            card.scheduled_days = 0
            card.due = card.last_review + timedelta(minutes=self.params.learning_steps[-1])
        elif rating == Rating.GOOD:
            card.state = State.REVIEW
            card.scheduled_days = self._next_interval(card.stability)
            card.due = card.last_review + timedelta(days=card.scheduled_days)
        else:  # EASY
            card.state = State.REVIEW
            easy_interval = self._next_interval(card.stability) * self.params.easy_bonus
            card.scheduled_days = int(easy_interval)
            card.due = card.last_review + timedelta(days=card.scheduled_days)

        return card

    def _handle_review_card(self, card: Card, rating: Rating, elapsed_days: int) -> Card:
        """복습 카드 처리"""
        card.difficulty = self._next_difficulty(card.difficulty, rating)
        retrievability = self._forgetting_curve(elapsed_days, card.stability)
        card.stability = self._next_stability(card.stability, card.difficulty, retrievability, rating)

        if rating == Rating.AGAIN:
            card.state = State.RELEARNING
            card.lapses += 1
            card.scheduled_days = 0
            card.due = card.last_review + timedelta(minutes=self.params.relearning_steps[0])
        else:
            card.state = State.REVIEW
            interval = self._next_interval(card.stability)

            if rating == Rating.HARD:
                interval = max(1, int(interval * self.params.hard_factor))
            elif rating == Rating.EASY:
                interval = int(interval * self.params.easy_bonus)

            card.scheduled_days = min(interval, self.params.maximum_interval)
            card.due = card.last_review + timedelta(days=card.scheduled_days)

        return card

    def _init_difficulty(self, rating: Rating) -> float:
        """초기 난이도 계산"""
        w = self.params.w
        difficulty = w[4] - (rating.value - 3) * w[5]
        return min(max(difficulty, 1), 10)

    def _init_stability(self, rating: Rating) -> float:
        """초기 안정성 계산"""
        return self.params.w[rating.value - 1]

    def _next_difficulty(self, d: float, rating: Rating) -> float:
        """다음 난이도 계산"""
        w = self.params.w
        new_d = d - w[6] * (rating.value - 3)
        # 평균 회귀
        new_d = w[7] * w[4] + (1 - w[7]) * new_d
        return min(max(new_d, 1), 10)

    def _next_stability(self, s: float, d: float, r: float, rating: Rating) -> float:
        """다음 안정성 계산"""
        w = self.params.w

        if rating == Rating.AGAIN:
            return w[11] * math.pow(d, -w[12]) * (math.pow(s + 1, w[13]) - 1) * math.exp((1 - r) * w[14])

        hard_penalty = w[15] if rating == Rating.HARD else 1
        easy_bonus = w[16] if rating == Rating.EASY else 1

        return s * (1 + math.exp(w[8]) * (11 - d) * math.pow(s, -w[9]) *
                   (math.exp((1 - r) * w[10]) - 1) * hard_penalty * easy_bonus)

    def _forgetting_curve(self, elapsed_days: int, stability: float) -> float:
        """망각 곡선 - 검색가능성(R) 계산"""
        if stability == 0:
            return 0
        return math.pow(1 + self.FACTOR * elapsed_days / stability, self.DECAY)

    def _next_interval(self, stability: float) -> int:
        """다음 복습 간격 계산"""
        interval = stability / self.FACTOR * (math.pow(self.params.request_retention, 1/self.DECAY) - 1)
        return max(1, min(int(round(interval)), self.params.maximum_interval))

    def _copy_card(self, card: Card) -> Card:
        """카드 복사"""
        from copy import deepcopy
        return deepcopy(card)

    def get_retrievability(self, card: Card, now: datetime) -> float:
        """현재 검색가능성 계산"""
        if card.state == State.NEW:
            return 0
        elapsed = (now - card.last_review).days
        return self._forgetting_curve(elapsed, card.stability)


# 사용 예시
if __name__ == "__main__":
    fsrs = FSRS()

    # 새 카드 생성
    card = Card(
        card_id="test_001",
        content="Python의 GIL(Global Interpreter Lock)이란?",
        answer="GIL은 Python 인터프리터가 한 번에 하나의 스레드만 Python 바이트코드를 실행하도록 하는 뮤텍스입니다.",
        tags=["python", "concurrency"],
        priority=8,
        energy_required="high"
    )

    now = datetime.now()

    # 첫 번째 복습: GOOD 응답
    card = fsrs.repeat(card, now, Rating.GOOD)
    print(f"첫 복습 후:")
    print(f"  다음 복습: {card.due}")
    print(f"  안정성: {card.stability:.2f}")
    print(f"  난이도: {card.difficulty:.2f}")
    print(f"  간격: {card.scheduled_days}일")
