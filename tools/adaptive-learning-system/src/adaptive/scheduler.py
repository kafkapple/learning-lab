"""
적응형 학습 스케줄러

핵심 기능:
1. 에너지 레벨 기반 태스크 배치
2. 포모도로 + 몰입(Flow) 관리
3. 우선순위 기반 자동 스케줄링
4. 시각적 피드백 & 알림
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from typing import List, Optional, Callable
import random


class EnergyLevel(Enum):
    """에너지 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskType(Enum):
    """태스크 유형"""
    NEW_LEARNING = "new_learning"      # 새 지식 습득
    REVIEW = "review"                  # 복습
    PRACTICE = "practice"              # 실습/적용
    REFLECTION = "reflection"          # 회고/정리


@dataclass
class TimeBlock:
    """시간 블록"""
    start: datetime
    duration_minutes: int
    task_type: TaskType
    energy_level: EnergyLevel
    is_break: bool = False
    completed: bool = False

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=self.duration_minutes)


@dataclass
class PomodoroSession:
    """포모도로 세션"""
    work_duration: int = 25  # Focus: 기본 25분, 조절 가능
    short_break: int = 5
    long_break: int = 15
    sessions_before_long_break: int = 4
    current_session: int = 0

    # 몰입 특화 설정
    min_work_duration: int = 15  # 최소 집중 시간
    max_work_duration: int = 45  # Deep Work 최대 시간
    auto_adjust: bool = True     # 성과에 따라 자동 조정


@dataclass
class DailySchedule:
    """일일 스케줄"""
    date: datetime
    blocks: List[TimeBlock] = field(default_factory=list)
    total_study_minutes: int = 0
    total_review_cards: int = 0
    completed_blocks: int = 0

    # 에너지 패턴 (시간대별)
    energy_pattern: dict = field(default_factory=lambda: {
        "morning": EnergyLevel.HIGH,    # 9-12시
        "afternoon": EnergyLevel.MEDIUM,  # 12-17시
        "evening": EnergyLevel.LOW       # 17-21시
    })


class AdaptiveScheduler:
    """적응형 학습 스케줄러"""

    def __init__(self):
        self.pomodoro = PomodoroSession()
        self.daily_schedule: Optional[DailySchedule] = None

        # 몰입 최적화 설정
        self.max_new_items_per_day = 20      # 하루 최대 새 항목
        self.max_review_minutes = 60         # 최대 복습 시간
        self.variety_threshold = 3           # 같은 유형 연속 최대 횟수
        self.energy_task_mapping = {
            EnergyLevel.HIGH: [TaskType.NEW_LEARNING, TaskType.PRACTICE],
            EnergyLevel.MEDIUM: [TaskType.REVIEW, TaskType.NEW_LEARNING],
            EnergyLevel.LOW: [TaskType.REVIEW, TaskType.REFLECTION]
        }

    def create_daily_schedule(
        self,
        date: datetime,
        available_hours: List[tuple],  # [(시작시간, 종료시간), ...]
        cards_due: int,
        new_cards: int,
        energy_pattern: Optional[dict] = None
    ) -> DailySchedule:
        """
        일일 스케줄 생성

        Args:
            date: 대상 날짜
            available_hours: 학습 가능 시간대 [(9, 12), (14, 17)]
            cards_due: 복습할 카드 수
            new_cards: 새로 학습할 카드 수
            energy_pattern: 시간대별 에너지 패턴
        """
        schedule = DailySchedule(date=date)
        if energy_pattern:
            schedule.energy_pattern = energy_pattern

        blocks = []

        for start_hour, end_hour in available_hours:
            current_time = datetime.combine(date.date(), time(start_hour, 0))
            end_time = datetime.combine(date.date(), time(end_hour, 0))

            # 시간대별 에너지 레벨 결정
            energy = self._get_energy_for_time(start_hour, schedule.energy_pattern)

            # 포모도로 블록 생성
            session_count = 0
            while current_time < end_time:
                # 작업 블록
                work_block = TimeBlock(
                    start=current_time,
                    duration_minutes=self.pomodoro.work_duration,
                    task_type=self._select_task_type(energy, blocks),
                    energy_level=energy
                )
                blocks.append(work_block)
                current_time = work_block.end

                # 휴식 블록
                session_count += 1
                if session_count % self.pomodoro.sessions_before_long_break == 0:
                    break_duration = self.pomodoro.long_break
                else:
                    break_duration = self.pomodoro.short_break

                if current_time + timedelta(minutes=break_duration) <= end_time:
                    break_block = TimeBlock(
                        start=current_time,
                        duration_minutes=break_duration,
                        task_type=TaskType.REFLECTION,
                        energy_level=EnergyLevel.LOW,
                        is_break=True
                    )
                    blocks.append(break_block)
                    current_time = break_block.end

        schedule.blocks = blocks
        schedule.total_study_minutes = sum(
            b.duration_minutes for b in blocks if not b.is_break
        )

        self.daily_schedule = schedule
        return schedule

    def _get_energy_for_time(self, hour: int, pattern: dict) -> EnergyLevel:
        """시간대별 에너지 레벨 반환"""
        if 6 <= hour < 12:
            return pattern.get("morning", EnergyLevel.HIGH)
        elif 12 <= hour < 17:
            return pattern.get("afternoon", EnergyLevel.MEDIUM)
        else:
            return pattern.get("evening", EnergyLevel.LOW)

    def _select_task_type(self, energy: EnergyLevel, existing_blocks: List[TimeBlock]) -> TaskType:
        """에너지 레벨과 다양성을 고려한 태스크 유형 선택"""
        suitable_types = self.energy_task_mapping[energy]

        # 최근 블록의 유형 확인 (다양성 보장)
        recent_types = [b.task_type for b in existing_blocks[-self.variety_threshold:]]
        if recent_types and all(t == recent_types[0] for t in recent_types):
            # 같은 유형이 연속되면 다른 유형 선택
            other_types = [t for t in suitable_types if t != recent_types[0]]
            if other_types:
                return random.choice(other_types)

        return random.choice(suitable_types)

    def get_current_block(self, now: datetime) -> Optional[TimeBlock]:
        """현재 시간의 블록 반환"""
        if not self.daily_schedule:
            return None

        for block in self.daily_schedule.blocks:
            if block.start <= now < block.end:
                return block
        return None

    def get_next_block(self, now: datetime) -> Optional[TimeBlock]:
        """다음 블록 반환"""
        if not self.daily_schedule:
            return None

        for block in self.daily_schedule.blocks:
            if block.start > now:
                return block
        return None

    def adjust_pomodoro_duration(self, completion_rate: float, focus_quality: float):
        """
        성과에 따른 포모도로 시간 자동 조정

        Args:
            completion_rate: 완료율 (0-1)
            focus_quality: 집중 품질 (0-1, 자기 평가)
        """
        if not self.pomodoro.auto_adjust:
            return

        current = self.pomodoro.work_duration

        # 완료율과 집중도 모두 높으면 시간 증가
        if completion_rate > 0.8 and focus_quality > 0.7:
            new_duration = min(current + 5, self.pomodoro.max_work_duration)
        # 완료율이나 집중도가 낮으면 시간 감소
        elif completion_rate < 0.5 or focus_quality < 0.4:
            new_duration = max(current - 5, self.pomodoro.min_work_duration)
        else:
            new_duration = current

        self.pomodoro.work_duration = new_duration

    def generate_flow_alert(self, elapsed_minutes: int) -> Optional[str]:
        """Deep Work 경고 메시지 생성"""
        if elapsed_minutes >= self.pomodoro.max_work_duration:
            return f"[경고] {elapsed_minutes}분 연속 작업 중! 휴식이 필요합니다."
        elif elapsed_minutes >= self.pomodoro.max_work_duration - 5:
            return f"[알림] 곧 최대 집중 시간({self.pomodoro.max_work_duration}분)에 도달합니다."
        return None


@dataclass
class WeeklyPlan:
    """주간 계획"""
    week_start: datetime
    daily_schedules: List[DailySchedule] = field(default_factory=list)

    # 주간 목표
    total_new_items_goal: int = 100
    total_review_goal: int = 500
    focus_topics: List[str] = field(default_factory=list)

    # 진행 상황
    completed_new_items: int = 0
    completed_reviews: int = 0


class WeeklyPlanner:
    """주간 계획 생성기"""

    def __init__(self, scheduler: AdaptiveScheduler):
        self.scheduler = scheduler

    def create_weekly_plan(
        self,
        week_start: datetime,
        daily_availability: dict,  # {요일: [(시작, 종료), ...]}
        total_cards_due: int,
        new_cards_available: int,
        focus_topics: List[str]
    ) -> WeeklyPlan:
        """주간 계획 생성"""

        plan = WeeklyPlan(
            week_start=week_start,
            focus_topics=focus_topics
        )

        # 요일별 스케줄 생성
        days_of_week = ["monday", "tuesday", "wednesday", "thursday",
                        "friday", "saturday", "sunday"]

        # 카드 분배 계산
        active_days = sum(1 for d in days_of_week if daily_availability.get(d))
        cards_per_day = total_cards_due // max(active_days, 1)
        new_per_day = min(
            new_cards_available // max(active_days, 1),
            self.scheduler.max_new_items_per_day
        )

        for i, day_name in enumerate(days_of_week):
            day_date = week_start + timedelta(days=i)
            availability = daily_availability.get(day_name, [])

            if availability:
                daily_schedule = self.scheduler.create_daily_schedule(
                    date=day_date,
                    available_hours=availability,
                    cards_due=cards_per_day,
                    new_cards=new_per_day
                )
                plan.daily_schedules.append(daily_schedule)

        # 목표 설정
        plan.total_new_items_goal = new_per_day * active_days
        plan.total_review_goal = cards_per_day * active_days

        return plan


# 사용 예시
if __name__ == "__main__":
    scheduler = AdaptiveScheduler()

    # 일일 스케줄 생성
    today = datetime.now()
    schedule = scheduler.create_daily_schedule(
        date=today,
        available_hours=[(9, 12), (14, 17)],
        cards_due=50,
        new_cards=15
    )

    print("=== 오늘의 학습 스케줄 ===")
    for i, block in enumerate(schedule.blocks, 1):
        status = "[휴식]" if block.is_break else f"[{block.task_type.value}]"
        energy = f"에너지:{block.energy_level.value}"
        print(f"{i}. {block.start.strftime('%H:%M')}-{block.end.strftime('%H:%M')} "
              f"{status} ({block.duration_minutes}분) {energy}")

    print(f"\n총 학습 시간: {schedule.total_study_minutes}분")
