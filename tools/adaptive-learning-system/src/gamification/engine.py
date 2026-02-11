"""
ADHD 최적화 게이미피케이션 엔진

핵심 기능:
1. XP/레벨 시스템
2. 스트릭 & 연속 학습 보상
3. 도전 과제 & 뱃지
4. 즉각적 피드백 (도파민 최적화)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from enum import Enum
import random
import json


class BadgeType(Enum):
    """뱃지 유형"""
    STREAK = "streak"           # 연속 학습
    MASTERY = "mastery"         # 마스터리
    EXPLORER = "explorer"       # 탐험 (새 주제)
    PERFECTIONIST = "perfectionist"  # 완벽주의
    COMEBACK = "comeback"       # 복귀
    SPEEDSTER = "speedster"     # 빠른 학습
    NIGHT_OWL = "night_owl"     # 야간 학습
    EARLY_BIRD = "early_bird"   # 아침형


@dataclass
class Badge:
    """뱃지"""
    badge_id: str
    name: str
    description: str
    badge_type: BadgeType
    icon: str  # 이모지 또는 아이콘 이름
    earned_at: Optional[datetime] = None
    is_rare: bool = False


@dataclass
class Achievement:
    """업적"""
    achievement_id: str
    name: str
    description: str
    requirement: int  # 달성 조건 (숫자)
    current_progress: int = 0
    completed: bool = False
    xp_reward: int = 100
    badge: Optional[Badge] = None


@dataclass
class DailyQuest:
    """일일 퀘스트"""
    quest_id: str
    name: str
    description: str
    target: int
    current: int = 0
    xp_reward: int = 50
    completed: bool = False
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=1))


@dataclass
class UserStats:
    """사용자 통계"""
    user_id: str
    total_xp: int = 0
    level: int = 1
    current_streak: int = 0
    longest_streak: int = 0
    total_cards_reviewed: int = 0
    total_study_minutes: int = 0
    cards_mastered: int = 0

    # 일별 기록
    last_study_date: Optional[datetime] = None
    daily_xp: int = 0
    daily_cards: int = 0

    # 뱃지 & 업적
    badges: List[str] = field(default_factory=list)
    achievements: Dict[str, int] = field(default_factory=dict)  # achievement_id -> progress


class GamificationEngine:
    """게이미피케이션 엔진"""

    # 레벨업에 필요한 XP (레벨별) - 초반 더 쉽게, 점진적 상승
    # 레벨 1->2: 30XP (카드 3-5장)
    # 레벨 2->3: 60XP (카드 6-8장)
    # 레벨 3->4: 100XP (카드 10-12장)
    # 레벨 4->5: 150XP
    # 이후 점진적으로 증가
    LEVEL_XP = [0, 30, 90, 190, 340, 550, 850, 1250, 1800, 2500, 3400,
                4500, 5900, 7600, 9700, 12300, 15500, 19500, 24500, 30500, 38000]

    # 연속 학습 보너스 배율
    STREAK_MULTIPLIERS = {
        2: 1.05,  # 2일 연속: 5% 보너스
        3: 1.1,   # 3일 연속: 10% 보너스
        5: 1.15,  # 5일 연속: 15% 보너스
        7: 1.25,  # 7일 연속: 25% 보너스
        14: 1.5,  # 14일 연속: 50% 보너스
        30: 2.0,  # 30일 연속: 100% 보너스
    }

    def __init__(self):
        self.stats: Dict[str, UserStats] = {}
        self.badges = self._init_badges()
        self.achievements = self._init_achievements()
        self.callbacks: Dict[str, List[Callable]] = {
            "level_up": [],
            "badge_earned": [],
            "streak_milestone": [],
            "xp_gained": [],
        }

    def _init_badges(self) -> Dict[str, Badge]:
        """기본 뱃지 초기화"""
        badges = {
            "streak_3": Badge("streak_3", "3일 연속", "3일 연속 학습 달성", BadgeType.STREAK, "🔥"),
            "streak_7": Badge("streak_7", "일주일 전사", "7일 연속 학습 달성", BadgeType.STREAK, "⚔️"),
            "streak_30": Badge("streak_30", "월간 마스터", "30일 연속 학습 달성", BadgeType.STREAK, "👑", is_rare=True),
            "first_100": Badge("first_100", "첫 100장", "100장 카드 복습 완료", BadgeType.MASTERY, "📚"),
            "first_1000": Badge("first_1000", "천 리 길", "1000장 카드 복습 완료", BadgeType.MASTERY, "🏆", is_rare=True),
            "perfect_day": Badge("perfect_day", "완벽한 하루", "하루 모든 복습 100% 정답", BadgeType.PERFECTIONIST, "⭐"),
            "explorer_5": Badge("explorer_5", "탐험가", "5개 주제 학습", BadgeType.EXPLORER, "🗺️"),
            "comeback": Badge("comeback", "불사조", "7일 이상 휴식 후 복귀", BadgeType.COMEBACK, "🔄"),
            "speedster": Badge("speedster", "스피드러너", "10분 내 50장 복습", BadgeType.SPEEDSTER, "⚡"),
            "night_owl": Badge("night_owl", "야행성", "자정 이후 학습 10회", BadgeType.NIGHT_OWL, "🦉"),
            "early_bird": Badge("early_bird", "아침형 인간", "오전 6시 이전 학습 10회", BadgeType.EARLY_BIRD, "🐦"),
        }
        return badges

    def _init_achievements(self) -> Dict[str, Achievement]:
        """기본 업적 초기화"""
        achievements = {
            "cards_100": Achievement("cards_100", "시작이 반", "100장 카드 복습", 100, xp_reward=200),
            "cards_500": Achievement("cards_500", "중급자", "500장 카드 복습", 500, xp_reward=500),
            "cards_1000": Achievement("cards_1000", "고수의 길", "1000장 카드 복습", 1000, xp_reward=1000),
            "streak_7": Achievement("streak_7", "일주일 완주", "7일 연속 학습", 7, xp_reward=300),
            "streak_30": Achievement("streak_30", "한 달의 습관", "30일 연속 학습", 30, xp_reward=1000),
            "mastery_10": Achievement("mastery_10", "마스터 시작", "10장 카드 마스터", 10, xp_reward=200),
            "mastery_100": Achievement("mastery_100", "지식의 탑", "100장 카드 마스터", 100, xp_reward=1000),
            "study_hours_10": Achievement("study_hours_10", "열공 시작", "총 10시간 학습", 600, xp_reward=300),
            "study_hours_100": Achievement("study_hours_100", "학습 달인", "총 100시간 학습", 6000, xp_reward=2000),
        }
        return achievements

    def get_or_create_user(self, user_id: str) -> UserStats:
        """사용자 통계 조회 또는 생성"""
        if user_id not in self.stats:
            self.stats[user_id] = UserStats(user_id=user_id)
        return self.stats[user_id]

    def award_xp(self, user_id: str, base_xp: int, reason: str = "") -> Dict:
        """XP 부여"""
        stats = self.get_or_create_user(user_id)

        # 스트릭 보너스 적용
        multiplier = 1.0
        for streak_days, mult in sorted(self.STREAK_MULTIPLIERS.items()):
            if stats.current_streak >= streak_days:
                multiplier = mult

        final_xp = int(base_xp * multiplier)
        stats.total_xp += final_xp
        stats.daily_xp += final_xp

        result = {
            "base_xp": base_xp,
            "multiplier": multiplier,
            "final_xp": final_xp,
            "reason": reason,
            "new_total": stats.total_xp,
        }

        # 레벨업 체크
        new_level = self._calculate_level(stats.total_xp)
        if new_level > stats.level:
            stats.level = new_level
            result["level_up"] = True
            result["new_level"] = new_level
            self._trigger_callback("level_up", user_id, new_level)

        self._trigger_callback("xp_gained", user_id, result)
        return result

    def _calculate_level(self, total_xp: int) -> int:
        """레벨 계산"""
        for level, required_xp in enumerate(self.LEVEL_XP):
            if total_xp < required_xp:
                return max(1, level)
        return len(self.LEVEL_XP)

    def record_review(self, user_id: str, correct: bool, card_difficulty: int = 5) -> Dict:
        """복습 기록 및 보상"""
        stats = self.get_or_create_user(user_id)
        stats.total_cards_reviewed += 1
        stats.daily_cards += 1

        # 기본 XP: 정답 여부 + 난이도
        base_xp = 5 if correct else 2
        base_xp += card_difficulty // 2  # 난이도 보너스

        result = self.award_xp(user_id, base_xp, "card_review")

        # 업적 체크
        self._check_achievement(user_id, "cards_100", stats.total_cards_reviewed)
        self._check_achievement(user_id, "cards_500", stats.total_cards_reviewed)
        self._check_achievement(user_id, "cards_1000", stats.total_cards_reviewed)

        return result

    def record_mastery(self, user_id: str) -> Dict:
        """마스터리 달성 기록"""
        stats = self.get_or_create_user(user_id)
        stats.cards_mastered += 1

        result = self.award_xp(user_id, 50, "card_mastered")

        self._check_achievement(user_id, "mastery_10", stats.cards_mastered)
        self._check_achievement(user_id, "mastery_100", stats.cards_mastered)

        return result

    def update_streak(self, user_id: str) -> Dict:
        """스트릭 업데이트"""
        stats = self.get_or_create_user(user_id)
        today = datetime.now().date()

        result = {"streak_maintained": False, "streak_broken": False, "new_streak": 0}

        if stats.last_study_date is None:
            # 첫 학습
            stats.current_streak = 1
            result["new_streak"] = 1
        else:
            last_date = stats.last_study_date.date()
            days_diff = (today - last_date).days

            if days_diff == 0:
                # 같은 날
                result["streak_maintained"] = True
                result["new_streak"] = stats.current_streak
            elif days_diff == 1:
                # 연속
                stats.current_streak += 1
                result["streak_maintained"] = True
                result["new_streak"] = stats.current_streak

                # 스트릭 마일스톤 체크
                if stats.current_streak in [3, 7, 14, 30, 60, 100]:
                    self._trigger_callback("streak_milestone", user_id, stats.current_streak)
                    self._check_streak_badge(user_id, stats.current_streak)
            else:
                # 끊김
                if stats.current_streak >= 7:
                    # 복귀 뱃지 대상
                    self._award_badge(user_id, "comeback")
                stats.current_streak = 1
                result["streak_broken"] = True
                result["new_streak"] = 1

        # 최장 스트릭 업데이트
        if stats.current_streak > stats.longest_streak:
            stats.longest_streak = stats.current_streak

        stats.last_study_date = datetime.now()

        # 스트릭 업적 체크
        self._check_achievement(user_id, "streak_7", stats.current_streak)
        self._check_achievement(user_id, "streak_30", stats.current_streak)

        return result

    def _check_streak_badge(self, user_id: str, streak: int):
        """스트릭 뱃지 체크"""
        if streak >= 3:
            self._award_badge(user_id, "streak_3")
        if streak >= 7:
            self._award_badge(user_id, "streak_7")
        if streak >= 30:
            self._award_badge(user_id, "streak_30")

    def _award_badge(self, user_id: str, badge_id: str) -> bool:
        """뱃지 부여"""
        stats = self.get_or_create_user(user_id)

        if badge_id in stats.badges:
            return False  # 이미 보유

        if badge_id in self.badges:
            stats.badges.append(badge_id)
            badge = self.badges[badge_id]
            badge.earned_at = datetime.now()

            # 뱃지 획득 XP
            bonus_xp = 100 if badge.is_rare else 50
            self.award_xp(user_id, bonus_xp, f"badge_{badge_id}")

            self._trigger_callback("badge_earned", user_id, badge)
            return True

        return False

    def _check_achievement(self, user_id: str, achievement_id: str, progress: int):
        """업적 체크 및 완료 처리"""
        if achievement_id not in self.achievements:
            return

        achievement = self.achievements[achievement_id]
        if achievement.completed:
            return

        stats = self.get_or_create_user(user_id)
        stats.achievements[achievement_id] = progress

        if progress >= achievement.requirement:
            achievement.completed = True
            achievement.current_progress = achievement.requirement
            self.award_xp(user_id, achievement.xp_reward, f"achievement_{achievement_id}")

    def generate_daily_quests(self, user_id: str) -> List[DailyQuest]:
        """일일 퀘스트 생성"""
        stats = self.get_or_create_user(user_id)

        # 레벨에 따른 난이도 조정
        base_cards = 20 + (stats.level * 5)

        quests = [
            DailyQuest(
                quest_id=f"daily_review_{datetime.now().date()}",
                name="오늘의 복습",
                description=f"오늘 {base_cards}장 카드 복습하기",
                target=base_cards,
                xp_reward=100
            ),
            DailyQuest(
                quest_id=f"daily_new_{datetime.now().date()}",
                name="새로운 시작",
                description="새 카드 5장 학습하기",
                target=5,
                xp_reward=50
            ),
            DailyQuest(
                quest_id=f"daily_streak_{datetime.now().date()}",
                name="꾸준함의 힘",
                description="오늘도 학습 완료하기",
                target=1,
                xp_reward=30
            ),
        ]

        # 랜덤 보너스 퀘스트
        bonus_quests = [
            DailyQuest(
                quest_id=f"bonus_perfect_{datetime.now().date()}",
                name="퍼펙트 게임",
                description="연속 10장 정답 맞추기",
                target=10,
                xp_reward=150
            ),
            DailyQuest(
                quest_id=f"bonus_speed_{datetime.now().date()}",
                name="스피드 런",
                description="5분 안에 20장 복습하기",
                target=20,
                xp_reward=100
            ),
        ]

        quests.append(random.choice(bonus_quests))
        return quests

    def get_motivational_message(self, user_id: str) -> str:
        """동기 부여 메시지 생성"""
        stats = self.get_or_create_user(user_id)

        messages = {
            "streak_building": [
                f"🔥 {stats.current_streak}일 연속 학습 중! 조금만 더 하면 보너스!",
                f"💪 {stats.current_streak}일째 꾸준히 하고 있어요!",
            ],
            "level_progress": [
                f"⬆️ 레벨 {stats.level}! 다음 레벨까지 {self._xp_to_next_level(stats)}XP",
                f"📈 꾸준히 성장 중! 현재 레벨 {stats.level}",
            ],
            "encouragement": [
                "오늘도 한 걸음 더! 작은 진전이 큰 변화를 만듭니다.",
                "완벽하지 않아도 괜찮아요. 꾸준함이 답입니다!",
                "ADHD는 슈퍼파워! 하이퍼포커스를 활용하세요!",
            ]
        }

        if stats.current_streak >= 3:
            return random.choice(messages["streak_building"])
        elif stats.daily_xp > 0:
            return random.choice(messages["level_progress"])
        else:
            return random.choice(messages["encouragement"])

    def _xp_to_next_level(self, stats: UserStats) -> int:
        """다음 레벨까지 필요한 XP"""
        if stats.level >= len(self.LEVEL_XP) - 1:
            return 0
        next_level_xp = self.LEVEL_XP[stats.level]
        return max(0, next_level_xp - stats.total_xp)

    def register_callback(self, event: str, callback: Callable):
        """이벤트 콜백 등록"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def _trigger_callback(self, event: str, *args):
        """콜백 실행"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                print(f"Callback error: {e}")

    def get_dashboard_data(self, user_id: str) -> Dict:
        """대시보드 데이터"""
        stats = self.get_or_create_user(user_id)

        # 레벨 진행률 계산 (안전하게)
        if stats.level < len(self.LEVEL_XP) - 1:
            current_level_xp = self.LEVEL_XP[stats.level - 1] if stats.level > 0 else 0
            next_level_xp = self.LEVEL_XP[stats.level]
            level_range = next_level_xp - current_level_xp
            if level_range > 0:
                xp_progress = (stats.total_xp - current_level_xp) / level_range
            else:
                xp_progress = 1.0
        else:
            xp_progress = 1.0

        return {
            "level": stats.level,
            "total_xp": stats.total_xp,
            "xp_to_next": self._xp_to_next_level(stats),
            "xp_progress": min(max(xp_progress, 0), 1.0),  # 0~1 사이로 클램핑
            "current_streak": stats.current_streak,
            "longest_streak": stats.longest_streak,
            "total_cards": stats.total_cards_reviewed,
            "cards_mastered": stats.cards_mastered,
            "badges_earned": len(stats.badges),
            "today_xp": stats.daily_xp,
            "today_cards": stats.daily_cards,
            "motivational_message": self.get_motivational_message(user_id),
        }


# 사용 예시
if __name__ == "__main__":
    engine = GamificationEngine()

    # 콜백 등록
    engine.register_callback("level_up", lambda uid, lvl: print(f"🎉 레벨 업! {lvl}"))
    engine.register_callback("badge_earned", lambda uid, badge: print(f"🏅 뱃지 획득: {badge.icon} {badge.name}"))

    user_id = "user_001"

    # 스트릭 시작
    streak_result = engine.update_streak(user_id)
    print(f"스트릭: {streak_result}")

    # 복습 시뮬레이션
    for i in range(10):
        result = engine.record_review(user_id, correct=random.random() > 0.3, card_difficulty=random.randint(3, 8))
        if i % 5 == 0:
            print(f"복습 {i+1}: +{result['final_xp']}XP")

    # 대시보드
    dashboard = engine.get_dashboard_data(user_id)
    print("\n=== 대시보드 ===")
    print(f"레벨: {dashboard['level']}")
    print(f"XP: {dashboard['total_xp']} (다음 레벨까지 {dashboard['xp_to_next']})")
    print(f"스트릭: {dashboard['current_streak']}일")
    print(f"오늘 카드: {dashboard['today_cards']}장")
    print(f"\n{dashboard['motivational_message']}")
