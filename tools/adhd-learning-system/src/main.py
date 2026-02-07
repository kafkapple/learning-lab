"""
ADHD 학습 시스템 메인 오케스트레이터

모든 모듈을 통합하여 완전한 학습 경험 제공
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from typing import List, Optional, Dict
import time
import json

from core.fsrs import FSRS, Card, Rating, State, FSRSParameters
from core.knowledge import KnowledgeProcessor, KnowledgeChunk, KnowledgeType, Priority
from core.database import Database
from adhd.scheduler import ADHDScheduler, WeeklyPlanner, EnergyLevel, TaskType
from gamification.engine import GamificationEngine


class LearningSession:
    """학습 세션"""

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.db = Database()
        self.fsrs = FSRS()
        self.knowledge = KnowledgeProcessor()
        self.scheduler = ADHDScheduler()
        self.gamification = GamificationEngine()

        # 세션 상태
        self.session_start: Optional[datetime] = None
        self.cards_reviewed: int = 0
        self.correct_count: int = 0
        self.session_xp: int = 0

        # 콜백 설정
        self._setup_callbacks()

    def _setup_callbacks(self):
        """게이미피케이션 콜백 설정"""
        self.gamification.register_callback(
            "level_up",
            lambda uid, lvl: print(f"\n🎉 레벨 업! 레벨 {lvl}에 도달했습니다!")
        )
        self.gamification.register_callback(
            "badge_earned",
            lambda uid, badge: print(f"\n🏅 뱃지 획득: {badge.icon} {badge.name}")
        )
        self.gamification.register_callback(
            "streak_milestone",
            lambda uid, streak: print(f"\n🔥 {streak}일 연속 학습 달성!")
        )

    # ===== 지식 입력 =====

    def add_knowledge(
        self,
        title: str,
        content: str,
        knowledge_type: str = "concept",
        source: str = "",
        tags: List[str] = None,
        auto_create_cards: bool = True
    ) -> KnowledgeChunk:
        """새 지식 추가"""
        ktype = KnowledgeType[knowledge_type.upper()]

        chunk = self.knowledge.create_chunk(
            title=title,
            content=content,
            knowledge_type=ktype,
            source=source,
            tags=tags
        )

        # DB에 저장
        self.db.save_knowledge_chunk({
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "content": chunk.content,
            "knowledge_type": chunk.knowledge_type.value,
            "source": chunk.source,
            "tags": chunk.tags,
            "links": chunk.links,
            "parent_topic": chunk.parent_topic,
            "priority": chunk.priority.name,
            "difficulty": chunk.difficulty,
            "energy_required": chunk.energy_required,
            "why_questions": chunk.why_questions,
            "how_questions": chunk.how_questions,
            "what_if_questions": chunk.what_if_questions,
            "examples": chunk.examples,
            "prerequisites": chunk.prerequisites,
        })

        # 자동 플래시카드 생성
        if auto_create_cards:
            self._create_cards_from_chunk(chunk)

        print(f"✅ 지식 추가 완료: {title}")
        print(f"   우선순위: {chunk.priority.name}, 난이도: {chunk.difficulty}/10")
        print(f"   태그: {', '.join(chunk.tags)}")

        return chunk

    def _create_cards_from_chunk(self, chunk: KnowledgeChunk):
        """청크에서 플래시카드 생성"""
        # 기본 카드
        card = Card(
            card_id=f"card_{chunk.chunk_id}_main",
            content=chunk.title,
            answer=chunk.content,
            tags=chunk.tags,
            priority=chunk.priority.value,
            energy_required=chunk.energy_required
        )
        self._save_card(card)

        # 정교화 질문 카드들
        for i, why_q in enumerate(chunk.why_questions[:2]):
            q_card = Card(
                card_id=f"card_{chunk.chunk_id}_why_{i}",
                content=why_q,
                answer=f"[자유 답변] {chunk.title}과 관련하여 생각해보세요.",
                tags=chunk.tags + ["elaboration"],
                priority=chunk.priority.value + 1,
                energy_required="medium"
            )
            self._save_card(q_card)

    def _save_card(self, card: Card):
        """카드 DB 저장"""
        self.db.save_card({
            "card_id": card.card_id,
            "content": card.content,
            "answer": card.answer,
            "tags": card.tags,
            "priority": card.priority,
            "energy_required": card.energy_required,
            "due": card.due,
            "stability": card.stability,
            "difficulty": card.difficulty,
            "elapsed_days": card.elapsed_days,
            "scheduled_days": card.scheduled_days,
            "reps": card.reps,
            "lapses": card.lapses,
            "state": card.state.value,
            "last_review": card.last_review,
            "created_at": card.created_at,
            "updated_at": datetime.now(),
            "source": "",
            "parent_topic": None
        })

    # ===== 학습 세션 =====

    def start_session(self, energy_level: str = "medium"):
        """학습 세션 시작"""
        self.session_start = datetime.now()
        self.cards_reviewed = 0
        self.correct_count = 0
        self.session_xp = 0

        # 스트릭 업데이트
        streak_result = self.gamification.update_streak(self.user_id)

        # 대시보드 표시
        dashboard = self.gamification.get_dashboard_data(self.user_id)
        stats = self.db.get_statistics(self.user_id)

        print("\n" + "="*50)
        print("🧠 ADHD 학습 시스템 - 세션 시작")
        print("="*50)
        print(f"\n📊 대시보드")
        print(f"   레벨: {dashboard['level']} ({dashboard['total_xp']} XP)")
        print(f"   스트릭: {dashboard['current_streak']}일 🔥")
        print(f"   마스터한 카드: {dashboard['cards_mastered']}장")
        print(f"\n📝 오늘의 학습")
        print(f"   복습 대기: {stats['due_today']}장")
        print(f"   새 카드: {stats['new_cards']}장")
        print(f"\n💬 {dashboard['motivational_message']}")
        print("="*50)

        # 일일 퀘스트 표시
        quests = self.gamification.generate_daily_quests(self.user_id)
        print("\n🎯 오늘의 퀘스트")
        for quest in quests:
            status = "✅" if quest.completed else "⬜"
            print(f"   {status} {quest.name}: {quest.description} (+{quest.xp_reward}XP)")

        return stats

    def get_next_card(self, energy_level: str = None) -> Optional[Dict]:
        """다음 복습 카드 가져오기"""
        # 복습할 카드 먼저
        due_cards = self.db.get_due_cards(limit=1, energy_level=energy_level)
        if due_cards:
            return due_cards[0]

        # 없으면 새 카드
        new_cards = self.db.get_new_cards(limit=1)
        if new_cards:
            return new_cards[0]

        return None

    def review_card(self, card_data: Dict, rating: int) -> Dict:
        """카드 복습 처리"""
        # Card 객체 복원
        card = Card(
            card_id=card_data["card_id"],
            content=card_data["content"],
            answer=card_data["answer"],
            tags=card_data.get("tags", []),
            priority=card_data.get("priority", 5),
            energy_required=card_data.get("energy_required", "medium"),
            due=datetime.fromisoformat(card_data["due"]) if card_data.get("due") else datetime.now(),
            stability=card_data.get("stability", 0),
            difficulty=card_data.get("difficulty", 0),
            elapsed_days=card_data.get("elapsed_days", 0),
            scheduled_days=card_data.get("scheduled_days", 0),
            reps=card_data.get("reps", 0),
            lapses=card_data.get("lapses", 0),
            state=State(card_data.get("state", 0)),
            last_review=datetime.fromisoformat(card_data["last_review"]) if card_data.get("last_review") else None
        )

        # FSRS 복습 처리
        now = datetime.now()
        rating_enum = Rating(rating)
        updated_card = self.fsrs.repeat(card, now, rating_enum)

        # DB 업데이트
        self._save_card(updated_card)

        # 복습 기록 저장
        self.db.save_review_log(
            card_id=card.card_id,
            rating=rating,
            scheduled_days=updated_card.scheduled_days,
            elapsed_days=updated_card.elapsed_days,
            state=updated_card.state.value
        )

        # 게이미피케이션 처리
        is_correct = rating >= 3
        xp_result = self.gamification.record_review(
            self.user_id,
            correct=is_correct,
            card_difficulty=int(updated_card.difficulty)
        )

        # 마스터리 체크 (안정성이 충분히 높으면)
        if updated_card.stability >= 30 and updated_card.reps >= 5:
            self.gamification.record_mastery(self.user_id)

        # 세션 통계 업데이트
        self.cards_reviewed += 1
        if is_correct:
            self.correct_count += 1
        self.session_xp += xp_result["final_xp"]

        result = {
            "card_id": updated_card.card_id,
            "next_due": updated_card.due.isoformat(),
            "interval_days": updated_card.scheduled_days,
            "stability": updated_card.stability,
            "xp_earned": xp_result["final_xp"],
            "is_correct": is_correct,
            "session_cards": self.cards_reviewed,
            "session_accuracy": self.correct_count / self.cards_reviewed if self.cards_reviewed > 0 else 0
        }

        # 피드백 메시지
        if xp_result.get("level_up"):
            result["message"] = f"🎉 레벨 업! 레벨 {xp_result['new_level']}"
        elif is_correct:
            result["message"] = f"✅ 정답! +{xp_result['final_xp']}XP (다음 복습: {updated_card.scheduled_days}일 후)"
        else:
            result["message"] = f"❌ 다시 학습! +{xp_result['final_xp']}XP"

        return result

    def end_session(self) -> Dict:
        """세션 종료"""
        if not self.session_start:
            return {"error": "No active session"}

        duration = datetime.now() - self.session_start
        duration_minutes = int(duration.total_seconds() / 60)

        # 일일 기록 저장
        self.db.save_daily_record(
            self.user_id,
            datetime.now(),
            {
                "xp_earned": self.session_xp,
                "cards_reviewed": self.cards_reviewed,
                "study_minutes": duration_minutes,
            }
        )

        accuracy = self.correct_count / self.cards_reviewed if self.cards_reviewed > 0 else 0

        summary = {
            "duration_minutes": duration_minutes,
            "cards_reviewed": self.cards_reviewed,
            "accuracy": accuracy,
            "xp_earned": self.session_xp,
        }

        print("\n" + "="*50)
        print("📊 세션 완료!")
        print("="*50)
        print(f"   학습 시간: {duration_minutes}분")
        print(f"   복습 카드: {self.cards_reviewed}장")
        print(f"   정확도: {accuracy*100:.1f}%")
        print(f"   획득 XP: {self.session_xp}")
        print("="*50)

        # 세션 초기화
        self.session_start = None
        self.cards_reviewed = 0
        self.correct_count = 0
        self.session_xp = 0

        return summary

    # ===== 스케줄 관리 =====

    def create_daily_schedule(
        self,
        available_hours: List[tuple],
        energy_pattern: Dict[str, str] = None
    ):
        """일일 스케줄 생성"""
        stats = self.db.get_statistics(self.user_id)

        # 에너지 패턴 변환
        pattern = None
        if energy_pattern:
            pattern = {k: EnergyLevel[v.upper()] for k, v in energy_pattern.items()}

        schedule = self.scheduler.create_daily_schedule(
            date=datetime.now(),
            available_hours=available_hours,
            cards_due=stats["due_today"],
            new_cards=min(stats["new_cards"], 20),
            energy_pattern=pattern
        )

        print("\n📅 오늘의 학습 스케줄")
        print("-" * 40)
        for i, block in enumerate(schedule.blocks, 1):
            if block.is_break:
                print(f"   {block.start.strftime('%H:%M')} - 휴식 ({block.duration_minutes}분)")
            else:
                task_names = {
                    TaskType.NEW_LEARNING: "새 지식 학습",
                    TaskType.REVIEW: "복습",
                    TaskType.PRACTICE: "실습",
                    TaskType.REFLECTION: "회고/정리"
                }
                print(f"   {block.start.strftime('%H:%M')} - {task_names[block.task_type]} "
                      f"({block.duration_minutes}분, {block.energy_level.value})")
        print("-" * 40)
        print(f"   총 학습 시간: {schedule.total_study_minutes}분")

        return schedule

    # ===== 리포트 =====

    def get_weekly_report(self) -> Dict:
        """주간 리포트"""
        records = self.db.get_daily_records(self.user_id, days=7)
        stats = self.gamification.get_dashboard_data(self.user_id)

        total_xp = sum(r["xp_earned"] for r in records)
        total_cards = sum(r["cards_reviewed"] for r in records)
        total_minutes = sum(r["study_minutes"] for r in records)
        active_days = len([r for r in records if r["cards_reviewed"] > 0])

        report = {
            "period": "last_7_days",
            "total_xp": total_xp,
            "total_cards": total_cards,
            "total_minutes": total_minutes,
            "active_days": active_days,
            "current_streak": stats["current_streak"],
            "level": stats["level"],
            "daily_breakdown": records
        }

        print("\n📊 주간 리포트")
        print("="*40)
        print(f"   활동 일수: {active_days}/7일")
        print(f"   총 복습 카드: {total_cards}장")
        print(f"   총 학습 시간: {total_minutes}분")
        print(f"   획득 XP: {total_xp}")
        print(f"   현재 스트릭: {stats['current_streak']}일 🔥")
        print("="*40)

        return report


# CLI 인터페이스
def main():
    """CLI 메인 함수"""
    session = LearningSession()

    print("\n🧠 ADHD 학습 시스템에 오신 것을 환영합니다!")
    print("\n명령어:")
    print("  start    - 학습 세션 시작")
    print("  add      - 새 지식 추가")
    print("  schedule - 일일 스케줄 생성")
    print("  report   - 주간 리포트")
    print("  quit     - 종료")

    while True:
        try:
            cmd = input("\n> ").strip().lower()

            if cmd == "quit" or cmd == "q":
                print("학습 시스템을 종료합니다. 내일 또 만나요! 👋")
                break

            elif cmd == "start":
                session.start_session()

                # 간단한 복습 루프
                while True:
                    card = session.get_next_card()
                    if not card:
                        print("\n✨ 오늘의 복습을 모두 완료했습니다!")
                        break

                    print(f"\n📝 질문: {card['content']}")
                    input("   (엔터를 눌러 답변 확인)")
                    print(f"   답변: {card['answer']}")

                    print("\n   평가: 1=다시 2=어려움 3=좋음 4=쉬움 (q=종료)")
                    rating_input = input("   > ").strip()

                    if rating_input.lower() == 'q':
                        break

                    try:
                        rating = int(rating_input)
                        if 1 <= rating <= 4:
                            result = session.review_card(card, rating)
                            print(f"   {result['message']}")
                        else:
                            print("   1-4 사이의 숫자를 입력하세요.")
                    except ValueError:
                        print("   올바른 숫자를 입력하세요.")

                session.end_session()

            elif cmd == "add":
                title = input("제목: ").strip()
                content = input("내용: ").strip()
                tags_input = input("태그 (쉼표 구분): ").strip()
                tags = [t.strip() for t in tags_input.split(",")] if tags_input else []

                session.add_knowledge(title, content, tags=tags)

            elif cmd == "schedule":
                print("학습 가능 시간을 입력하세요 (예: 9-12, 14-17)")
                hours_input = input("> ").strip()
                hours = []
                for h in hours_input.split(","):
                    parts = h.strip().split("-")
                    if len(parts) == 2:
                        hours.append((int(parts[0]), int(parts[1])))

                if hours:
                    session.create_daily_schedule(hours)
                else:
                    print("올바른 형식으로 입력하세요.")

            elif cmd == "report":
                session.get_weekly_report()

            else:
                print("알 수 없는 명령어입니다.")

        except KeyboardInterrupt:
            print("\n\n학습 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
