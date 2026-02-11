"""
자동화 및 루틴 관리

기능:
1. 일일/주간 자동 스케줄링
2. 알림 시스템
3. 자동 백업
4. 주기적 리뷰 생성
"""

import schedule
import time
import subprocess
import platform
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional
from pathlib import Path
import json
import threading


class NotificationManager:
    """알림 관리자"""

    def __init__(self):
        self.system = platform.system()

    def send_notification(self, title: str, message: str, sound: bool = True):
        """시스템 알림 전송"""
        try:
            if self.system == "Darwin":  # macOS
                script = f'display notification "{message}" with title "{title}"'
                if sound:
                    script += ' sound name "default"'
                subprocess.run(["osascript", "-e", script], check=True)

            elif self.system == "Linux":
                subprocess.run(["notify-send", title, message], check=True)

            elif self.system == "Windows":
                # Windows Toast 알림 (PowerShell)
                ps_script = f'''
                [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                $template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02
                $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template)
                $text = $xml.GetElementsByTagName("text")
                $text[0].AppendChild($xml.CreateTextNode("{title}")) | Out-Null
                $text[1].AppendChild($xml.CreateTextNode("{message}")) | Out-Null
                $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
                [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Adaptive Learning").Show($toast)
                '''
                subprocess.run(["powershell", "-Command", ps_script], check=True)

        except Exception as e:
            print(f"알림 전송 실패: {e}")

    def send_learning_reminder(self):
        """학습 리마인더"""
        self.send_notification(
            "🧠 학습 시간!",
            "오늘의 복습 카드가 기다리고 있어요. 잠깐 시간 내볼까요?"
        )

    def send_break_reminder(self, minutes: int = 5):
        """휴식 리마인더"""
        self.send_notification(
            "☕ 휴식 시간!",
            f"집중 시간이 끝났습니다. {minutes}분 휴식하세요!"
        )

    def send_streak_warning(self, streak: int):
        """스트릭 경고"""
        self.send_notification(
            "🔥 스트릭 위험!",
            f"오늘 학습하지 않으면 {streak}일 스트릭이 끊깁니다!"
        )

    def send_achievement(self, achievement: str):
        """업적 달성 알림"""
        self.send_notification(
            "🏆 업적 달성!",
            achievement
        )


class DailyRoutine:
    """일일 루틴 관리"""

    def __init__(self, notification_manager: NotificationManager = None):
        self.notifications = notification_manager or NotificationManager()
        self.tasks: List[Dict] = []
        self.completed_today: List[str] = []

    def add_task(self, time_str: str, task_name: str, callback: Callable):
        """루틴 태스크 추가"""
        self.tasks.append({
            "time": time_str,
            "name": task_name,
            "callback": callback
        })
        schedule.every().day.at(time_str).do(callback)

    def setup_default_routine(self, config: Dict):
        """기본 일일 루틴 설정"""

        # 아침 리마인더
        morning_time = config.get("morning_reminder", "09:00")
        schedule.every().day.at(morning_time).do(
            self.notifications.send_learning_reminder
        )

        # 저녁 스트릭 경고
        evening_time = config.get("evening_warning", "20:00")
        schedule.every().day.at(evening_time).do(
            lambda: self._check_daily_progress()
        )

        # 자정 리셋
        schedule.every().day.at("00:00").do(self._daily_reset)

    def _check_daily_progress(self):
        """일일 진행 상황 체크"""
        # 여기에 실제 진행 상황 체크 로직
        # 학습하지 않았으면 경고
        if not self.completed_today:
            self.notifications.send_streak_warning(7)  # 예시

    def _daily_reset(self):
        """자정 리셋"""
        self.completed_today = []
        print(f"[{datetime.now()}] 일일 루틴 리셋 완료")


class WeeklyRoutine:
    """주간 루틴 관리"""

    def __init__(self, notification_manager: NotificationManager = None):
        self.notifications = notification_manager or NotificationManager()

    def setup_weekly_routine(self, config: Dict):
        """주간 루틴 설정"""

        # 주간 리뷰 (일요일 저녁)
        schedule.every().sunday.at("18:00").do(self._weekly_review)

        # 주간 계획 (월요일 아침)
        schedule.every().monday.at("08:00").do(self._weekly_planning)

    def _weekly_review(self):
        """주간 리뷰"""
        self.notifications.send_notification(
            "📊 주간 리뷰 시간",
            "이번 주 학습을 돌아보고 다음 주를 계획해보세요!"
        )

    def _weekly_planning(self):
        """주간 계획"""
        self.notifications.send_notification(
            "📝 새로운 한 주 시작!",
            "이번 주 학습 목표를 설정해보세요."
        )


class AutomationRunner:
    """자동화 실행기"""

    def __init__(self):
        self.notifications = NotificationManager()
        self.daily = DailyRoutine(self.notifications)
        self.weekly = WeeklyRoutine(self.notifications)
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def setup(self, config: Dict):
        """자동화 설정"""
        # 일일 루틴
        daily_config = config.get("daily", {})
        self.daily.setup_default_routine(daily_config)

        # 주간 루틴
        weekly_config = config.get("weekly", {})
        self.weekly.setup_weekly_routine(weekly_config)

        # 포모도로 알림 (예: 25분마다)
        work_duration = config.get("pomodoro", {}).get("work_duration", 25)
        # 실시간 포모도로는 별도 세션에서 관리

        print("자동화 설정 완료")
        self._print_schedule()

    def _print_schedule(self):
        """스케줄 출력"""
        print("\n📅 설정된 자동화 스케줄:")
        for job in schedule.get_jobs():
            print(f"   - {job}")

    def start(self):
        """백그라운드에서 스케줄러 시작"""
        if self.running:
            return

        self.running = True

        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크

        self._thread = threading.Thread(target=run_scheduler, daemon=True)
        self._thread.start()
        print("자동화 스케줄러 시작됨")

    def stop(self):
        """스케줄러 중지"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("자동화 스케줄러 중지됨")


class PomodoroTimer:
    """포모도로 타이머"""

    def __init__(
        self,
        work_duration: int = 25,
        short_break: int = 5,
        long_break: int = 15,
        sessions_before_long: int = 4
    ):
        self.work_duration = work_duration
        self.short_break = short_break
        self.long_break = long_break
        self.sessions_before_long = sessions_before_long

        self.current_session = 0
        self.is_working = False
        self.is_break = False
        self.start_time: Optional[datetime] = None

        self.notifications = NotificationManager()

        # 콜백
        self.on_work_start: Optional[Callable] = None
        self.on_work_end: Optional[Callable] = None
        self.on_break_start: Optional[Callable] = None
        self.on_break_end: Optional[Callable] = None

    def start_work(self):
        """작업 시작"""
        self.is_working = True
        self.is_break = False
        self.start_time = datetime.now()
        self.current_session += 1

        print(f"\n🍅 포모도로 #{self.current_session} 시작! ({self.work_duration}분)")

        if self.on_work_start:
            self.on_work_start()

        # 타이머 (실제 구현에서는 비동기로)
        return self.work_duration * 60  # 초 단위 반환

    def end_work(self):
        """작업 종료"""
        self.is_working = False
        elapsed = (datetime.now() - self.start_time).seconds // 60 if self.start_time else 0

        print(f"\n✅ 작업 완료! ({elapsed}분)")
        self.notifications.send_break_reminder(self._get_break_duration())

        if self.on_work_end:
            self.on_work_end()

    def start_break(self):
        """휴식 시작"""
        self.is_break = True
        self.start_time = datetime.now()
        break_duration = self._get_break_duration()

        break_type = "긴 휴식" if break_duration == self.long_break else "짧은 휴식"
        print(f"\n☕ {break_type} 시작! ({break_duration}분)")

        if self.on_break_start:
            self.on_break_start()

        return break_duration * 60

    def end_break(self):
        """휴식 종료"""
        self.is_break = False
        print("\n🔔 휴식 종료!")

        self.notifications.send_notification(
            "🍅 다음 포모도로",
            "휴식이 끝났습니다. 다음 세션을 시작할 준비가 되셨나요?"
        )

        if self.on_break_end:
            self.on_break_end()

    def _get_break_duration(self) -> int:
        """휴식 시간 계산"""
        if self.current_session % self.sessions_before_long == 0:
            return self.long_break
        return self.short_break

    def get_status(self) -> Dict:
        """현재 상태"""
        elapsed = 0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).seconds // 60

        return {
            "session": self.current_session,
            "is_working": self.is_working,
            "is_break": self.is_break,
            "elapsed_minutes": elapsed,
            "next_break_type": "long" if (self.current_session + 1) % self.sessions_before_long == 0 else "short"
        }


# 사용 예시
if __name__ == "__main__":
    # 알림 테스트
    notifier = NotificationManager()
    notifier.send_notification("테스트", "적응형 학습 시스템 알림 테스트입니다!")

    # 자동화 설정
    automation = AutomationRunner()
    automation.setup({
        "daily": {
            "morning_reminder": "09:00",
            "evening_warning": "20:00"
        },
        "weekly": {},
        "pomodoro": {
            "work_duration": 25
        }
    })

    # 포모도로 테스트
    pomodoro = PomodoroTimer(work_duration=25, short_break=5)
    print("\n포모도로 상태:", pomodoro.get_status())

    print("\n자동화 시스템 준비 완료!")
    print("실제 운영 시 automation.start()를 호출하세요.")
