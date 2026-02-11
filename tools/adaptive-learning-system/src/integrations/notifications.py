"""
모바일 알림 시스템

지원 방법:
1. Pushover (iOS/Android 푸시 알림)
2. Telegram Bot
3. Email
4. Webhook (IFTTT, Zapier 등)
"""

import os
import json
import requests
from datetime import datetime
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@dataclass
class NotificationPayload:
    """알림 페이로드"""
    title: str
    message: str
    priority: int = 0  # -2 to 2 (Pushover 호환)
    url: Optional[str] = None
    sound: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class NotificationProvider(ABC):
    """알림 제공자 추상 클래스"""

    @abstractmethod
    def send(self, payload: NotificationPayload) -> bool:
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        pass


class PushoverProvider(NotificationProvider):
    """
    Pushover 알림 제공자
    https://pushover.net/

    설정 필요:
    - PUSHOVER_USER_KEY: 사용자 키
    - PUSHOVER_API_TOKEN: 앱 API 토큰
    """

    API_URL = "https://api.pushover.net/1/messages.json"

    def __init__(self, user_key: str = None, api_token: str = None):
        self.user_key = user_key or os.environ.get("PUSHOVER_USER_KEY")
        self.api_token = api_token or os.environ.get("PUSHOVER_API_TOKEN")

    def is_configured(self) -> bool:
        return bool(self.user_key and self.api_token)

    def send(self, payload: NotificationPayload) -> bool:
        if not self.is_configured():
            print("Pushover not configured")
            return False

        data = {
            "token": self.api_token,
            "user": self.user_key,
            "title": payload.title,
            "message": payload.message,
            "priority": payload.priority,
            "timestamp": int(payload.timestamp.timestamp()),
        }

        if payload.url:
            data["url"] = payload.url
            data["url_title"] = "열기"

        if payload.sound:
            data["sound"] = payload.sound

        try:
            response = requests.post(self.API_URL, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Pushover error: {e}")
            return False


class TelegramProvider(NotificationProvider):
    """
    Telegram Bot 알림 제공자

    설정 필요:
    - TELEGRAM_BOT_TOKEN: Bot 토큰 (@BotFather에서 생성)
    - TELEGRAM_CHAT_ID: 채팅 ID
    """

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, payload: NotificationPayload) -> bool:
        if not self.is_configured():
            print("Telegram not configured")
            return False

        # 이모지 기반 우선순위 표시
        priority_emoji = {
            -2: "💤",
            -1: "📝",
            0: "📢",
            1: "⚠️",
            2: "🚨"
        }
        emoji = priority_emoji.get(payload.priority, "📢")

        text = f"{emoji} *{payload.title}*\n\n{payload.message}"

        if payload.url:
            text += f"\n\n[열기]({payload.url})"

        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }

        try:
            url = self.API_URL.format(token=self.bot_token)
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False


class EmailProvider(NotificationProvider):
    """
    이메일 알림 제공자

    설정 필요:
    - EMAIL_SMTP_HOST
    - EMAIL_SMTP_PORT
    - EMAIL_USERNAME
    - EMAIL_PASSWORD
    - EMAIL_TO
    """

    def __init__(self, config: Dict = None):
        config = config or {}
        self.smtp_host = config.get("smtp_host") or os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(config.get("smtp_port") or os.environ.get("EMAIL_SMTP_PORT", 587))
        self.username = config.get("username") or os.environ.get("EMAIL_USERNAME")
        self.password = config.get("password") or os.environ.get("EMAIL_PASSWORD")
        self.to_email = config.get("to") or os.environ.get("EMAIL_TO")

    def is_configured(self) -> bool:
        return bool(self.username and self.password and self.to_email)

    def send(self, payload: NotificationPayload) -> bool:
        if not self.is_configured():
            print("Email not configured")
            return False

        msg = MIMEMultipart()
        msg["From"] = self.username
        msg["To"] = self.to_email
        msg["Subject"] = f"[ADHD Learning] {payload.title}"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #6366f1;">{payload.title}</h2>
            <p style="font-size: 16px; line-height: 1.6;">{payload.message}</p>
            {f'<p><a href="{payload.url}" style="color: #6366f1;">열기</a></p>' if payload.url else ''}
            <hr style="border: 1px solid #eee; margin: 20px 0;">
            <p style="color: #888; font-size: 12px;">
                ADHD Learning System - {payload.timestamp.strftime('%Y-%m-%d %H:%M')}
            </p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Email error: {e}")
            return False


class WebhookProvider(NotificationProvider):
    """
    Webhook 알림 제공자 (IFTTT, Zapier, Make 등)

    설정 필요:
    - WEBHOOK_URL: Webhook URL
    """

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.environ.get("WEBHOOK_URL")

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, payload: NotificationPayload) -> bool:
        if not self.is_configured():
            print("Webhook not configured")
            return False

        data = {
            "title": payload.title,
            "message": payload.message,
            "priority": payload.priority,
            "url": payload.url,
            "timestamp": payload.timestamp.isoformat()
        }

        try:
            response = requests.post(
                self.webhook_url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            return response.status_code in [200, 201, 204]
        except Exception as e:
            print(f"Webhook error: {e}")
            return False


class MobileNotificationManager:
    """모바일 알림 관리자"""

    def __init__(self):
        self.providers: List[NotificationProvider] = []
        self._init_providers()

    def _init_providers(self):
        """환경변수 기반 제공자 초기화"""
        providers = [
            PushoverProvider(),
            TelegramProvider(),
            EmailProvider(),
            WebhookProvider()
        ]

        for provider in providers:
            if provider.is_configured():
                self.providers.append(provider)
                print(f"Notification provider enabled: {provider.__class__.__name__}")

    def add_provider(self, provider: NotificationProvider):
        """수동 제공자 추가"""
        if provider.is_configured():
            self.providers.append(provider)

    def send(self, title: str, message: str, **kwargs) -> Dict[str, bool]:
        """모든 제공자로 알림 전송"""
        payload = NotificationPayload(
            title=title,
            message=message,
            priority=kwargs.get("priority", 0),
            url=kwargs.get("url"),
            sound=kwargs.get("sound")
        )

        results = {}
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            results[provider_name] = provider.send(payload)

        return results

    # ===== ADHD 학습 전용 알림 =====

    def send_study_reminder(self, cards_due: int = 0):
        """학습 리마인더"""
        if cards_due > 0:
            message = f"오늘 {cards_due}장의 복습 카드가 기다리고 있어요! 잠깐 시간 내볼까요?"
        else:
            message = "오늘 학습 시간이에요! 새로운 지식을 추가해보세요."

        return self.send(
            title="🧠 학습 시간!",
            message=message,
            priority=0,
            sound="pushover"
        )

    def send_streak_warning(self, current_streak: int):
        """스트릭 경고"""
        return self.send(
            title="🔥 스트릭 위험!",
            message=f"오늘 학습하지 않으면 {current_streak}일 스트릭이 끊깁니다!",
            priority=1,
            sound="siren"
        )

    def send_break_reminder(self, minutes: int = 5):
        """휴식 리마인더"""
        return self.send(
            title="☕ 휴식 시간!",
            message=f"집중 시간이 끝났습니다. {minutes}분 휴식하세요!",
            priority=0,
            sound="magic"
        )

    def send_achievement(self, achievement_name: str, description: str):
        """업적 달성"""
        return self.send(
            title=f"🏆 업적 달성: {achievement_name}",
            message=description,
            priority=0,
            sound="cosmic"
        )

    def send_level_up(self, new_level: int):
        """레벨업 알림"""
        return self.send(
            title=f"⭐ 레벨 업!",
            message=f"축하합니다! 레벨 {new_level}에 도달했습니다!",
            priority=0,
            sound="magic"
        )

    def send_weekly_report(self, stats: Dict):
        """주간 리포트"""
        message = f"""
이번 주 학습 현황:
• 활동 일수: {stats.get('active_days', 0)}일
• 복습 카드: {stats.get('total_cards', 0)}장
• 획득 XP: {stats.get('total_xp', 0)}
• 현재 스트릭: {stats.get('streak', 0)}일
        """.strip()

        return self.send(
            title="📊 주간 학습 리포트",
            message=message,
            priority=-1
        )


# 사용 예시
if __name__ == "__main__":
    manager = MobileNotificationManager()

    if manager.providers:
        print(f"Active providers: {len(manager.providers)}")

        # 테스트 알림
        results = manager.send(
            title="테스트 알림",
            message="ADHD Learning System이 정상 작동 중입니다!",
            priority=0
        )

        for provider, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {provider}")
    else:
        print("No notification providers configured.")
        print("\nTo enable notifications, set environment variables:")
        print("  Pushover: PUSHOVER_USER_KEY, PUSHOVER_API_TOKEN")
        print("  Telegram: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        print("  Email: EMAIL_SMTP_HOST, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO")
        print("  Webhook: WEBHOOK_URL")
