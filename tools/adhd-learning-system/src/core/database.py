"""
SQLite 기반 데이터 저장소

모든 학습 데이터의 영구 저장 및 조회
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import contextmanager


class Database:
    """학습 데이터베이스"""

    def __init__(self, db_path: str = "data/learning.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """컨텍스트 매니저로 연결 관리"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        """데이터베이스 초기화"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 카드 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cards (
                    card_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    tags TEXT,
                    priority INTEGER DEFAULT 5,
                    energy_required TEXT DEFAULT 'medium',

                    -- FSRS 파라미터
                    due TEXT,
                    stability REAL DEFAULT 0,
                    difficulty REAL DEFAULT 0,
                    elapsed_days INTEGER DEFAULT 0,
                    scheduled_days INTEGER DEFAULT 0,
                    reps INTEGER DEFAULT 0,
                    lapses INTEGER DEFAULT 0,
                    state INTEGER DEFAULT 0,
                    last_review TEXT,

                    -- 메타데이터
                    created_at TEXT,
                    updated_at TEXT,
                    source TEXT,
                    parent_topic TEXT
                )
            """)

            # 복습 기록 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    scheduled_days INTEGER,
                    elapsed_days INTEGER,
                    review_time TEXT,
                    state INTEGER,
                    response_time_ms INTEGER,
                    FOREIGN KEY (card_id) REFERENCES cards(card_id)
                )
            """)

            # 사용자 통계 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_stats (
                    user_id TEXT PRIMARY KEY,
                    total_xp INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 1,
                    current_streak INTEGER DEFAULT 0,
                    longest_streak INTEGER DEFAULT 0,
                    total_cards_reviewed INTEGER DEFAULT 0,
                    total_study_minutes INTEGER DEFAULT 0,
                    cards_mastered INTEGER DEFAULT 0,
                    last_study_date TEXT,
                    badges TEXT,
                    achievements TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            # 일일 기록 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    xp_earned INTEGER DEFAULT 0,
                    cards_reviewed INTEGER DEFAULT 0,
                    cards_new INTEGER DEFAULT 0,
                    study_minutes INTEGER DEFAULT 0,
                    perfect_reviews INTEGER DEFAULT 0,
                    created_at TEXT,
                    UNIQUE(user_id, date)
                )
            """)

            # 지식 청크 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    knowledge_type TEXT,
                    source TEXT,
                    tags TEXT,
                    links TEXT,
                    parent_topic TEXT,
                    priority TEXT,
                    difficulty INTEGER,
                    energy_required TEXT,
                    why_questions TEXT,
                    how_questions TEXT,
                    what_if_questions TEXT,
                    examples TEXT,
                    prerequisites TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            # 카테고리 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    category_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    icon TEXT DEFAULT '📁',
                    color TEXT DEFAULT '#6366f1',
                    parent_id TEXT,
                    sort_order INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (parent_id) REFERENCES categories(category_id)
                )
            """)

            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_due ON cards(due)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_tags ON cards(tags)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_state ON cards(state)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_logs_card ON review_logs(card_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_records_date ON daily_records(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON knowledge_chunks(knowledge_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_topic ON knowledge_chunks(parent_topic)")

    # ===== 카드 관련 =====

    def save_card(self, card_data: Dict) -> bool:
        """카드 저장"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # tags를 JSON으로 변환
            if "tags" in card_data and isinstance(card_data["tags"], list):
                card_data["tags"] = json.dumps(card_data["tags"])

            # 날짜를 문자열로 변환
            for date_field in ["due", "last_review", "created_at", "updated_at"]:
                if date_field in card_data and isinstance(card_data[date_field], datetime):
                    card_data[date_field] = card_data[date_field].isoformat()

            cursor.execute("""
                INSERT OR REPLACE INTO cards
                (card_id, content, answer, tags, priority, energy_required,
                 due, stability, difficulty, elapsed_days, scheduled_days,
                 reps, lapses, state, last_review, created_at, updated_at, source, parent_topic)
                VALUES
                (:card_id, :content, :answer, :tags, :priority, :energy_required,
                 :due, :stability, :difficulty, :elapsed_days, :scheduled_days,
                 :reps, :lapses, :state, :last_review, :created_at, :updated_at, :source, :parent_topic)
            """, card_data)

            return True

    def get_card(self, card_id: str) -> Optional[Dict]:
        """카드 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cards WHERE card_id = ?", (card_id,))
            row = cursor.fetchone()

            if row:
                data = dict(row)
                if data.get("tags"):
                    data["tags"] = json.loads(data["tags"])
                return data
            return None

    def get_due_cards(self, limit: int = 100, energy_level: str = None) -> List[Dict]:
        """복습할 카드 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            query = "SELECT * FROM cards WHERE due <= ? AND state > 0"
            params = [now]

            if energy_level:
                query += " AND energy_required = ?"
                params.append(energy_level)

            query += " ORDER BY due ASC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            cards = []
            for row in cursor.fetchall():
                data = dict(row)
                if data.get("tags"):
                    data["tags"] = json.loads(data["tags"])
                cards.append(data)

            return cards

    def get_new_cards(self, limit: int = 20, topic: str = None) -> List[Dict]:
        """새 카드 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM cards WHERE state = 0"
            params = []

            if topic:
                query += " AND parent_topic = ?"
                params.append(topic)

            query += " ORDER BY priority ASC, created_at ASC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            cards = []
            for row in cursor.fetchall():
                data = dict(row)
                if data.get("tags"):
                    data["tags"] = json.loads(data["tags"])
                cards.append(data)

            return cards

    # ===== 복습 기록 =====

    def save_review_log(self, card_id: str, rating: int, scheduled_days: int,
                       elapsed_days: int, state: int, response_time_ms: int = 0):
        """복습 기록 저장"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO review_logs
                (card_id, rating, scheduled_days, elapsed_days, review_time, state, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (card_id, rating, scheduled_days, elapsed_days,
                  datetime.now().isoformat(), state, response_time_ms))

    def get_review_history(self, card_id: str, limit: int = 50) -> List[Dict]:
        """카드 복습 이력 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM review_logs
                WHERE card_id = ?
                ORDER BY review_time DESC
                LIMIT ?
            """, (card_id, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_reviews_by_date(self, date: str) -> List[Dict]:
        """특정 날짜의 복습 기록 조회 (카드 정보 포함)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    r.id,
                    r.card_id,
                    r.rating,
                    r.review_time,
                    r.state,
                    c.content as question,
                    c.answer,
                    c.tags,
                    c.parent_topic,
                    k.title as knowledge_title,
                    k.chunk_id
                FROM review_logs r
                LEFT JOIN cards c ON r.card_id = c.card_id
                LEFT JOIN knowledge_chunks k ON c.card_id LIKE '%' || k.chunk_id || '%'
                WHERE date(r.review_time) = ?
                ORDER BY r.review_time DESC
            """, (date,))

            reviews = []
            for row in cursor.fetchall():
                data = dict(row)
                if data.get("tags"):
                    try:
                        data["tags"] = json.loads(data["tags"])
                    except:
                        pass
                reviews.append(data)

            return reviews

    # ===== 사용자 통계 =====

    def save_user_stats(self, stats_data: Dict):
        """사용자 통계 저장"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 리스트/딕트를 JSON으로 변환
            if isinstance(stats_data.get("badges"), list):
                stats_data["badges"] = json.dumps(stats_data["badges"])
            if isinstance(stats_data.get("achievements"), dict):
                stats_data["achievements"] = json.dumps(stats_data["achievements"])

            # 날짜 변환
            if isinstance(stats_data.get("last_study_date"), datetime):
                stats_data["last_study_date"] = stats_data["last_study_date"].isoformat()

            stats_data["updated_at"] = datetime.now().isoformat()

            cursor.execute("""
                INSERT OR REPLACE INTO user_stats
                (user_id, total_xp, level, current_streak, longest_streak,
                 total_cards_reviewed, total_study_minutes, cards_mastered,
                 last_study_date, badges, achievements, created_at, updated_at)
                VALUES
                (:user_id, :total_xp, :level, :current_streak, :longest_streak,
                 :total_cards_reviewed, :total_study_minutes, :cards_mastered,
                 :last_study_date, :badges, :achievements,
                 COALESCE(:created_at, datetime('now')), :updated_at)
            """, stats_data)

    def get_user_stats(self, user_id: str) -> Optional[Dict]:
        """사용자 통계 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

            if row:
                data = dict(row)
                if data.get("badges"):
                    data["badges"] = json.loads(data["badges"])
                if data.get("achievements"):
                    data["achievements"] = json.loads(data["achievements"])
                return data
            return None

    # ===== 일일 기록 =====

    def save_daily_record(self, user_id: str, date: datetime, record: Dict):
        """일일 기록 저장"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            date_str = date.strftime("%Y-%m-%d")

            cursor.execute("""
                INSERT OR REPLACE INTO daily_records
                (user_id, date, xp_earned, cards_reviewed, cards_new,
                 study_minutes, perfect_reviews, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, date_str, record.get("xp_earned", 0),
                  record.get("cards_reviewed", 0), record.get("cards_new", 0),
                  record.get("study_minutes", 0), record.get("perfect_reviews", 0),
                  datetime.now().isoformat()))

    def get_daily_records(self, user_id: str, days: int = 30) -> List[Dict]:
        """일일 기록 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM daily_records
                WHERE user_id = ?
                ORDER BY date DESC
                LIMIT ?
            """, (user_id, days))

            return [dict(row) for row in cursor.fetchall()]

    # ===== 지식 청크 =====

    def save_knowledge_chunk(self, chunk_data: Dict):
        """지식 청크 저장"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 리스트를 JSON으로 변환
            for field in ["tags", "links", "why_questions", "how_questions",
                         "what_if_questions", "examples", "prerequisites"]:
                if field in chunk_data and isinstance(chunk_data[field], list):
                    chunk_data[field] = json.dumps(chunk_data[field])

            chunk_data["updated_at"] = datetime.now().isoformat()
            if "created_at" not in chunk_data:
                chunk_data["created_at"] = chunk_data["updated_at"]

            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_chunks
                (chunk_id, title, content, knowledge_type, source, tags, links,
                 parent_topic, priority, difficulty, energy_required,
                 why_questions, how_questions, what_if_questions, examples,
                 prerequisites, created_at, updated_at)
                VALUES
                (:chunk_id, :title, :content, :knowledge_type, :source, :tags, :links,
                 :parent_topic, :priority, :difficulty, :energy_required,
                 :why_questions, :how_questions, :what_if_questions, :examples,
                 :prerequisites, :created_at, :updated_at)
            """, chunk_data)

    def get_statistics(self, user_id: str) -> Dict:
        """종합 통계 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 총 카드 수
            cursor.execute("SELECT COUNT(*) FROM cards")
            total_cards = cursor.fetchone()[0]

            # 상태별 카드 수
            cursor.execute("""
                SELECT state, COUNT(*)
                FROM cards
                GROUP BY state
            """)
            cards_by_state = {row[0]: row[1] for row in cursor.fetchall()}

            # 오늘 복습 예정
            cursor.execute("""
                SELECT COUNT(*) FROM cards
                WHERE due <= datetime('now') AND state > 0
            """)
            due_today = cursor.fetchone()[0]

            # 주간 복습 수
            cursor.execute("""
                SELECT COUNT(*) FROM review_logs
                WHERE review_time >= datetime('now', '-7 days')
            """)
            weekly_reviews = cursor.fetchone()[0]

            return {
                "total_cards": total_cards,
                "new_cards": cards_by_state.get(0, 0),
                "learning_cards": cards_by_state.get(1, 0) + cards_by_state.get(3, 0),
                "review_cards": cards_by_state.get(2, 0),
                "due_today": due_today,
                "weekly_reviews": weekly_reviews,
            }


    # ===== 카테고리 관리 =====

    def save_category(self, category_data: Dict) -> bool:
        """카테고리 저장"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            category_data["updated_at"] = datetime.now().isoformat()
            if "created_at" not in category_data:
                category_data["created_at"] = category_data["updated_at"]

            cursor.execute("""
                INSERT OR REPLACE INTO categories
                (category_id, name, description, icon, color, parent_id, sort_order, created_at, updated_at)
                VALUES
                (:category_id, :name, :description, :icon, :color, :parent_id, :sort_order, :created_at, :updated_at)
            """, category_data)
            return True

    def get_category(self, category_id: str) -> Optional[Dict]:
        """카테고리 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM categories WHERE category_id = ?", (category_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_categories(self) -> List[Dict]:
        """모든 카테고리 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM categories ORDER BY sort_order, name")
            return [dict(row) for row in cursor.fetchall()]

    def delete_category(self, category_id: str) -> bool:
        """카테고리 삭제"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM categories WHERE category_id = ?", (category_id,))
            return cursor.rowcount > 0

    # ===== 지식 청크 CRUD =====

    def get_knowledge_chunk(self, chunk_id: str) -> Optional[Dict]:
        """지식 청크 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                for field in ["tags", "links", "why_questions", "how_questions",
                             "what_if_questions", "examples", "prerequisites"]:
                    if data.get(field):
                        try:
                            data[field] = json.loads(data[field])
                        except:
                            pass
                return data
            return None

    def get_all_knowledge_chunks(self, knowledge_type: str = None,
                                  parent_topic: str = None,
                                  limit: int = 500) -> List[Dict]:
        """모든 지식 청크 조회 (필터 옵션)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM knowledge_chunks WHERE 1=1"
            params = []

            if knowledge_type:
                query += " AND knowledge_type = ?"
                params.append(knowledge_type)
            if parent_topic:
                query += " AND parent_topic = ?"
                params.append(parent_topic)

            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            chunks = []
            for row in cursor.fetchall():
                data = dict(row)
                for field in ["tags", "links", "why_questions", "how_questions",
                             "what_if_questions", "examples", "prerequisites"]:
                    if data.get(field):
                        try:
                            data[field] = json.loads(data[field])
                        except:
                            pass
                chunks.append(data)
            return chunks

    def update_knowledge_chunk(self, chunk_id: str, updates: Dict) -> bool:
        """지식 청크 수정"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 기존 데이터 조회
            existing = self.get_knowledge_chunk(chunk_id)
            if not existing:
                return False

            # 업데이트 병합
            for key, value in updates.items():
                existing[key] = value

            existing["updated_at"] = datetime.now().isoformat()

            # 리스트를 JSON으로 변환
            for field in ["tags", "links", "why_questions", "how_questions",
                         "what_if_questions", "examples", "prerequisites"]:
                if field in existing and isinstance(existing[field], list):
                    existing[field] = json.dumps(existing[field])

            cursor.execute("""
                UPDATE knowledge_chunks SET
                    title = :title,
                    content = :content,
                    knowledge_type = :knowledge_type,
                    source = :source,
                    tags = :tags,
                    links = :links,
                    parent_topic = :parent_topic,
                    priority = :priority,
                    difficulty = :difficulty,
                    energy_required = :energy_required,
                    why_questions = :why_questions,
                    how_questions = :how_questions,
                    what_if_questions = :what_if_questions,
                    examples = :examples,
                    prerequisites = :prerequisites,
                    updated_at = :updated_at
                WHERE chunk_id = :chunk_id
            """, existing)

            return True

    def delete_knowledge_chunk(self, chunk_id: str) -> bool:
        """지식 청크 삭제"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 연관 카드도 삭제
            cursor.execute("DELETE FROM cards WHERE card_id LIKE ?", (f"card_{chunk_id}%",))
            cursor.execute("DELETE FROM knowledge_chunks WHERE chunk_id = ?", (chunk_id,))
            return cursor.rowcount > 0

    def delete_card(self, card_id: str) -> bool:
        """카드 삭제"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM review_logs WHERE card_id = ?", (card_id,))
            cursor.execute("DELETE FROM cards WHERE card_id = ?", (card_id,))
            return cursor.rowcount > 0

    def get_all_cards(self, limit: int = 500) -> List[Dict]:
        """모든 카드 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cards ORDER BY created_at DESC LIMIT ?", (limit,))
            cards = []
            for row in cursor.fetchall():
                data = dict(row)
                if data.get("tags"):
                    try:
                        data["tags"] = json.loads(data["tags"])
                    except:
                        pass
                cards.append(data)
            return cards

    def get_knowledge_overview(self) -> Dict:
        """지식 개요 통계 (유형별, 태그별)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 유형별 카운트
            cursor.execute("""
                SELECT knowledge_type, COUNT(*) as count
                FROM knowledge_chunks
                GROUP BY knowledge_type
            """)
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

            # 토픽별 카운트
            cursor.execute("""
                SELECT parent_topic, COUNT(*) as count
                FROM knowledge_chunks
                WHERE parent_topic IS NOT NULL AND parent_topic != ''
                GROUP BY parent_topic
                ORDER BY count DESC
            """)
            by_topic = {row[0]: row[1] for row in cursor.fetchall()}

            # 우선순위별 카운트
            cursor.execute("""
                SELECT priority, COUNT(*) as count
                FROM knowledge_chunks
                GROUP BY priority
            """)
            by_priority = {row[0]: row[1] for row in cursor.fetchall()}

            # 총 개수
            cursor.execute("SELECT COUNT(*) FROM knowledge_chunks")
            total_chunks = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM cards")
            total_cards = cursor.fetchone()[0]

            # 최근 추가
            cursor.execute("""
                SELECT chunk_id, title, knowledge_type, created_at
                FROM knowledge_chunks
                ORDER BY created_at DESC
                LIMIT 10
            """)
            recent = [dict(row) for row in cursor.fetchall()]

            return {
                "total_chunks": total_chunks,
                "total_cards": total_cards,
                "by_type": by_type,
                "by_topic": by_topic,
                "by_priority": by_priority,
                "recent": recent
            }

    # ===== Export / Import =====

    def export_all_data(self) -> Dict:
        """전체 데이터 내보내기 (JSON 형식)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 지식 청크
            chunks = self.get_all_knowledge_chunks(limit=10000)

            # 카드
            cards = self.get_all_cards(limit=10000)

            # 카테고리
            categories = self.get_all_categories()

            # 복습 기록
            cursor.execute("SELECT * FROM review_logs ORDER BY review_time DESC LIMIT 10000")
            review_logs = [dict(row) for row in cursor.fetchall()]

            return {
                "export_version": "1.0",
                "export_date": datetime.now().isoformat(),
                "knowledge_chunks": chunks,
                "cards": cards,
                "categories": categories,
                "review_logs": review_logs,
                "statistics": self.get_knowledge_overview()
            }

    def import_data(self, data: Dict) -> Dict:
        """데이터 가져오기"""
        stats = {
            "chunks_imported": 0,
            "cards_imported": 0,
            "categories_imported": 0,
            "errors": []
        }

        # 카테고리 가져오기
        for cat in data.get("categories", []):
            try:
                self.save_category(cat)
                stats["categories_imported"] += 1
            except Exception as e:
                stats["errors"].append(f"Category {cat.get('name')}: {e}")

        # 지식 청크 가져오기
        for chunk in data.get("knowledge_chunks", []):
            try:
                self.save_knowledge_chunk(chunk)
                stats["chunks_imported"] += 1
            except Exception as e:
                stats["errors"].append(f"Chunk {chunk.get('title')}: {e}")

        # 카드 가져오기
        for card in data.get("cards", []):
            try:
                self.save_card(card)
                stats["cards_imported"] += 1
            except Exception as e:
                stats["errors"].append(f"Card {card.get('card_id')}: {e}")

        return stats


# 사용 예시
if __name__ == "__main__":
    db = Database("data/test_learning.db")

    # 테스트 카드 저장
    test_card = {
        "card_id": "test_001",
        "content": "Python GIL이란?",
        "answer": "Global Interpreter Lock - 한 번에 하나의 스레드만 실행",
        "tags": ["python", "concurrency"],
        "priority": 8,
        "energy_required": "high",
        "due": datetime.now(),
        "stability": 2.5,
        "difficulty": 5.0,
        "elapsed_days": 0,
        "scheduled_days": 1,
        "reps": 0,
        "lapses": 0,
        "state": 0,
        "last_review": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "source": "Python 문서",
        "parent_topic": "Python 고급"
    }

    db.save_card(test_card)
    print("카드 저장 완료")

    # 카드 조회
    card = db.get_card("test_001")
    print(f"조회된 카드: {card['content']}")

    # 통계
    stats = db.get_statistics("user_001")
    print(f"총 카드: {stats['total_cards']}")
