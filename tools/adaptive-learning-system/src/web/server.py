"""
Flask 기반 웹 대시보드 서버

기능:
- 인터랙티브 대시보드
- 실시간 통계
- REST API
- 히트맵 캘린더
- LLM 기반 스마트 지식 분해
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import sys
import os

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 상위 디렉토리 모듈 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.database import Database
from core.fsrs import FSRS, Card, Rating, State
from gamification.engine import GamificationEngine
from adaptive.scheduler import AdaptiveScheduler

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
CORS(app)

# 전역 인스턴스
db = Database()
fsrs = FSRS()
gamification = GamificationEngine()
scheduler = AdaptiveScheduler()

DEFAULT_USER = "default_user"


# ===== 페이지 라우트 =====

@app.route('/')
def index():
    """메인 대시보드"""
    return render_template('dashboard.html')


@app.route('/review')
def review_page():
    """복습 페이지"""
    return render_template('review.html')


@app.route('/knowledge')
def knowledge_page():
    """지식 관리 페이지"""
    return render_template('knowledge.html')


@app.route('/schedule')
def schedule_page():
    """스케줄 페이지"""
    return render_template('schedule.html')


@app.route('/library')
def library_page():
    """지식 라이브러리 (전체 개요)"""
    return render_template('library.html')


# ===== API 엔드포인트 =====

@app.route('/api/dashboard')
def api_dashboard():
    """대시보드 데이터"""
    user_id = request.args.get('user_id', DEFAULT_USER)

    # 게이미피케이션 데이터
    dashboard = gamification.get_dashboard_data(user_id)

    # 통계 데이터
    stats = db.get_statistics(user_id)

    # 일일 기록 (30일)
    daily_records = db.get_daily_records(user_id, days=365)

    return jsonify({
        "success": True,
        "data": {
            "gamification": dashboard,
            "statistics": stats,
            "daily_records": daily_records,
            "timestamp": datetime.now().isoformat()
        }
    })


@app.route('/api/heatmap')
def api_heatmap():
    """히트맵 캘린더 데이터"""
    user_id = request.args.get('user_id', DEFAULT_USER)
    days = int(request.args.get('days', 365))

    records = db.get_daily_records(user_id, days=days)

    # 히트맵 형식으로 변환
    heatmap_data = {}
    for record in records:
        date = record['date']
        intensity = min(record.get('cards_reviewed', 0) / 50, 1.0)  # 50장 = 최대
        heatmap_data[date] = {
            "count": record.get('cards_reviewed', 0),
            "intensity": intensity,
            "xp": record.get('xp_earned', 0),
            "minutes": record.get('study_minutes', 0)
        }

    return jsonify({
        "success": True,
        "data": heatmap_data
    })


@app.route('/api/reviews/by-date')
def api_reviews_by_date():
    """특정 날짜의 복습 기록 조회"""
    date = request.args.get('date')  # YYYY-MM-DD 형식
    if not date:
        return jsonify({"success": False, "error": "date parameter required"}), 400

    # 해당 날짜의 복습 기록 조회
    reviews = db.get_reviews_by_date(date)

    return jsonify({
        "success": True,
        "data": {
            "date": date,
            "reviews": reviews,
            "count": len(reviews)
        }
    })


@app.route('/api/cards/due')
def api_cards_due():
    """복습할 카드 목록"""
    limit = int(request.args.get('limit', 20))
    energy = request.args.get('energy')

    cards = db.get_due_cards(limit=limit, energy_level=energy)

    return jsonify({
        "success": True,
        "data": {
            "cards": cards,
            "count": len(cards)
        }
    })


@app.route('/api/cards/new')
def api_cards_new():
    """새 카드 목록"""
    limit = int(request.args.get('limit', 20))
    topic = request.args.get('topic')

    cards = db.get_new_cards(limit=limit, topic=topic)

    return jsonify({
        "success": True,
        "data": {
            "cards": cards,
            "count": len(cards)
        }
    })


@app.route('/api/review', methods=['POST'])
def api_review():
    """카드 복습 처리"""
    data = request.json
    card_id = data.get('card_id')
    rating = int(data.get('rating', 3))
    user_id = data.get('user_id', DEFAULT_USER)

    # 카드 조회
    card_data = db.get_card(card_id)
    if not card_data:
        return jsonify({"success": False, "error": "Card not found"}), 404

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

    # FSRS 처리
    now = datetime.now()
    rating_enum = Rating(rating)
    updated_card = fsrs.repeat(card, now, rating_enum)

    # DB 저장
    db.save_card({
        "card_id": updated_card.card_id,
        "content": updated_card.content,
        "answer": updated_card.answer,
        "tags": updated_card.tags,
        "priority": updated_card.priority,
        "energy_required": updated_card.energy_required,
        "due": updated_card.due,
        "stability": updated_card.stability,
        "difficulty": updated_card.difficulty,
        "elapsed_days": updated_card.elapsed_days,
        "scheduled_days": updated_card.scheduled_days,
        "reps": updated_card.reps,
        "lapses": updated_card.lapses,
        "state": updated_card.state.value,
        "last_review": updated_card.last_review,
        "created_at": card_data.get("created_at"),
        "updated_at": datetime.now(),
        "source": card_data.get("source", ""),
        "parent_topic": card_data.get("parent_topic")
    })

    # 복습 기록
    db.save_review_log(
        card_id=card_id,
        rating=rating,
        scheduled_days=updated_card.scheduled_days,
        elapsed_days=updated_card.elapsed_days,
        state=updated_card.state.value
    )

    # 게이미피케이션
    is_correct = rating >= 3
    xp_result = gamification.record_review(user_id, correct=is_correct, card_difficulty=int(updated_card.difficulty))

    # 일일 기록 업데이트 (히트맵용)
    today = datetime.now()
    existing_records = db.get_daily_records(user_id, days=1)
    today_str = today.strftime("%Y-%m-%d")

    if existing_records and existing_records[0].get('date') == today_str:
        # 기존 기록 업데이트
        current_record = existing_records[0]
        db.save_daily_record(user_id, today, {
            "xp_earned": current_record.get('xp_earned', 0) + xp_result["final_xp"],
            "cards_reviewed": current_record.get('cards_reviewed', 0) + 1,
            "cards_new": current_record.get('cards_new', 0),
            "study_minutes": current_record.get('study_minutes', 0),
            "perfect_reviews": current_record.get('perfect_reviews', 0) + (1 if rating == 4 else 0)
        })
    else:
        # 새 기록 생성
        db.save_daily_record(user_id, today, {
            "xp_earned": xp_result["final_xp"],
            "cards_reviewed": 1,
            "cards_new": 0,
            "study_minutes": 0,
            "perfect_reviews": 1 if rating == 4 else 0
        })

    return jsonify({
        "success": True,
        "data": {
            "next_due": updated_card.due.isoformat(),
            "interval_days": updated_card.scheduled_days,
            "stability": updated_card.stability,
            "xp_earned": xp_result["final_xp"],
            "is_correct": is_correct,
            "level_up": xp_result.get("level_up", False)
        }
    })


@app.route('/api/knowledge/smart', methods=['POST'])
def api_smart_knowledge():
    """LLM 기반 스마트 지식 추가 (자동 분해)"""
    data = request.json

    from core.knowledge import SmartKnowledgeProcessor, KnowledgeType

    processor = SmartKnowledgeProcessor(llm_provider="auto")

    if not processor.llm.is_available():
        return jsonify({
            "success": False,
            "error": "LLM not configured. Set GOOGLE_API_KEY or OPENAI_API_KEY environment variable.",
            "hint": "pip install google-generativeai 또는 pip install openai"
        }), 400

    text = data.get('content', '')
    topic = data.get('topic', data.get('title', ''))
    source = data.get('source', '')

    # LLM으로 텍스트 분해
    chunks = processor.process_large_text(
        text=text,
        topic=topic,
        source=source,
        auto_link=True
    )

    # DB에 저장하고 플래시카드 생성
    results = []
    for chunk in chunks:
        # 청크 저장
        db.save_knowledge_chunk({
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

        # 스마트 플래시카드 생성
        cards = processor.create_smart_flashcards(chunk)
        for card in cards:
            from core.fsrs import Card
            new_card = Card(
                card_id=card["card_id"],
                content=card["content"],
                answer=card["answer"],
                tags=card.get("tags", []),
                priority=chunk.priority.value,
                energy_required=chunk.energy_required
            )
            db.save_card({
                "card_id": new_card.card_id,
                "content": new_card.content,
                "answer": new_card.answer,
                "tags": new_card.tags,
                "priority": new_card.priority,
                "energy_required": new_card.energy_required,
                "due": new_card.due,
                "stability": new_card.stability,
                "difficulty": new_card.difficulty,
                "elapsed_days": new_card.elapsed_days,
                "scheduled_days": new_card.scheduled_days,
                "reps": new_card.reps,
                "lapses": new_card.lapses,
                "state": new_card.state.value,
                "last_review": new_card.last_review,
                "created_at": new_card.created_at,
                "updated_at": datetime.now(),
                "source": source,
                "parent_topic": topic
            })

        results.append({
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "type": chunk.knowledge_type.value,
            "cards_created": len(cards)
        })

    return jsonify({
        "success": True,
        "data": {
            "chunks_created": len(results),
            "chunks": results,
            "llm_provider": processor.llm.provider
        }
    })


@app.route('/api/knowledge', methods=['POST'])
def api_add_knowledge():
    """새 지식 추가 (수동)"""
    data = request.json

    from core.knowledge import KnowledgeProcessor, KnowledgeType

    processor = KnowledgeProcessor()

    ktype = KnowledgeType[data.get('type', 'CONCEPT').upper()]

    chunk = processor.create_chunk(
        title=data['title'],
        content=data['content'],
        knowledge_type=ktype,
        source=data.get('source', ''),
        tags=data.get('tags', [])
    )

    # DB 저장
    db.save_knowledge_chunk({
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

    # 플래시카드 생성
    card = Card(
        card_id=f"card_{chunk.chunk_id}_main",
        content=chunk.title,
        answer=chunk.content,
        tags=chunk.tags,
        priority=chunk.priority.value,
        energy_required=chunk.energy_required
    )

    db.save_card({
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
        "source": chunk.source,
        "parent_topic": chunk.parent_topic
    })

    return jsonify({
        "success": True,
        "data": {
            "chunk_id": chunk.chunk_id,
            "card_id": card.card_id,
            "priority": chunk.priority.name,
            "difficulty": chunk.difficulty
        }
    })


@app.route('/api/streak')
def api_streak():
    """스트릭 업데이트"""
    user_id = request.args.get('user_id', DEFAULT_USER)

    result = gamification.update_streak(user_id)

    return jsonify({
        "success": True,
        "data": result
    })


@app.route('/api/quests')
def api_quests():
    """일일 퀘스트"""
    user_id = request.args.get('user_id', DEFAULT_USER)

    quests = gamification.generate_daily_quests(user_id)

    return jsonify({
        "success": True,
        "data": [{
            "id": q.quest_id,
            "name": q.name,
            "description": q.description,
            "target": q.target,
            "current": q.current,
            "xp_reward": q.xp_reward,
            "completed": q.completed
        } for q in quests]
    })


@app.route('/api/schedule', methods=['POST'])
def api_create_schedule():
    """일일 스케줄 생성"""
    data = request.json

    from adhd.scheduler import EnergyLevel

    available_hours = data.get('available_hours', [(9, 12), (14, 17)])
    energy_pattern = data.get('energy_pattern')

    pattern = None
    if energy_pattern:
        pattern = {k: EnergyLevel[v.upper()] for k, v in energy_pattern.items()}

    stats = db.get_statistics(DEFAULT_USER)

    schedule = scheduler.create_daily_schedule(
        date=datetime.now(),
        available_hours=available_hours,
        cards_due=stats.get("due_today", 0),
        new_cards=min(stats.get("new_cards", 0), 20),
        energy_pattern=pattern
    )

    blocks = []
    for block in schedule.blocks:
        blocks.append({
            "start": block.start.isoformat(),
            "end": block.end.isoformat(),
            "duration": block.duration_minutes,
            "task_type": block.task_type.value,
            "energy_level": block.energy_level.value,
            "is_break": block.is_break
        })

    return jsonify({
        "success": True,
        "data": {
            "blocks": blocks,
            "total_study_minutes": schedule.total_study_minutes
        }
    })


# ===== 지식 관리 API =====

@app.route('/api/knowledge/overview')
def api_knowledge_overview():
    """지식 개요 통계"""
    overview = db.get_knowledge_overview()
    return jsonify({
        "success": True,
        "data": overview
    })


@app.route('/api/knowledge/list')
def api_knowledge_list():
    """지식 목록 조회"""
    knowledge_type = request.args.get('type')
    parent_topic = request.args.get('topic')
    limit = int(request.args.get('limit', 100))

    chunks = db.get_all_knowledge_chunks(
        knowledge_type=knowledge_type,
        parent_topic=parent_topic,
        limit=limit
    )

    return jsonify({
        "success": True,
        "data": {
            "chunks": chunks,
            "count": len(chunks)
        }
    })


@app.route('/api/knowledge/<chunk_id>')
def api_get_knowledge(chunk_id):
    """개별 지식 조회"""
    chunk = db.get_knowledge_chunk(chunk_id)
    if not chunk:
        return jsonify({"success": False, "error": "Not found"}), 404

    return jsonify({
        "success": True,
        "data": chunk
    })


@app.route('/api/knowledge/<chunk_id>', methods=['PUT'])
def api_update_knowledge(chunk_id):
    """지식 수정"""
    data = request.json

    success = db.update_knowledge_chunk(chunk_id, data)

    if not success:
        return jsonify({"success": False, "error": "Update failed"}), 400

    # 관련 카드도 업데이트
    if 'title' in data or 'content' in data:
        card_id = f"card_{chunk_id}_main"
        card = db.get_card(card_id)
        if card:
            card['content'] = data.get('title', card['content'])
            card['answer'] = data.get('content', card['answer'])
            if 'tags' in data:
                card['tags'] = data['tags']
            db.save_card(card)

    return jsonify({
        "success": True,
        "message": "Updated successfully"
    })


@app.route('/api/knowledge/<chunk_id>', methods=['DELETE'])
def api_delete_knowledge(chunk_id):
    """지식 삭제"""
    success = db.delete_knowledge_chunk(chunk_id)

    if not success:
        return jsonify({"success": False, "error": "Delete failed"}), 400

    return jsonify({
        "success": True,
        "message": "Deleted successfully"
    })


# ===== 카테고리 API =====

@app.route('/api/categories')
def api_get_categories():
    """카테고리 목록"""
    categories = db.get_all_categories()
    return jsonify({
        "success": True,
        "data": categories
    })


@app.route('/api/categories', methods=['POST'])
def api_create_category():
    """카테고리 생성"""
    data = request.json

    import hashlib
    category_id = hashlib.md5(data['name'].encode()).hexdigest()[:12]

    category = {
        "category_id": category_id,
        "name": data['name'],
        "description": data.get('description', ''),
        "icon": data.get('icon', '📁'),
        "color": data.get('color', '#6366f1'),
        "parent_id": data.get('parent_id'),
        "sort_order": data.get('sort_order', 0)
    }

    db.save_category(category)

    return jsonify({
        "success": True,
        "data": category
    })


@app.route('/api/categories/<category_id>', methods=['PUT'])
def api_update_category(category_id):
    """카테고리 수정"""
    data = request.json

    existing = db.get_category(category_id)
    if not existing:
        return jsonify({"success": False, "error": "Not found"}), 404

    for key, value in data.items():
        existing[key] = value

    db.save_category(existing)

    return jsonify({
        "success": True,
        "data": existing
    })


@app.route('/api/categories/<category_id>', methods=['DELETE'])
def api_delete_category(category_id):
    """카테고리 삭제"""
    success = db.delete_category(category_id)

    if not success:
        return jsonify({"success": False, "error": "Delete failed"}), 400

    return jsonify({
        "success": True,
        "message": "Deleted successfully"
    })


# ===== Export / Import API =====

@app.route('/api/export')
def api_export():
    """전체 데이터 내보내기"""
    data = db.export_all_data()

    return jsonify({
        "success": True,
        "data": data
    })


@app.route('/api/import', methods=['POST'])
def api_import():
    """데이터 가져오기"""
    data = request.json

    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    stats = db.import_data(data)

    return jsonify({
        "success": True,
        "data": stats
    })


# ===== 카드 관리 API =====

@app.route('/api/cards')
def api_get_cards():
    """모든 카드 조회"""
    limit = int(request.args.get('limit', 100))
    cards = db.get_all_cards(limit=limit)

    return jsonify({
        "success": True,
        "data": {
            "cards": cards,
            "count": len(cards)
        }
    })


@app.route('/api/cards/<card_id>', methods=['DELETE'])
def api_delete_card(card_id):
    """카드 삭제"""
    success = db.delete_card(card_id)

    if not success:
        return jsonify({"success": False, "error": "Delete failed"}), 400

    return jsonify({
        "success": True,
        "message": "Deleted successfully"
    })


# ===== 에러 핸들러 =====

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    print("🌐 Adaptive Learning System - Web Dashboard")
    print("   http://localhost:5000")
    app.run(debug=True, port=5000)
