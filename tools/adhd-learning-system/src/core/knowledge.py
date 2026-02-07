"""
지식 처리 및 구조화 모듈

핵심 기능:
1. 지식 청킹 (Atomic Notes)
2. 자동 태깅 및 연결
3. 우선순위 자동 산정
4. 정교화 질문 생성
5. LLM 기반 자동 분해 (Gemini/OpenAI)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import re
import hashlib
import json
import os


class KnowledgeType(Enum):
    """지식 유형"""
    CONCEPT = "concept"          # 개념/정의
    PROCEDURE = "procedure"      # 절차/방법
    FACT = "fact"               # 사실/데이터
    PRINCIPLE = "principle"      # 원리/규칙
    EXAMPLE = "example"         # 예시
    ANALOGY = "analogy"         # 비유/유사성
    QUESTION = "question"       # 질문/문제


class Priority(Enum):
    """우선순위"""
    CRITICAL = 1    # 핵심, 기초가 되는 지식
    HIGH = 2        # 중요, 자주 사용
    MEDIUM = 3      # 보통
    LOW = 4         # 참조용
    OPTIONAL = 5    # 선택적


@dataclass
class KnowledgeChunk:
    """원자적 지식 단위 (Atomic Note)"""
    chunk_id: str
    title: str
    content: str
    knowledge_type: KnowledgeType

    # 메타데이터
    source: str = ""                    # 출처
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 구조화
    tags: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)  # 연결된 청크 ID들
    parent_topic: Optional[str] = None
    children: List[str] = field(default_factory=list)

    # 학습 관련
    priority: Priority = Priority.MEDIUM
    difficulty: int = 5                 # 1-10
    energy_required: str = "medium"     # low, medium, high
    estimated_minutes: int = 5          # 예상 학습 시간

    # 정교화 질문
    why_questions: List[str] = field(default_factory=list)
    how_questions: List[str] = field(default_factory=list)
    what_if_questions: List[str] = field(default_factory=list)

    # 연결
    similar_chunks: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

    # 예시
    examples: List[str] = field(default_factory=list)
    counter_examples: List[str] = field(default_factory=list)

    def to_flashcard_format(self) -> Dict[str, str]:
        """플래시카드 형식으로 변환"""
        return {
            "front": self.title,
            "back": self.content,
            "tags": self.tags,
            "extra": {
                "why": self.why_questions,
                "examples": self.examples
            }
        }


class KnowledgeProcessor:
    """지식 처리기"""

    def __init__(self):
        self.chunks: Dict[str, KnowledgeChunk] = {}
        self.tag_index: Dict[str, List[str]] = {}  # tag -> chunk_ids
        self.topic_tree: Dict[str, List[str]] = {}  # topic -> chunk_ids

    def create_chunk(
        self,
        title: str,
        content: str,
        knowledge_type: KnowledgeType,
        source: str = "",
        tags: List[str] = None,
        parent_topic: str = None
    ) -> KnowledgeChunk:
        """새 지식 청크 생성"""

        # ID 생성
        chunk_id = self._generate_id(title, content)

        chunk = KnowledgeChunk(
            chunk_id=chunk_id,
            title=title,
            content=content,
            knowledge_type=knowledge_type,
            source=source,
            tags=tags or [],
            parent_topic=parent_topic
        )

        # 자동 처리
        chunk = self._auto_tag(chunk)
        chunk = self._calculate_priority(chunk)
        chunk = self._generate_elaboration_questions(chunk)
        chunk = self._estimate_difficulty(chunk)

        # 저장 및 인덱싱
        self.chunks[chunk_id] = chunk
        self._update_indices(chunk)

        return chunk

    def _generate_id(self, title: str, content: str) -> str:
        """고유 ID 생성"""
        combined = f"{title}:{content[:100]}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def _auto_tag(self, chunk: KnowledgeChunk) -> KnowledgeChunk:
        """자동 태깅"""
        text = f"{chunk.title} {chunk.content}".lower()

        # 키워드 기반 태그 추출
        keyword_tags = {
            "python": ["python", "파이썬", "py"],
            "javascript": ["javascript", "js", "자바스크립트"],
            "database": ["database", "db", "sql", "데이터베이스"],
            "algorithm": ["algorithm", "알고리즘", "big-o", "시간복잡도"],
            "design-pattern": ["pattern", "패턴", "singleton", "factory"],
            "web": ["http", "api", "rest", "web", "웹"],
            "ai-ml": ["machine learning", "머신러닝", "ai", "딥러닝", "neural"],
        }

        for tag, keywords in keyword_tags.items():
            if any(kw in text for kw in keywords):
                if tag not in chunk.tags:
                    chunk.tags.append(tag)

        return chunk

    def _calculate_priority(self, chunk: KnowledgeChunk) -> KnowledgeChunk:
        """우선순위 자동 계산"""
        score = 0

        # 지식 유형 기반 점수
        type_scores = {
            KnowledgeType.CONCEPT: 3,
            KnowledgeType.PRINCIPLE: 3,
            KnowledgeType.PROCEDURE: 2,
            KnowledgeType.FACT: 1,
            KnowledgeType.EXAMPLE: 1,
            KnowledgeType.ANALOGY: 2,
            KnowledgeType.QUESTION: 2,
        }
        score += type_scores.get(chunk.knowledge_type, 1)

        # 연결 수 기반 (많이 연결된 지식 = 중요)
        score += min(len(chunk.links), 3)

        # 우선순위 결정
        if score >= 5:
            chunk.priority = Priority.CRITICAL
        elif score >= 4:
            chunk.priority = Priority.HIGH
        elif score >= 2:
            chunk.priority = Priority.MEDIUM
        else:
            chunk.priority = Priority.LOW

        return chunk

    def _estimate_difficulty(self, chunk: KnowledgeChunk) -> KnowledgeChunk:
        """난이도 추정"""
        text = chunk.content.lower()

        # 복잡도 지표
        difficulty = 5  # 기본값

        # 길이 기반
        if len(chunk.content) > 500:
            difficulty += 1
        if len(chunk.content) > 1000:
            difficulty += 1

        # 전문 용어 밀도
        technical_terms = ["algorithm", "complexity", "paradigm", "abstraction",
                         "polymorphism", "recursion", "optimization"]
        term_count = sum(1 for term in technical_terms if term in text)
        difficulty += min(term_count, 2)

        # 필수 지식 수
        difficulty += min(len(chunk.prerequisites), 2)

        chunk.difficulty = min(max(difficulty, 1), 10)

        # 에너지 레벨 결정
        if chunk.difficulty >= 7:
            chunk.energy_required = "high"
        elif chunk.difficulty >= 4:
            chunk.energy_required = "medium"
        else:
            chunk.energy_required = "low"

        return chunk

    def _generate_elaboration_questions(self, chunk: KnowledgeChunk) -> KnowledgeChunk:
        """정교화 질문 생성 (Elaborative Interrogation)"""

        title = chunk.title

        # WHY 질문
        chunk.why_questions = [
            f"왜 {title}이/가 중요한가?",
            f"왜 {title}이/가 이런 방식으로 작동하는가?",
            f"{title}의 근본적인 이유는 무엇인가?",
        ]

        # HOW 질문
        chunk.how_questions = [
            f"{title}은/는 어떻게 작동하는가?",
            f"{title}을/를 실제로 어떻게 적용하는가?",
            f"{title}과/와 다른 개념은 어떻게 연결되는가?",
        ]

        # WHAT IF 질문
        chunk.what_if_questions = [
            f"{title}이/가 없다면 어떻게 될까?",
            f"{title}을/를 다르게 구현한다면?",
            f"{title}의 반대 경우는 무엇일까?",
        ]

        return chunk

    def _update_indices(self, chunk: KnowledgeChunk):
        """인덱스 업데이트"""
        # 태그 인덱스
        for tag in chunk.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(chunk.chunk_id)

        # 토픽 트리
        if chunk.parent_topic:
            if chunk.parent_topic not in self.topic_tree:
                self.topic_tree[chunk.parent_topic] = []
            self.topic_tree[chunk.parent_topic].append(chunk.chunk_id)

    def find_similar(self, chunk_id: str, limit: int = 5) -> List[KnowledgeChunk]:
        """유사한 청크 찾기"""
        if chunk_id not in self.chunks:
            return []

        target = self.chunks[chunk_id]
        scores = []

        for cid, chunk in self.chunks.items():
            if cid == chunk_id:
                continue

            score = 0
            # 태그 유사도
            common_tags = set(target.tags) & set(chunk.tags)
            score += len(common_tags) * 2

            # 같은 토픽
            if target.parent_topic == chunk.parent_topic:
                score += 3

            # 같은 유형
            if target.knowledge_type == chunk.knowledge_type:
                score += 1

            if score > 0:
                scores.append((score, chunk))

        # 점수순 정렬
        scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scores[:limit]]

    def link_chunks(self, chunk_id1: str, chunk_id2: str):
        """두 청크 연결"""
        if chunk_id1 in self.chunks and chunk_id2 in self.chunks:
            if chunk_id2 not in self.chunks[chunk_id1].links:
                self.chunks[chunk_id1].links.append(chunk_id2)
            if chunk_id1 not in self.chunks[chunk_id2].links:
                self.chunks[chunk_id2].links.append(chunk_id1)

    def get_learning_path(self, target_chunk_id: str) -> List[KnowledgeChunk]:
        """학습 경로 생성 (선수 지식 포함)"""
        if target_chunk_id not in self.chunks:
            return []

        path = []
        visited = set()

        def collect_prerequisites(chunk_id: str):
            if chunk_id in visited:
                return
            visited.add(chunk_id)

            chunk = self.chunks.get(chunk_id)
            if not chunk:
                return

            for prereq_id in chunk.prerequisites:
                collect_prerequisites(prereq_id)

            path.append(chunk)

        collect_prerequisites(target_chunk_id)
        return path

    def export_to_obsidian(self, chunk: KnowledgeChunk) -> str:
        """Obsidian 마크다운 형식으로 내보내기"""
        lines = [
            f"# {chunk.title}",
            "",
            f"**유형**: {chunk.knowledge_type.value}",
            f"**우선순위**: {chunk.priority.name}",
            f"**난이도**: {chunk.difficulty}/10",
            f"**에너지**: {chunk.energy_required}",
            "",
            "## 내용",
            chunk.content,
            "",
        ]

        if chunk.examples:
            lines.extend(["## 예시", ""])
            for ex in chunk.examples:
                lines.append(f"- {ex}")
            lines.append("")

        if chunk.why_questions:
            lines.extend(["## 정교화 질문", "", "### Why?"])
            for q in chunk.why_questions:
                lines.append(f"- {q}")

        if chunk.how_questions:
            lines.extend(["", "### How?"])
            for q in chunk.how_questions:
                lines.append(f"- {q}")

        # 태그
        if chunk.tags:
            lines.extend(["", "---", ""])
            lines.append("Tags: " + " ".join(f"#{tag}" for tag in chunk.tags))

        # 연결
        if chunk.links:
            lines.extend(["", "## 관련 노트"])
            for link_id in chunk.links:
                if link_id in self.chunks:
                    linked = self.chunks[link_id]
                    lines.append(f"- [[{linked.title}]]")

        return "\n".join(lines)


class LLMChunker:
    """
    LLM 기반 지식 자동 분해기

    지원 모델:
    - Google Gemini (GOOGLE_API_KEY)
    - OpenAI GPT (OPENAI_API_KEY)
    """

    CHUNKING_PROMPT = """당신은 지식 관리 전문가입니다. 주어진 텍스트를 원자적 지식 단위(Atomic Notes)로 분해해주세요.

## 원자적 지식 단위란?
- 하나의 개념/사실/절차만 포함
- 독립적으로 이해 가능
- 다른 지식과 연결 가능
- 플래시카드로 변환 가능한 크기

## 각 청크에 포함할 정보:
1. title: 간결한 제목 (한 줄)
2. content: 핵심 내용 (2-5문장)
3. type: concept(개념), procedure(절차), fact(사실), principle(원리), example(예시)
4. tags: 관련 태그 배열
5. prerequisites: 이 지식을 이해하기 위해 먼저 알아야 할 개념들
6. examples: 구체적인 예시 (있다면)

## 출력 형식 (JSON 배열):
```json
[
  {
    "title": "제목",
    "content": "내용",
    "type": "concept",
    "tags": ["tag1", "tag2"],
    "prerequisites": ["선수지식1"],
    "examples": ["예시1"]
  }
]
```

## 입력 텍스트:
{text}

## 주제 컨텍스트: {topic}

위 텍스트를 원자적 지식 단위로 분해하여 JSON 배열로만 출력하세요. 다른 설명 없이 JSON만 출력하세요."""

    FLASHCARD_PROMPT = """당신은 효과적인 플래시카드 작성 전문가입니다. 주어진 지식을 복습에 최적화된 플래시카드로 변환해주세요.

## 좋은 플래시카드의 특징:
- 질문이 명확하고 구체적
- 답변이 간결 (1-3문장)
- 능동적 회상을 유도
- 모호함 없음

## 카드 유형:
1. 기본형: "X란 무엇인가?" → 정의
2. 비교형: "A와 B의 차이는?" → 차이점
3. 적용형: "X를 언제 사용하는가?" → 사용 시나리오
4. Why형: "왜 X가 필요한가?" → 이유/목적

## 입력:
제목: {title}
내용: {content}

## 출력 형식 (JSON 배열):
```json
[
  {
    "question": "질문",
    "answer": "답변",
    "card_type": "basic|compare|apply|why"
  }
]
```

JSON 배열만 출력하세요."""

    def __init__(self, provider: str = "auto"):
        """
        Args:
            provider: "gemini", "openai", or "auto" (환경변수 기반 자동 선택)
        """
        self.provider = provider
        self._client = None
        self._model = None
        self._init_client()

    def _init_client(self):
        """API 클라이언트 초기화"""
        gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if self.provider == "auto":
            if gemini_key:
                self.provider = "gemini"
            elif openai_key:
                self.provider = "openai"
            else:
                self.provider = None
                return

        if self.provider == "gemini" and gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self._client = genai
                self._model = genai.GenerativeModel('gemini-1.5-flash')
            except ImportError:
                print("google-generativeai 패키지가 필요합니다: pip install google-generativeai")
                self.provider = None

        elif self.provider == "openai" and openai_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=openai_key)
                self._model = "gpt-4o-mini"
            except ImportError:
                print("openai 패키지가 필요합니다: pip install openai")
                self.provider = None

    def is_available(self) -> bool:
        """LLM 사용 가능 여부"""
        return self.provider is not None and self._client is not None

    def chunk_text(self, text: str, topic: str = "") -> List[Dict]:
        """
        텍스트를 원자적 지식 단위로 분해

        Args:
            text: 분해할 텍스트
            topic: 주제 컨텍스트 (선택)

        Returns:
            분해된 청크 리스트
        """
        if not self.is_available():
            return []

        prompt = self.CHUNKING_PROMPT.format(text=text, topic=topic or "일반")

        try:
            response_text = self._call_llm(prompt)
            chunks = self._parse_json_response(response_text)
            return chunks
        except Exception as e:
            print(f"LLM chunking error: {e}")
            return []

    def generate_flashcards(self, title: str, content: str) -> List[Dict]:
        """
        지식을 플래시카드로 변환

        Args:
            title: 지식 제목
            content: 지식 내용

        Returns:
            플래시카드 리스트
        """
        if not self.is_available():
            return []

        prompt = self.FLASHCARD_PROMPT.format(title=title, content=content)

        try:
            response_text = self._call_llm(prompt)
            cards = self._parse_json_response(response_text)
            return cards
        except Exception as e:
            print(f"LLM flashcard generation error: {e}")
            return []

    def _call_llm(self, prompt: str) -> str:
        """LLM API 호출"""
        if self.provider == "gemini":
            response = self._model.generate_content(prompt)
            return response.text

        elif self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content

        return ""

    def _parse_json_response(self, text: str) -> List[Dict]:
        """JSON 응답 파싱"""
        # JSON 블록 추출
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSON 배열 직접 찾기
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                json_str = json_match.group(0)
            else:
                return []

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return []


class SmartKnowledgeProcessor(KnowledgeProcessor):
    """
    LLM 기반 지능형 지식 처리기

    KnowledgeProcessor를 확장하여 LLM 기반 자동 분해 기능 추가
    """

    def __init__(self, llm_provider: str = "auto"):
        super().__init__()
        self.llm = LLMChunker(provider=llm_provider)

    def process_large_text(
        self,
        text: str,
        topic: str = "",
        source: str = "",
        auto_link: bool = True
    ) -> List[KnowledgeChunk]:
        """
        큰 텍스트를 자동으로 원자적 단위로 분해하고 연결

        Args:
            text: 분해할 텍스트
            topic: 주제
            source: 출처
            auto_link: 자동 연결 여부

        Returns:
            생성된 KnowledgeChunk 리스트
        """
        if not self.llm.is_available():
            # LLM 사용 불가시 단일 청크로 생성
            print("LLM not available, creating single chunk")
            chunk = self.create_chunk(
                title=topic or "Untitled",
                content=text[:2000],  # 최대 길이 제한
                knowledge_type=KnowledgeType.CONCEPT,
                source=source,
                parent_topic=topic
            )
            return [chunk]

        # LLM으로 분해
        raw_chunks = self.llm.chunk_text(text, topic)

        if not raw_chunks:
            # 분해 실패시 단일 청크
            chunk = self.create_chunk(
                title=topic or "Untitled",
                content=text[:2000],
                knowledge_type=KnowledgeType.CONCEPT,
                source=source,
                parent_topic=topic
            )
            return [chunk]

        # 청크 생성
        created_chunks = []
        for raw in raw_chunks:
            try:
                ktype = self._map_type(raw.get("type", "concept"))

                chunk = self.create_chunk(
                    title=raw.get("title", "Untitled"),
                    content=raw.get("content", ""),
                    knowledge_type=ktype,
                    source=source,
                    tags=raw.get("tags", []),
                    parent_topic=topic
                )

                # 예시 추가
                if raw.get("examples"):
                    chunk.examples = raw["examples"]

                created_chunks.append(chunk)
            except Exception as e:
                print(f"Error creating chunk: {e}")
                continue

        # 자동 연결
        if auto_link and len(created_chunks) > 1:
            self._auto_link_chunks(created_chunks)

        return created_chunks

    def create_smart_flashcards(self, chunk: KnowledgeChunk) -> List[Dict]:
        """
        지식 청크에서 스마트 플래시카드 생성

        Args:
            chunk: 지식 청크

        Returns:
            플래시카드 리스트 [{card_id, content, answer, tags}, ...]
        """
        if not self.llm.is_available():
            # 기본 플래시카드만 생성
            return [{
                "card_id": f"card_{chunk.chunk_id}_main",
                "content": chunk.title,
                "answer": chunk.content,
                "tags": chunk.tags
            }]

        # LLM으로 다양한 플래시카드 생성
        raw_cards = self.llm.generate_flashcards(chunk.title, chunk.content)

        cards = []
        for i, raw in enumerate(raw_cards):
            cards.append({
                "card_id": f"card_{chunk.chunk_id}_{i}",
                "content": raw.get("question", chunk.title),
                "answer": raw.get("answer", chunk.content),
                "tags": chunk.tags,
                "card_type": raw.get("card_type", "basic")
            })

        # 최소 1개 카드 보장
        if not cards:
            cards.append({
                "card_id": f"card_{chunk.chunk_id}_main",
                "content": chunk.title,
                "answer": chunk.content,
                "tags": chunk.tags
            })

        return cards

    def _map_type(self, type_str: str) -> KnowledgeType:
        """문자열을 KnowledgeType으로 변환"""
        mapping = {
            "concept": KnowledgeType.CONCEPT,
            "procedure": KnowledgeType.PROCEDURE,
            "fact": KnowledgeType.FACT,
            "principle": KnowledgeType.PRINCIPLE,
            "example": KnowledgeType.EXAMPLE,
            "analogy": KnowledgeType.ANALOGY,
            "question": KnowledgeType.QUESTION,
        }
        return mapping.get(type_str.lower(), KnowledgeType.CONCEPT)

    def _auto_link_chunks(self, chunks: List[KnowledgeChunk]):
        """청크들 자동 연결"""
        for i, chunk1 in enumerate(chunks):
            for chunk2 in chunks[i+1:]:
                # 같은 토픽이면 연결
                if chunk1.parent_topic == chunk2.parent_topic:
                    self.link_chunks(chunk1.chunk_id, chunk2.chunk_id)

                # 태그 겹치면 연결
                common_tags = set(chunk1.tags) & set(chunk2.tags)
                if len(common_tags) >= 2:
                    self.link_chunks(chunk1.chunk_id, chunk2.chunk_id)


# 사용 예시
if __name__ == "__main__":
    processor = KnowledgeProcessor()

    # 지식 청크 생성
    chunk1 = processor.create_chunk(
        title="Python GIL (Global Interpreter Lock)",
        content="""GIL은 Python 인터프리터가 한 번에 하나의 스레드만
Python 바이트코드를 실행하도록 하는 뮤텍스입니다.

이로 인해 CPU-bound 작업에서 멀티스레딩의 이점을 얻기 어렵습니다.
해결책으로 multiprocessing 모듈을 사용하거나,
I/O-bound 작업에서는 asyncio를 활용할 수 있습니다.""",
        knowledge_type=KnowledgeType.CONCEPT,
        source="Python 공식 문서",
        tags=["python", "concurrency"]
    )

    chunk2 = processor.create_chunk(
        title="Python multiprocessing",
        content="""multiprocessing 모듈은 GIL의 제약을 우회하여
실제 병렬 처리를 가능하게 합니다.

각 프로세스가 독립적인 Python 인터프리터와 메모리 공간을 가집니다.""",
        knowledge_type=KnowledgeType.PROCEDURE,
        source="Python 공식 문서",
        tags=["python", "concurrency"]
    )

    # 청크 연결
    processor.link_chunks(chunk1.chunk_id, chunk2.chunk_id)

    print("=== 생성된 지식 청크 ===")
    print(f"제목: {chunk1.title}")
    print(f"우선순위: {chunk1.priority.name}")
    print(f"난이도: {chunk1.difficulty}/10")
    print(f"에너지 필요: {chunk1.energy_required}")
    print(f"\n정교화 질문 (Why):")
    for q in chunk1.why_questions[:2]:
        print(f"  - {q}")

    print(f"\n=== Obsidian 형식 ===")
    print(processor.export_to_obsidian(chunk1)[:500] + "...")
