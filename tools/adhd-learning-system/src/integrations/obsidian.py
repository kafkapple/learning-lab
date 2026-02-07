"""
Obsidian 연동 모듈

기능:
1. 마크다운 노트 가져오기/내보내기
2. 플래시카드 자동 생성
3. 양방향 동기화
4. 백링크 지원
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class ObsidianNote:
    """Obsidian 노트"""
    path: str
    title: str
    content: str
    frontmatter: Dict
    tags: List[str]
    links: List[str]
    modified_time: datetime


class ObsidianIntegration:
    """Obsidian 연동 클래스"""

    # 플래시카드 마커
    FLASHCARD_START = "<!-- flashcard-start -->"
    FLASHCARD_END = "<!-- flashcard-end -->"

    # Cloze 패턴
    CLOZE_PATTERN = r'\{\{c(\d+)::([^}]+)\}\}'
    HIGHLIGHT_PATTERN = r'==([^=]+)=='
    BOLD_CLOZE_PATTERN = r'\*\*\{([^}]+)\}\*\*'

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)

        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

        self.notes_cache: Dict[str, ObsidianNote] = {}
        self.sync_log_path = self.vault_path / ".adhd-learning-sync.json"

    def scan_vault(self, folders: List[str] = None) -> List[ObsidianNote]:
        """볼트 스캔하여 노트 목록 반환"""
        notes = []

        search_paths = []
        if folders:
            for folder in folders:
                folder_path = self.vault_path / folder
                if folder_path.exists():
                    search_paths.append(folder_path)
        else:
            search_paths = [self.vault_path]

        for search_path in search_paths:
            for md_file in search_path.rglob("*.md"):
                # .obsidian 폴더 제외
                if ".obsidian" in str(md_file):
                    continue

                try:
                    note = self._parse_note(md_file)
                    if note:
                        notes.append(note)
                        self.notes_cache[str(md_file)] = note
                except Exception as e:
                    print(f"Error parsing {md_file}: {e}")

        return notes

    def _parse_note(self, file_path: Path) -> Optional[ObsidianNote]:
        """노트 파싱"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Frontmatter 파싱
        frontmatter = {}
        body = content

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(parts[1]) or {}
                except:
                    # YAML 파싱 실패 시 수동 파싱
                    for line in parts[1].strip().split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            frontmatter[key.strip()] = value.strip()
                body = parts[2]

        # 태그 추출
        tags = frontmatter.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]

        # 인라인 태그 추출
        inline_tags = re.findall(r'#([a-zA-Z0-9_/\-]+)', body)
        tags.extend(inline_tags)
        tags = list(set(tags))

        # 링크 추출 [[link]]
        links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', body)

        return ObsidianNote(
            path=str(file_path.relative_to(self.vault_path)),
            title=file_path.stem,
            content=body.strip(),
            frontmatter=frontmatter,
            tags=tags,
            links=links,
            modified_time=datetime.fromtimestamp(file_path.stat().st_mtime)
        )

    def extract_flashcards(self, note: ObsidianNote) -> List[Dict]:
        """노트에서 플래시카드 추출"""
        flashcards = []
        content = note.content

        # 1. 명시적 플래시카드 블록
        flashcards.extend(self._extract_marked_flashcards(content, note))

        # 2. Q&A 형식
        flashcards.extend(self._extract_qa_flashcards(content, note))

        # 3. Cloze 삭제
        flashcards.extend(self._extract_cloze_flashcards(content, note))

        # 4. 헤딩 기반
        flashcards.extend(self._extract_heading_flashcards(content, note))

        return flashcards

    def _extract_marked_flashcards(self, content: str, note: ObsidianNote) -> List[Dict]:
        """마커로 표시된 플래시카드 추출"""
        flashcards = []

        # <!-- flashcard --> 마커 찾기
        pattern = r'<!-- flashcard -->\s*\n(.+?)\n---\n(.+?)(?=\n<!-- flashcard -->|\n<!-- /flashcard -->|$)'
        matches = re.findall(pattern, content, re.DOTALL)

        for i, (question, answer) in enumerate(matches):
            flashcards.append({
                "card_id": self._generate_card_id(note.path, f"marked_{i}"),
                "content": question.strip(),
                "answer": answer.strip(),
                "source": note.path,
                "tags": note.tags,
                "type": "marked"
            })

        return flashcards

    def _extract_qa_flashcards(self, content: str, note: ObsidianNote) -> List[Dict]:
        """Q: A: 형식의 플래시카드 추출"""
        flashcards = []

        # Q: ... A: ... 패턴
        pattern = r'(?:^|\n)Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:|\n\n|$)'
        matches = re.findall(pattern, content, re.DOTALL)

        for i, (question, answer) in enumerate(matches):
            flashcards.append({
                "card_id": self._generate_card_id(note.path, f"qa_{i}"),
                "content": question.strip(),
                "answer": answer.strip(),
                "source": note.path,
                "tags": note.tags,
                "type": "qa"
            })

        return flashcards

    def _extract_cloze_flashcards(self, content: str, note: ObsidianNote) -> List[Dict]:
        """Cloze 삭제 플래시카드 추출"""
        flashcards = []

        # {{c1::answer}} 형식
        lines_with_cloze = []
        for line in content.split('\n'):
            if re.search(self.CLOZE_PATTERN, line):
                lines_with_cloze.append(line)
            # ==highlighted== 형식도 cloze로 처리
            elif re.search(self.HIGHLIGHT_PATTERN, line):
                lines_with_cloze.append(line)

        for i, line in enumerate(lines_with_cloze):
            # Cloze를 [...]로 대체하여 질문 생성
            question = re.sub(self.CLOZE_PATTERN, r'[...]', line)
            question = re.sub(self.HIGHLIGHT_PATTERN, r'[...]', question)

            # 정답 추출
            answers = re.findall(self.CLOZE_PATTERN, line)
            if not answers:
                answers = [(str(i+1), m) for m in re.findall(self.HIGHLIGHT_PATTERN, line)]

            answer = ", ".join([a[1] for a in answers])

            flashcards.append({
                "card_id": self._generate_card_id(note.path, f"cloze_{i}"),
                "content": question.strip(),
                "answer": answer,
                "source": note.path,
                "tags": note.tags,
                "type": "cloze"
            })

        return flashcards

    def _extract_heading_flashcards(self, content: str, note: ObsidianNote) -> List[Dict]:
        """헤딩 기반 플래시카드 (## 질문? 형식)"""
        flashcards = []

        # ## 헤딩이 ?로 끝나면 질문으로 처리
        pattern = r'^(#{1,3})\s+(.+\?)\s*\n((?:(?!^#{1,3}\s).+\n?)+)'
        matches = re.findall(pattern, content, re.MULTILINE)

        for i, (level, heading, body) in enumerate(matches):
            flashcards.append({
                "card_id": self._generate_card_id(note.path, f"heading_{i}"),
                "content": heading.strip(),
                "answer": body.strip(),
                "source": note.path,
                "tags": note.tags,
                "type": "heading"
            })

        return flashcards

    def _generate_card_id(self, path: str, suffix: str) -> str:
        """카드 ID 생성"""
        combined = f"{path}:{suffix}"
        return f"obs_{hashlib.md5(combined.encode()).hexdigest()[:12]}"

    def import_to_system(self, notes: List[ObsidianNote], database) -> Dict:
        """노트를 학습 시스템으로 가져오기"""
        from core.knowledge import KnowledgeProcessor, KnowledgeType
        from core.fsrs import Card

        processor = KnowledgeProcessor()
        stats = {
            "notes_processed": 0,
            "cards_created": 0,
            "chunks_created": 0,
            "errors": []
        }

        for note in notes:
            try:
                # 지식 청크 생성
                chunk = processor.create_chunk(
                    title=note.title,
                    content=note.content[:1000],  # 내용 제한
                    knowledge_type=KnowledgeType.CONCEPT,
                    source=f"Obsidian: {note.path}",
                    tags=note.tags
                )

                # DB 저장
                database.save_knowledge_chunk({
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "content": chunk.content,
                    "knowledge_type": chunk.knowledge_type.value,
                    "source": chunk.source,
                    "tags": chunk.tags,
                    "links": note.links,
                    "parent_topic": note.frontmatter.get('topic'),
                    "priority": chunk.priority.name,
                    "difficulty": chunk.difficulty,
                    "energy_required": chunk.energy_required,
                    "why_questions": chunk.why_questions,
                    "how_questions": chunk.how_questions,
                    "what_if_questions": chunk.what_if_questions,
                    "examples": chunk.examples,
                    "prerequisites": chunk.prerequisites,
                })
                stats["chunks_created"] += 1

                # 플래시카드 추출 및 저장
                flashcards = self.extract_flashcards(note)
                for fc in flashcards:
                    card = Card(
                        card_id=fc["card_id"],
                        content=fc["content"],
                        answer=fc["answer"],
                        tags=fc["tags"],
                    )

                    database.save_card({
                        "card_id": card.card_id,
                        "content": card.content,
                        "answer": card.answer,
                        "tags": card.tags,
                        "priority": 5,
                        "energy_required": "medium",
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
                        "source": fc["source"],
                        "parent_topic": None
                    })
                    stats["cards_created"] += 1

                stats["notes_processed"] += 1

            except Exception as e:
                stats["errors"].append(f"{note.path}: {str(e)}")

        # 동기화 로그 저장
        self._save_sync_log(stats)

        return stats

    def export_to_obsidian(self, cards: List[Dict], folder: str = "ADHD Learning") -> Dict:
        """학습 카드를 Obsidian 노트로 내보내기"""
        export_folder = self.vault_path / folder
        export_folder.mkdir(parents=True, exist_ok=True)

        stats = {
            "notes_created": 0,
            "notes_updated": 0,
            "errors": []
        }

        # 태그별로 그룹화
        cards_by_tag = {}
        for card in cards:
            tags = card.get("tags", ["untagged"])
            primary_tag = tags[0] if tags else "untagged"
            if primary_tag not in cards_by_tag:
                cards_by_tag[primary_tag] = []
            cards_by_tag[primary_tag].append(card)

        for tag, tag_cards in cards_by_tag.items():
            try:
                note_path = export_folder / f"{tag}.md"
                content = self._generate_obsidian_note(tag, tag_cards)

                if note_path.exists():
                    stats["notes_updated"] += 1
                else:
                    stats["notes_created"] += 1

                with open(note_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            except Exception as e:
                stats["errors"].append(f"{tag}: {str(e)}")

        return stats

    def _generate_obsidian_note(self, topic: str, cards: List[Dict]) -> str:
        """Obsidian 노트 생성"""
        lines = [
            "---",
            f"title: {topic}",
            f"created: {datetime.now().isoformat()}",
            "tags:",
            f"  - {topic}",
            "  - adhd-learning",
            "---",
            "",
            f"# {topic}",
            "",
            "*ADHD Learning System에서 자동 생성됨*",
            "",
            "---",
            "",
        ]

        for card in cards:
            lines.extend([
                "<!-- flashcard -->",
                card.get("content", ""),
                "---",
                card.get("answer", ""),
                "",
                f"*마지막 복습: {card.get('last_review', 'N/A')}*",
                "",
            ])

        lines.extend([
            "---",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ])

        return "\n".join(lines)

    def _save_sync_log(self, stats: Dict):
        """동기화 로그 저장"""
        log = {
            "last_sync": datetime.now().isoformat(),
            "stats": stats
        }

        with open(self.sync_log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

    def get_last_sync(self) -> Optional[Dict]:
        """마지막 동기화 정보"""
        if self.sync_log_path.exists():
            with open(self.sync_log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


# CLI 도구
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Obsidian Integration CLI")
    parser.add_argument("vault_path", help="Obsidian vault path")
    parser.add_argument("--scan", action="store_true", help="Scan vault for notes")
    parser.add_argument("--import", dest="do_import", action="store_true", help="Import to learning system")
    parser.add_argument("--export", action="store_true", help="Export to Obsidian")
    parser.add_argument("--folder", default=None, help="Specific folder to scan")

    args = parser.parse_args()

    try:
        obsidian = ObsidianIntegration(args.vault_path)

        if args.scan:
            folders = [args.folder] if args.folder else None
            notes = obsidian.scan_vault(folders)
            print(f"Found {len(notes)} notes")

            for note in notes[:10]:
                print(f"  - {note.title} ({len(note.tags)} tags, {len(note.links)} links)")

                # 플래시카드 추출 테스트
                flashcards = obsidian.extract_flashcards(note)
                if flashcards:
                    print(f"    -> {len(flashcards)} flashcards found")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
