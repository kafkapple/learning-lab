# learning-lab

Personal mono-repo for study materials, self-development tools, and learning projects.

## Structure

```
learning-lab/
├── courses/              # Academic course study
│   └── MVG/              # Multi-View Geometry (RANSAC, SfM)
├── tools/                # Self-development & productivity tools
│   └── adaptive-learning-system/  # Focus-optimized spaced repetition system
├── projects/             # Standalone projects (future)
└── docs/                 # Shared documentation & guides
```

## Courses

| Course | Topics | Status |
|--------|--------|--------|
| `MVG` | RANSAC, Structure from Motion | In Progress |

## Tools

| Tool | Description |
|------|-------------|
| `adaptive-learning-system` | FSRS-based learning automation with gamification |

## Adding New Projects

| Category | Path | Examples |
|----------|------|----------|
| Academic courses | `courses/<subject>/` | `courses/NLP/`, `courses/RL/` |
| Tools & utilities | `tools/<tool-name>/` | `tools/pomodoro/`, `tools/flashcard/` |
| Standalone projects | `projects/<name>/` | `projects/paper-impl/` |
| Notes & references | `notes/<topic>/` | `notes/linear-algebra/` |

Each sub-project should have its own `README.md` and `requirements.txt` (if applicable).
