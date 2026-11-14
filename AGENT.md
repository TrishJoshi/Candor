# AGENT.md — Candor Project: Development Rules

> This file defines the **mandatory conventions** every agent and developer must follow when working in this repository. Read it before writing any code.

---

## 1. Environment Setup

**Always use a virtual environment. Never install packages globally.**

```bash
# Create the venv (run once at repo root)
python3 -m venv .venv

# Activate it (every session)
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install all dependencies
pip install -r requirements.txt
```

> [!IMPORTANT]
> Never run `pip install <package>` directly. Always add the package to `requirements.txt` first, then run `pip install -r requirements.txt`.

---

## 2. Dependency Management

- All Python dependencies live in **`requirements.txt`** at the repo root.
- Pin major versions (e.g., `sentence-transformers>=2.7,<3`).
- Group dependencies by track/module with comments.
- Never commit a modified `requirements.txt` without testing it installs cleanly in a fresh venv.

---

## 3. Project Structure

```
Candor/
├── .venv/                  # Virtual environment (gitignored)
├── requirements.txt        # All Python deps
├── AGENT.md                # This file — read before coding
├── schema.py               # Shared Pydantic models (source of truth)
├── ingest.py               # Track 1 — ingestion
├── spatial_engine.py       # Track 2 — Modal GPU clustering
├── scorer.py               # Track 3 — authenticity scoring
├── decoder.py              # Track 4 — LLM labelling
├── main.py                 # Track 5 — FastAPI orchestration
├── visualize.py            # Local dev utility — embedding visualizer
├── BIG_PICTURE.md          # Architecture overview
├── TASKS.md                # Track-by-track task breakdown
└── MODAL_BERTOPIC_NOTES.md # Modal + BERTopic reference notes
```

---

## 4. Data Contract

**All data passed between modules must conform to `schema.py`.**

- Input/output types must be `List[Comment]` or `List[NarrativeCluster]`.
- Never pass raw dicts between modules — deserialize with Pydantic at the boundary.
- When crossing the Modal boundary (`.remote()` calls), serialize with `.model_dump()` and deserialize back with `Model(**data)`.

---

## 5. Module Boundaries

| File | Owns | Does NOT own |
|---|---|---|
| `ingest.py` | Scraping, comment parsing, flattening | Any scoring or clustering |
| `spatial_engine.py` | Embedding, UMAP, HDBSCAN, centroids | Any LLM calls |
| `scorer.py` | Authenticity math, blacklist feedback loop | Embedding computation |
| `decoder.py` | LLM label generation | Clustering or scoring |
| `main.py` | Pipeline wiring, FastAPI routes, HTML serving | Business logic |

---

## 6. Local Testing

- `visualize.py` is the **local dev tool** for the spatial engine — it runs the full embed → reduce → cluster → plot pipeline on a local CPU (no Modal GPU required).
- Run it with: `python visualize.py --input Facebook_JSON_Output.json`
- It produces `cluster_visualization.html` — open in a browser to inspect cluster separation.
- For Modal GPU runs: `modal run spatial_engine.py`

---

## 7. Code Style

- Python 3.11+.
- Type hints on all function signatures.
- Pydantic models in `schema.py` — no raw `dict` types in function signatures.
- Short, single-purpose functions. No function longer than 50 lines.
- `f-strings` for formatting, never `%` or `.format()`.
