# Track 2 – Spatial Engine: Modal + BERTopic Reference Notes

> **Verdict: Use BERTopic on Modal GPUs.** BERTopic fully supports custom UMAP/HDBSCAN backends and runs cleanly on Modal GPU containers. This avoids reinventing the wheel while keeping the exact pipeline specified in TASKS.md.

---

## 1. Architecture Decision

| Option | Verdict | Reason |
|---|---|---|
| **BERTopic on Modal GPU** | ✅ **Use this** | BERTopic accepts plug-in UMAP/HDBSCAN backends; `fit_transform()` returns `topics` list (= our `cluster_id`); Modal T4/A10G supports it out of the box |
| Build our own | ❌ Avoid | Only needed if BERTopic's internals break our schema; they don't |

---

## 2. Modal Primitives We Use

### App & Image
```python
import modal

app = modal.App("candor-spatial-engine")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "sentence-transformers",
        "umap-learn",
        "hdbscan",
        "bertopic",
        "numpy",
        "scikit-learn",
    )
)
```

### GPU Function / Class Pattern
Two valid patterns — **prefer the Class pattern** so the model loads once per container (avoids cold-start re-download on every call):

```python
# Pattern A: @app.cls (recommended — model loaded once)
@app.cls(gpu="T4", image=image, timeout=600)
class SpatialEngine:
    @modal.enter()
    def load(self):
        ...  # load model here

    @modal.method()
    def run(self, comments): ...

# Pattern B: @app.function (simpler, model reloads per call)
@app.function(gpu="T4", image=image, timeout=600)
def run_spatial(comments): ...
```

### GPU Options (cheapest → most powerful)
| Token | GPU | VRAM | Notes |
|---|---|---|---|
| `"T4"` | NVIDIA T4 | 16 GB | ✅ Sufficient for all-MiniLM-L6-v2 + HDBSCAN |
| `"L4"` | NVIDIA L4 | 24 GB | Good for larger batches |
| `"A10G"` | NVIDIA A10G | 24 GB | Faster than L4 |
| `"A100"` | NVIDIA A100 | 40/80 GB | Overkill for this task |

**Use `gpu="T4"` as specified in TASKS.md.** `all-MiniLM-L6-v2` is 90 MB; T4 16 GB is more than enough.

### Volume (model weight caching — avoids re-download)
```python
volume = modal.Volume.from_name("candor-model-cache", create_if_missing=True)

@app.cls(
    gpu="T4",
    image=image,
    volumes={"/root/.cache": volume},
    timeout=600
)
class SpatialEngine:
    ...
```

### Calling Remotely (from FastAPI / orchestrator)
```python
# From local/orchestration layer:
engine = SpatialEngine()
result = engine.run.remote(comments_list)

# Or via deployed app:
# modal deploy spatial_engine.py
# then import and call .remote() from main.py
```

---

## 3. BERTopic Integration

### How `fit_transform` maps to our schema
```
docs (List[str])  →  topics (List[int])  →  comment.cluster_id
```
- `topics[i]` is the cluster ID for `docs[i]`  
- `-1` = outlier/noise (HDBSCAN convention) — keep as-is or remap to `None`

### Minimal BERTopic setup matching TASKS.md spec
```python
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

umap_model = UMAP(
    n_components=5,       # as specified in TASKS.md
    n_neighbors=15,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

hdbscan_model = HDBSCAN(
    min_cluster_size=3,   # as specified in TASKS.md
    metric="euclidean",
    prediction_data=True,
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True,
)

docs = [c.text for c in comments]
topics, _ = topic_model.fit_transform(docs)
# topics[i] → comments[i].cluster_id
```

### Why BERTopic (not raw pipeline)
- BERTopic coordinates the embed → reduce → cluster pipeline and handles edge cases (empty clusters, outliers)
- `get_topic_info()` gives us cluster metadata for free
- Custom UMAP/HDBSCAN models are injected as plain arguments — no lock-in

---

## 4. Centroid Calculation (TASKS.md requirement)

BERTopic does **not** expose raw centroids directly. Compute them from the UMAP-reduced embeddings **after** `fit_transform`:

```python
import numpy as np

# After: topics, _ = topic_model.fit_transform(docs)
reduced = topic_model.umap_model.embedding_  # shape: (N, 5)

centroids = {}
for cluster_id in set(topics):
    if cluster_id == -1:
        continue  # skip noise
    mask = np.array(topics) == cluster_id
    centroids[cluster_id] = reduced[mask].mean(axis=0).tolist()
# Returns: Dict[int, List[float]]  (5-dim vector per cluster)
```

---

## 5. Full `spatial_engine.py` Skeleton

```python
import modal
from typing import List
# schema imported from shared schema.py
# from schema import Comment

app = modal.App("candor-spatial-engine")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers",
        "umap-learn",
        "hdbscan",
        "bertopic",
        "numpy",
        "pydantic",
    )
)

volume = modal.Volume.from_name("candor-model-cache", create_if_missing=True)

@app.cls(
    gpu="T4",
    image=image,
    volumes={"/root/.cache": volume},
    timeout=600,
)
class SpatialEngine:

    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN
        from bertopic import BERTopic

        self.topic_model = BERTopic(
            embedding_model=SentenceTransformer("all-MiniLM-L6-v2", device="cuda"),
            umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                            metric="cosine", random_state=42),
            hdbscan_model=HDBSCAN(min_cluster_size=3, metric="euclidean",
                                  prediction_data=True),
            verbose=True,
        )

    @modal.method()
    def run(self, comments: list) -> dict:
        """
        Input:  list of Comment dicts (serialised from Pydantic)
        Output: {"comments": [...with cluster_id set...], "centroids": {...}}
        """
        import numpy as np

        docs = [c["text"] for c in comments]
        topics, _ = self.topic_model.fit_transform(docs)

        # Attach cluster_id
        for c, t in zip(comments, topics):
            c["cluster_id"] = int(t)  # -1 stays as -1 (noise)

        # Compute centroids
        reduced = self.topic_model.umap_model.embedding_
        centroids = {}
        for cluster_id in set(topics):
            if cluster_id == -1:
                continue
            mask = np.array(topics) == cluster_id
            centroids[int(cluster_id)] = reduced[mask].mean(axis=0).tolist()

        return {"comments": comments, "centroids": centroids}


@app.local_entrypoint()
def main():
    from ingest import get_mock_comments  # Track 1 mock
    comments = [c.model_dump() for c in get_mock_comments()]
    engine = SpatialEngine()
    result = engine.run.remote(comments)
    print(f"Clusters found: {set(c['cluster_id'] for c in result['comments'])}")
    print(f"Centroids: {list(result['centroids'].keys())}")
```

---

## 6. Deployment Commands

```bash
# One-off test run (uses Modal cloud, billed per second of GPU time)
modal run spatial_engine.py

# Persistent deployment (callable via .remote() from main.py)
modal deploy spatial_engine.py
```

---

## 7. Key Gotchas

| Issue | Fix |
|---|---|
| **Model re-downloads on every cold start** | Mount a `modal.Volume` to `/root/.cache` |
| **UMAP/HDBSCAN run on CPU** | That's fine — only embedding inference benefits from GPU. UMAP/HDBSCAN on CPU is fast enough for Reddit-scale data |
| **`-1` cluster_id (noise)** | Leave as `-1` or convert to `None`. Track 3 scorer should skip these |
| **Serialisation** | Pass `comment.model_dump()` (dict) over Modal boundary, not Pydantic objects |
| **BERTopic re-fit on each call** | For a hackathon pipeline this is fine. If needed, persist `topic_model` via `modal.Volume` between calls |
| **cuML (GPU UMAP/HDBSCAN)** | Optional: swap `umap-learn`/`hdbscan` for `cuml.manifold.UMAP` / `cuml.cluster.HDBSCAN`. Needs `pip_install("cuml-cu12", extra_index_url="https://pypi.nvidia.com")`. Not needed for T4 at hackathon scale |
