"""
spatial_engine.py — Track 2: The Spatial Engine (Modal GPU deployment).

Inputs : List[Comment] (Candor pipeline schema)
Outputs: List[Comment] with cluster_id populated + centroids dict

Modal deployment:
    modal run spatial_engine.py        # one-off test
    modal deploy spatial_engine.py     # persistent deployment

Local CPU test (no Modal):
    See visualize.py instead.
"""
from __future__ import annotations

import json
from typing import List, Tuple
from datetime import datetime, timezone

import modal

from schema import Comment, FacebookPost, CentroidsDict

# ── Modal image ───────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0,<4",
        "umap-learn>=0.5,<1",
        "hdbscan>=0.8,<1",
        "bertopic>=0.16,<1",
        "numpy>=1.26,<3",
        "scikit-learn>=1.4,<2",
        "pydantic>=2.7,<3",
    )
)

volume = modal.Volume.from_name("candor-model-cache", create_if_missing=True)
app = modal.App("candor-spatial-engine", image=image)


# ── Modal class ───────────────────────────────────────────────────────────────

@app.cls(
    volumes={"/root/.cache/model_cache": volume},
    timeout=600,
)
class SpatialEngine:
    """Loads the embedding model once per container lifecycle."""

    @modal.enter()
    def load(self) -> None:
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN
        from bertopic import BERTopic

        self.topic_model = BERTopic(
            embedding_model=SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu"
            ),
            umap_model=UMAP(
                n_components=5,
                n_neighbors=15,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=3,
                metric="euclidean",
                prediction_data=True,
            ),
            verbose=True,
        )

    @modal.method()
    def run(self, comment_dicts: list[dict]) -> dict:
        """
        Args:
            comment_dicts: List of Comment.model_dump() dicts.
        Returns:
            {
                "comments": [Comment dicts with cluster_id set],
                "centroids": {cluster_id: [5-dim float list]},
            }
        """
        import numpy as np

        if not comment_dicts:
            return {"comments": [], "centroids": {}}

        docs = [c["text"] for c in comment_dicts]
        topics, _ = self.topic_model.fit_transform(docs)

        for c, t in zip(comment_dicts, topics):
            c["cluster_id"] = int(t)

        reduced_5d = self.topic_model.umap_model.embedding_  # shape (N, 5)
        
        # Project 5D to 2D for visualization using PCA to be fast
        from sklearn.decomposition import PCA
        if len(reduced_5d) > 2:
            reduced_2d = PCA(n_components=2).fit_transform(reduced_5d)
        else:
            reduced_2d = reduced_5d[:, :2] if reduced_5d.shape[1] >= 2 else np.zeros((len(reduced_5d), 2))

        for c, vec in zip(comment_dicts, reduced_2d):
            c["vector"] = vec.tolist()

        centroids: dict[int, list[float]] = {}
        for cluster_id in set(topics):
            if cluster_id == -1:
                continue
            mask = np.array(topics) == cluster_id
            centroids[int(cluster_id)] = reduced_5d[mask].mean(axis=0).tolist()

        return {"comments": comment_dicts, "centroids": centroids}


# ── Helpers ───────────────────────────────────────────────────────────────────

def facebook_post_to_comments(post: FacebookPost) -> List[Comment]:
    """Flatten a FacebookPost into the Candor Comment schema."""
    comments: List[Comment] = []

    def _created_time_to_epoch(ts: str | None) -> float:
        if not ts:
            return 0.0
        try:
            return datetime.fromisoformat(ts).timestamp()
        except ValueError:
            return 0.0

    def _process(fb_comment, depth: int = 0) -> None:
        text = (fb_comment.message or "").strip()
        if not text:
            # skip image-only / empty comments
            return
        comments.append(
            Comment(
                comment_id=fb_comment.comment_id,
                author=fb_comment.author.name,
                account_age_days=0,
                timestamp=_created_time_to_epoch(fb_comment.created_time),
                text=text,
            )
        )
        for reply in fb_comment.replies:
            _process(reply, depth + 1)

    for fb_comment in post.comments:
        _process(fb_comment)

    return comments



def run_spatial_engine(comments: List[Comment]) -> Tuple[List[Comment], CentroidsDict]:
    """
    Groups comments into clusters.
    Returns the updated comments with `cluster_id` and a dictionary mapping `cluster_id` to its mathematical centroid.
    """
    engine = SpatialEngine()
    comment_dicts = [c.model_dump() for c in comments]
    result = engine.run.remote(comment_dicts)
    
    updated_comments = [Comment(**c) for c in result["comments"]]
    centroids = {int(k): v for k, v in result["centroids"].items()}
    
    return updated_comments, centroids


# ── Local entrypoint (modal run spatial_engine.py) ───────────────────────────

@app.local_entrypoint()
def main() -> None:
    import pathlib, json

    data_path = pathlib.Path(__file__).parent / "Facebook_JSON_Output.json"
    raw = json.loads(data_path.read_text())
    post = FacebookPost(**raw)
    comments = facebook_post_to_comments(post)

    print(f"Loaded {len(comments)} comments with text.")

    comment_dicts = [c.model_dump() for c in comments]
    engine = SpatialEngine()
    result = engine.run.remote(comment_dicts)

    cluster_ids = {c["cluster_id"] for c in result["comments"]}
    print(f"Clusters found: {cluster_ids}")
    print(f"Centroids computed for: {list(result['centroids'].keys())}")
    for c in result["comments"]:
        print(f"  [{c['cluster_id']:>2}] {c['author']}: {c['text'][:60]}")
