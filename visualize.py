"""
visualize.py — Local development tool for Track 2 (Spatial Engine).

Runs the full embed → UMAP(2D) → HDBSCAN → interactive plot pipeline
on the LOCAL CPU (no Modal GPU required). Great for rapid iteration.

Usage:
    # activate venv first
    source .venv/bin/activate

    python visualize.py
    python visualize.py --input Facebook_JSON_Output.json
    python visualize.py --input Facebook_JSON_Output.json --output my_plot.html

Output:
    cluster_visualization.html  — interactive Plotly scatter plot
    cluster_results.json        — comments with cluster_id + centroids

Visualisation features:
  • 2D UMAP scatter (embed-space positions)
  • Colour-coded by cluster (noise = grey)
  • Hover shows author, text preview, reaction count
  • Cluster centroid markers (★)
  • Silhouette score and cluster count in title
"""
from __future__ import annotations

import argparse
import json
import pathlib
import textwrap
from datetime import datetime
from typing import List

import numpy as np


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Candor embedding visualizer")
    p.add_argument(
        "--input",
        default="Facebook_JSON_Output.json",
        help="Path to Facebook JSON post file (default: Facebook_JSON_Output.json)",
    )
    p.add_argument(
        "--output",
        default="cluster_visualization.html",
        help="Output HTML file (default: cluster_visualization.html)",
    )
    p.add_argument(
        "--n-components-umap-2d",
        type=int,
        default=2,
        help="UMAP components for visualization (default: 2)",
    )
    p.add_argument(
        "--n-components-umap-cluster",
        type=int,
        default=5,
        help="UMAP components for clustering (default: 5, per TASKS.md)",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="HDBSCAN min_cluster_size (default: 2 for small test sets)",
    )
    p.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_comments(json_path: pathlib.Path) -> list[dict]:
    """
    Parse a FacebookPost JSON file and return a flat list of comment dicts
    with keys: comment_id, author, text, reaction_count, created_time.
    Skips empty / image-only comments.
    """
    from schema import FacebookPost, Comment

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    post = FacebookPost(**raw)

    flat: list[dict] = []

    def _epoch(ts: str | None) -> float:
        if not ts:
            return 0.0
        try:
            return datetime.fromisoformat(ts).timestamp()
        except ValueError:
            return 0.0

    def _process(fb_comment, depth: int = 0) -> None:
        text = (fb_comment.message or "").strip()
        if not text:
            return
        flat.append({
            "comment_id": fb_comment.comment_id,
            "author": fb_comment.author.name,
            "text": text,
            "reaction_count": fb_comment.reaction_count,
            "timestamp": _epoch(fb_comment.created_time),
            "cluster_id": None,
        })
        for reply in fb_comment.replies:
            _process(reply, depth + 1)

    for c in post.comments:
        _process(c)

    return flat


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"[embed] Loading '{model_name}'…")
    model = SentenceTransformer(model_name)
    print(f"[embed] Encoding {len(texts)} documents…")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"[embed] Done — shape {embeddings.shape}")
    return embeddings


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_umap(embeddings: np.ndarray, n_components: int, label: str = "") -> np.ndarray:
    from umap import UMAP
    # n_neighbors must be < n_samples
    n_neighbors = min(15, len(embeddings) - 1)
    print(f"[umap{label}] Reducing to {n_components}D (n_neighbors={n_neighbors})…")
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        compatibility_layout=True,  # pin legacy behavior, silence FutureWarning
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"[umap{label}] Done — shape {reduced.shape}")
    return reduced


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_hdbscan(reduced_5d: np.ndarray, min_cluster_size: int) -> np.ndarray:
    from hdbscan import HDBSCAN
    print(f"[hdbscan] Clustering (min_cluster_size={min_cluster_size})…")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced_5d)
    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    print(f"[hdbscan] Found {n_clusters} cluster(s), {n_noise} noise point(s).")
    return labels


# ── Centroids ─────────────────────────────────────────────────────────────────

def compute_centroids(
    reduced: np.ndarray, labels: np.ndarray
) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for cid in set(labels):
        if cid == -1:
            continue
        centroids[int(cid)] = reduced[labels == cid].mean(axis=0)
    return centroids


# ── Quality metric ────────────────────────────────────────────────────────────

def silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    from sklearn.metrics import silhouette_score
    valid = labels != -1
    if valid.sum() < 2 or len(set(labels[valid])) < 2:
        return None
    try:
        return float(silhouette_score(embeddings[valid], labels[valid]))
    except Exception:
        return None


# ── Plotly visualisation ──────────────────────────────────────────────────────

CLUSTER_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
NOISE_COLOR = "#CCCCCC"


def build_plot(
    comments: list[dict],
    xy: np.ndarray,
    labels: np.ndarray,
    centroids_2d: dict[int, np.ndarray],
    sil_score: float | None,
    output_path: pathlib.Path,
) -> None:
    import plotly.graph_objects as go

    n_clusters = len(set(labels) - {-1})
    sil_text = f"Silhouette: {sil_score:.3f}" if sil_score is not None else "Silhouette: N/A"
    title = (
        f"Candor — Embedding Space  |  {n_clusters} cluster(s)  |  {sil_text}"
        f"<br><sup>Model: all-MiniLM-L6-v2 · UMAP(5D→2D) · HDBSCAN</sup>"
    )

    fig = go.Figure()

    unique_labels = sorted(set(labels))

    for idx, cid in enumerate(unique_labels):
        is_noise = cid == -1
        mask = labels == cid
        color = NOISE_COLOR if is_noise else CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        name = "Noise / Outlier" if is_noise else f"Cluster {cid}"

        hover_texts = []
        for i, c in enumerate(comments):
            if not mask[i]:
                continue
            snippet = textwrap.shorten(c["text"], width=80, placeholder="…")
            hover_texts.append(
                f"<b>{c['author']}</b><br>"
                f"{snippet}<br>"
                f"❤ {c['reaction_count']}"
            )

        fig.add_trace(go.Scatter(
            x=xy[mask, 0],
            y=xy[mask, 1],
            mode="markers",
            name=name,
            marker=dict(
                size=12,
                color=color,
                opacity=0.85 if not is_noise else 0.4,
                line=dict(width=1, color="white"),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))

    # Centroid stars
    for cid, centroid_xy in centroids_2d.items():
        color = CLUSTER_COLORS[
            sorted(set(labels) - {-1}).index(cid) % len(CLUSTER_COLORS)
        ]
        fig.add_trace(go.Scatter(
            x=[centroid_xy[0]],
            y=[centroid_xy[1]],
            mode="markers+text",
            name=f"Centroid {cid}",
            marker=dict(
                size=20,
                color=color,
                symbol="star",
                line=dict(width=2, color="black"),
            ),
            text=[f"★ C{cid}"],
            textposition="top center",
            hovertemplate=f"Centroid of Cluster {cid}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
        plot_bgcolor="rgb(245,245,250)",
        paper_bgcolor="white",
        hovermode="closest",
        width=1000,
        height=700,
        font=dict(family="Inter, Arial, sans-serif"),
    )

    fig.write_html(str(output_path), include_plotlyjs=True)  # fully self-contained, no internet needed
    print(f"[plot] Saved → {output_path.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    base = pathlib.Path(__file__).parent
    json_path = base / args.input
    output_path = base / args.output
    results_path = base / "cluster_results.json"

    # 1. Load
    print(f"\n{'='*60}")
    print(f"[load] Reading {json_path.name}…")
    comments = load_comments(json_path)
    if not comments:
        print("[load] No text comments found. Exiting.")
        return
    print(f"[load] {len(comments)} comments with text.")

    texts = [c["text"] for c in comments]

    # 2. Embed
    embeddings = embed(texts, args.model)

    # 3. Reduce to 5D for clustering (per TASKS.md spec)
    reduced_5d = reduce_umap(embeddings, args.n_components_umap_cluster, label="(cluster)")

    # 4. Cluster
    labels = cluster_hdbscan(reduced_5d, args.min_cluster_size)

    # 5. Reduce to 2D for visualization
    reduced_2d = reduce_umap(embeddings, 2, label="(viz)")

    # 6. Centroids (in 2D for plotting, in 5D for pipeline)
    centroids_2d = compute_centroids(reduced_2d, labels)
    centroids_5d = compute_centroids(reduced_5d, labels)

    # 7. Silhouette score
    sil = silhouette(reduced_5d, labels)
    print(f"[eval] Silhouette score: {sil:.4f}" if sil is not None else "[eval] Silhouette: N/A (need ≥2 clusters with ≥2 points each)")

    # 8. Attach cluster_id to comments
    for c, label in zip(comments, labels):
        c["cluster_id"] = int(label)

    # 9. Save results JSON
    results = {
        "comments": comments,
        "centroids": {str(k): v.tolist() for k, v in centroids_5d.items()},
        "silhouette_score": sil,
        "n_clusters": len(set(labels) - {-1}),
    }
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[save] Results saved → {results_path.resolve()}")

    # 10. Visualize
    build_plot(comments, reduced_2d, labels, centroids_2d, sil, output_path)

    # 11. Summary table
    print(f"\n{'='*60}")
    print(f"{'CLUSTER':>8}  {'AUTHOR':<22}  TEXT PREVIEW")
    print(f"{'-'*60}")
    for c in sorted(comments, key=lambda x: x["cluster_id"]):
        cid = c["cluster_id"]
        label = f"[noise]" if cid == -1 else f"[  {cid:>2}  ]"
        preview = textwrap.shorten(c["text"], width=50, placeholder="…")
        print(f"{label:>8}  {c['author']:<22}  {preview}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
