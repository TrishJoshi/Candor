"""
spatial_engine.py — Track 2: The Spatial Engine (Local Execution).
"""
from typing import List, Tuple
import numpy as np
from schema import Comment, CentroidsDict

# Lazy load so we don't incur import cost if not used
_engine = None

class LocalSpatialEngine:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN
        from bertopic import BERTopic

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer on device: {device}")

        self.topic_model = BERTopic(
            embedding_model=SentenceTransformer("all-MiniLM-L6-v2", device=device),
            umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42),
            hdbscan_model=HDBSCAN(min_cluster_size=3, metric="euclidean", prediction_data=True),
            verbose=True,
        )

    def run(self, comment_dicts: list[dict]) -> dict:
        from sklearn.decomposition import PCA

        if not comment_dicts:
            return {"comments": [], "centroids": {}}

        docs = [c["text"] for c in comment_dicts]
        topics, _ = self.topic_model.fit_transform(docs)

        for c, t in zip(comment_dicts, topics):
            c["cluster_id"] = int(t)

        reduced_5d = self.topic_model.umap_model.embedding_  # shape (N, 5)
        
        # Project 5D to 2D for visualization using PCA to be fast
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


def run_spatial_engine(comments: List[Comment]) -> Tuple[List[Comment], CentroidsDict]:
    """
    Groups comments into clusters.
    Returns the updated comments with `cluster_id` and a dictionary mapping `cluster_id` to its mathematical centroid.
    """
    global _engine
    if _engine is None:
        _engine = LocalSpatialEngine()
        
    comment_dicts = [c.model_dump() for c in comments]
    result = _engine.run(comment_dicts)
    
    updated_comments = [Comment(**c) for c in result["comments"]]
    centroids = {int(k): v for k, v in result["centroids"].items()}
    
    return updated_comments, centroids

if __name__ == "__main__":
    print("Run `python run_and_plot.py` to test the local spatial engine.")
