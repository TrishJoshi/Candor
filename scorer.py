import os
import statistics
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from schema import Comment, NarrativeCluster, CentroidsDict

BLACKLIST_PATH = "blacklist.npy"

def _load_blacklist() -> np.ndarray:
    if os.path.exists(BLACKLIST_PATH):
        try:
            return np.load(BLACKLIST_PATH)
        except Exception:
            pass
    return np.empty((0, 0))

def _append_to_blacklist(centroid: List[float]):
    bl = _load_blacklist()
    centroid_arr = np.array([centroid], dtype=float)
    if bl.size == 0:
        bl = centroid_arr
    else:
        if bl.shape[1] == centroid_arr.shape[1]:
            bl = np.vstack((bl, centroid_arr))
    np.save(BLACKLIST_PATH, bl)

def score_clusters(comments: List[Comment], centroids: CentroidsDict) -> List[NarrativeCluster]:
    """
    Evaluates comments for authenticity and groups them into NarrativeCluster objects.
    Returns a list of NarrativeClusters with `account_authenticity_score` and `cluster_score` populated.
    """
    # Group comments by author
    author_comments = {}
    for c in comments:
        author_comments.setdefault(c.author, []).append(c)

    # 1. Velocity Math
    author_velocity_score = {}
    for author, author_comms in author_comments.items():
        if len(author_comms) > 1:
            timestamps = [c.timestamp for c in author_comms]
            var = np.var(timestamps)
            # Map variance to score (e.g. var >= 1000 -> 100 score)
            score = min(100.0, (var / 1000.0) * 100.0) if var > 0 else 0.0
            author_velocity_score[author] = score
        else:
            author_velocity_score[author] = 100.0

    # Group comments by cluster_id
    cluster_comments = {}
    for c in comments:
        if c.cluster_id is not None:
            cluster_comments.setdefault(c.cluster_id, []).append(c)

    # 2. n-Gram Math
    author_ngram_score = {author: 100.0 for author in author_comments}
    for cluster_id, comms_in_cluster in cluster_comments.items():
        authors_in_cluster = list(set(c.author for c in comms_in_cluster))
        if len(authors_in_cluster) > 1:
            author_texts = []
            for author in authors_in_cluster:
                texts = [c.text for c in comms_in_cluster if c.author == author]
                author_texts.append(" ".join(texts))
            
            vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
            try:
                tfidf_matrix = vec.fit_transform(author_texts)
                sim_matrix = cosine_similarity(tfidf_matrix)
                
                for i, author in enumerate(authors_in_cluster):
                    sims = list(sim_matrix[i])
                    sims.pop(i) # remove self similarity
                    max_sim = max(sims) if sims else 0.0
                    score = max(0.0, (1.0 - max_sim) * 100.0)
                    if score < author_ngram_score[author]:
                        author_ngram_score[author] = score
            except ValueError:
                pass

    # Combine signals
    author_authenticity = {}
    for author in author_comments:
        v_score = author_velocity_score[author]
        n_score = author_ngram_score[author]
        author_authenticity[author] = (v_score + n_score) / 2.0

    # 3. Feedback Loop Penalty
    blacklist = _load_blacklist()
    if blacklist.size > 0:
        for c in comments:
            if c.cluster_id is not None and c.cluster_id in centroids:
                centroid = np.array([centroids[c.cluster_id]])
                if blacklist.shape[1] == centroid.shape[1]:
                    distances = cosine_distances(centroid, blacklist)[0]
                    if min(distances) < 0.1: # Tight cosine distance
                        author_authenticity[c.author] -= 50.0

    # Assign scores to comments and clamp to 0-100
    for author in author_authenticity:
        author_authenticity[author] = max(0.0, min(100.0, author_authenticity[author]))

    for c in comments:
        c.account_authenticity_score = author_authenticity[c.author]

    # 4. Aggregate to cluster_score and update blacklist
    narrative_clusters = []
    for cluster_id, comms in cluster_comments.items():
        authors_in_cluster = list(set(c.author for c in comms))
        scores = [author_authenticity[author] for author in authors_in_cluster]
        cluster_score = statistics.median(scores) if scores else 100.0
        
        if cluster_score < 30.0 and cluster_id in centroids:
            _append_to_blacklist(centroids[cluster_id])

        nc = NarrativeCluster(
            cluster_id=cluster_id,
            cluster_score=cluster_score,
            comments=comms
        )
        narrative_clusters.append(nc)

    return narrative_clusters
