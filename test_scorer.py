import os
import pytest
import numpy as np
from schema import Comment, NarrativeCluster
from scorer import score_clusters, BLACKLIST_PATH, _load_blacklist

@pytest.fixture(autouse=True)
def clean_blacklist():
    if os.path.exists(BLACKLIST_PATH):
        os.remove(BLACKLIST_PATH)
    yield
    if os.path.exists(BLACKLIST_PATH):
        os.remove(BLACKLIST_PATH)

def test_single_comment():
    comments = [
        Comment(
            comment_id="1",
            author="alice",
            timestamp=1000.0,
            text="This is a normal comment.",
            cluster_id=1
        )
    ]
    centroids = {1: [0.1, 0.2, 0.3]}
    
    clusters = score_clusters(comments, centroids)
    
    assert len(clusters) == 1
    assert clusters[0].cluster_id == 1
    assert clusters[0].cluster_score == 100.0
    assert comments[0].account_authenticity_score == 100.0

def test_velocity_penalty():
    # Bob posts multiple times with very low variance (bursty)
    comments = [
        Comment(
            comment_id="1",
            author="bob",
            timestamp=1000.0,
            text="First post",
            cluster_id=1
        ),
        Comment(
            comment_id="2",
            author="bob",
            timestamp=1001.0, # only 1 sec apart
            text="Second post",
            cluster_id=1
        )
    ]
    centroids = {1: [0.1, 0.2, 0.3]}
    
    clusters = score_clusters(comments, centroids)
    
    assert len(clusters) == 1
    # Variance is 0.25 -> (0.25/1000)*100 = 0.025 v_score
    # n-gram score is 100 (only one author in cluster)
    # Total score = (0.025 + 100) / 2 = 50.0125
    assert 49.0 <= clusters[0].cluster_score <= 51.0
    assert 49.0 <= comments[0].account_authenticity_score <= 51.0

def test_ngram_penalty():
    # Two different authors in the same cluster post identical text
    comments = [
        Comment(
            comment_id="1",
            author="charlie",
            timestamp=1000.0,
            text="This is identical spam text across accounts.",
            cluster_id=2
        ),
        Comment(
            comment_id="2",
            author="dave",
            timestamp=2000.0,
            text="This is identical spam text across accounts.",
            cluster_id=2
        )
    ]
    centroids = {2: [0.5, 0.5, 0.5]}
    
    clusters = score_clusters(comments, centroids)
    
    assert len(clusters) == 1
    # n-gram score should be 0 because of identical text
    # velocity score is 100 because single comment per author
    # Total score = (100 + 0) / 2 = 50
    assert clusters[0].cluster_score == 50.0
    assert comments[0].account_authenticity_score == 50.0
    assert comments[1].account_authenticity_score == 50.0

def test_feedback_loop_and_blacklist():
    # 1. Create a bad cluster to populate blacklist
    # Both low velocity variance AND high n-gram similarity to force score < 30
    comments_bad = [
        Comment(comment_id="1", author="eve", timestamp=1000.0, text="spam text", cluster_id=3),
        Comment(comment_id="2", author="eve", timestamp=1000.0, text="spam text", cluster_id=3),
        Comment(comment_id="3", author="frank", timestamp=1000.0, text="spam text", cluster_id=3),
        Comment(comment_id="4", author="frank", timestamp=1000.0, text="spam text", cluster_id=3),
    ]
    centroids = {3: [1.0, 0.0, 0.0]}
    
    clusters = score_clusters(comments_bad, centroids)
    
    assert len(clusters) == 1
    # Variance is 0, so v_score = 0
    # Text identical, so n_score = 0
    # Total score = 0
    assert clusters[0].cluster_score == 0.0
    
    # Blacklist should now contain [1.0, 0.0, 0.0]
    bl = _load_blacklist()
    assert bl.shape == (1, 3)
    assert np.allclose(bl[0], [1.0, 0.0, 0.0])
    
    # 2. Future run hits the blacklist
    comments_future = [
        Comment(
            comment_id="5",
            author="grace",
            timestamp=3000.0,
            text="looks normal",
            cluster_id=4
        )
    ]
    # Cluster 4 centroid is extremely close to cluster 3's blacklisted centroid
    centroids_future = {4: [0.99, 0.01, 0.0]}
    
    clusters_future = score_clusters(comments_future, centroids_future)
    
    # Grace has normal v_score (100) and n_score (100) -> 100
    # But hits blacklist -> -50 penalty -> 50 final score
    assert clusters_future[0].cluster_score == 50.0
    assert comments_future[0].account_authenticity_score == 50.0

def test_missing_cluster_id():
    comments = [
        Comment(
            comment_id="1",
            author="harry",
            timestamp=1000.0,
            text="No cluster.",
            cluster_id=None
        )
    ]
    centroids = {}
    
    clusters = score_clusters(comments, centroids)
    
    # Comments without cluster_id are not grouped into NarrativeClusters
    assert len(clusters) == 0
    assert comments[0].account_authenticity_score == 100.0
