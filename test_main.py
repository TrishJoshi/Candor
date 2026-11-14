from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import pytest

from main import app
from schema import NarrativeCluster, Comment
client = TestClient(app)

@patch("ingest.ingest_url")
@patch("spatial_engine.run_spatial_engine")
@patch("scorer.score_clusters")
@patch("decoder.decode_labels")
def test_analyze_url_success(mock_decode_labels, mock_score_clusters, mock_run_spatial_engine, mock_ingest_url):
    """Test successful orchestration of the pipeline."""
    # Setup mock data
    mock_comments = [Comment(comment_id="1", author="alice", timestamp=1.0, text="hello")]
    mock_clustered = [Comment(comment_id="1", author="alice", timestamp=1.0, text="hello", cluster_id=1)]
    mock_centroids = {1: [0.1, 0.2]}
    mock_scored = [NarrativeCluster(cluster_id=1, comments=mock_clustered, cluster_score=95.0)]
    mock_final = [NarrativeCluster(cluster_id=1, comments=mock_clustered, cluster_score=95.0, label="Greetings")]

    # Setup mock returns
    mock_ingest_url.return_value = mock_comments
    mock_run_spatial_engine.return_value = (mock_clustered, mock_centroids)
    mock_score_clusters.return_value = mock_scored
    mock_decode_labels.return_value = mock_final

    # Execute
    url = "http://reddit.com/r/test"
    response = client.get(f"/analyze?url={url}")
    
    # Assert HTTP response
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["cluster_id"] == 1
    assert data[0]["label"] == "Greetings"

    # Assert orchestration calls happen in correct sequence and wiring
    mock_ingest_url.assert_called_once_with(url)
    mock_run_spatial_engine.assert_called_once_with(mock_comments)
    mock_score_clusters.assert_called_once_with(mock_clustered, mock_centroids)
    mock_decode_labels.assert_called_once_with(mock_scored)

@patch("ingest.ingest_url")
def test_analyze_url_ingest_failure(mock_ingest_url):
    """Test error handling when ingestion fails."""
    mock_ingest_url.side_effect = Exception("Scraping failed due to rate limit")
    
    response = client.get("/analyze?url=http://example.com")
    assert response.status_code == 500
    assert response.json()["detail"] == "Scraping failed due to rate limit"

def test_index_route():
    """Test serving index.html."""
    response = client.get("/")
    assert response.status_code in [200, 404]
