import sys
from unittest.mock import MagicMock, patch

def passthrough_decorator_factory(*args, **kwargs):
    def decorator(func_or_cls):
        return func_or_cls
    return decorator

mock_modal = MagicMock()
mock_modal.method = passthrough_decorator_factory
mock_modal.enter = passthrough_decorator_factory
mock_modal.local_entrypoint = passthrough_decorator_factory

# Also mock App and its cls method
mock_app = MagicMock()
mock_app.cls = passthrough_decorator_factory
mock_modal.App.return_value = mock_app

sys.modules['modal'] = mock_modal

import pytest
import numpy as np
from schema import Comment
from spatial_engine import run_spatial_engine, SpatialEngine

@pytest.fixture
def mock_comments():
    return [
        Comment(
            comment_id="1",
            author="Alice",
            timestamp=123456789.0,
            text="This is a test comment about something."
        ),
        Comment(
            comment_id="2",
            author="Bob",
            timestamp=123456790.0,
            text="Another test comment discussing things."
        )
    ]

@patch("spatial_engine.SpatialEngine")
def test_run_spatial_engine(mock_engine_cls, mock_comments):
    # Setup mock remote execution
    mock_engine_instance = MagicMock()
    mock_engine_cls.return_value = mock_engine_instance
    
    # Mock the return value of remote run
    mock_result = {
        "comments": [
            {
                "comment_id": "1",
                "author": "Alice",
                "account_age_days": 0,
                "timestamp": 123456789.0,
                "text": "This is a test comment about something.",
                "cluster_id": 0,
                "account_authenticity_score": None
            },
            {
                "comment_id": "2",
                "author": "Bob",
                "account_age_days": 0,
                "timestamp": 123456790.0,
                "text": "Another test comment discussing things.",
                "cluster_id": 0,
                "account_authenticity_score": None
            }
        ],
        "centroids": {
            0: [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    }
    mock_engine_instance.run.remote.return_value = mock_result
    
    # Execute the function
    updated_comments, centroids = run_spatial_engine(mock_comments)
    
    # Verify the results
    assert len(updated_comments) == 2
    assert updated_comments[0].cluster_id == 0
    assert updated_comments[1].cluster_id == 0
    assert updated_comments[0].comment_id == "1"
    assert updated_comments[1].comment_id == "2"
    
    assert isinstance(centroids, dict)
    assert 0 in centroids
    assert centroids[0] == [0.1, 0.2, 0.3, 0.4, 0.5]

    # Verify that remote was called correctly
    mock_engine_instance.run.remote.assert_called_once()
    args, kwargs = mock_engine_instance.run.remote.call_args
    assert len(args[0]) == 2
    assert args[0][0]["comment_id"] == "1"

def test_spatial_engine_class_methods():
    with patch.dict(sys.modules, {
        "sentence_transformers": MagicMock(),
        "umap": MagicMock(),
        "hdbscan": MagicMock(),
        "bertopic": MagicMock()
    }):
        import sentence_transformers
        import umap
        import hdbscan
        import bertopic
        
        # Setup mock BERTopic instance
        mock_topic_model = MagicMock()
        bertopic.BERTopic.return_value = mock_topic_model
        
        # Mock fit_transform to return topics (and probabilities which we ignore)
        mock_topic_model.fit_transform.return_value = ([0, 0, -1], None)
        
        # Mock umap embedding
        mock_umap_model = MagicMock()
        mock_umap_model.embedding_ = np.array([
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3, 0.3, 0.3],
            [0.9, 0.9, 0.9, 0.9, 0.9]  # Noise point, shouldn't affect centroid
        ])
        mock_topic_model.umap_model = mock_umap_model
        
        # Initialize engine and load
        engine = SpatialEngine()
        engine.load()
        
        # Prepare inputs
        inputs = [
            {"comment_id": "1", "author": "Alice", "text": "text1", "account_age_days": 0, "timestamp": 0.0},
            {"comment_id": "2", "author": "Bob", "text": "text2", "account_age_days": 0, "timestamp": 0.0},
            {"comment_id": "3", "author": "Charlie", "text": "text3", "account_age_days": 0, "timestamp": 0.0}
        ]
        
        # Run
        result = engine.run(inputs)
        
        # Check outputs
        assert "comments" in result
        assert "centroids" in result
        
        assert result["comments"][0]["cluster_id"] == 0
        assert result["comments"][1]["cluster_id"] == 0
        assert result["comments"][2]["cluster_id"] == -1
        
        assert 0 in result["centroids"]
        assert -1 not in result["centroids"]  # Noise cluster shouldn't have centroid
        assert len(result["centroids"][0]) == 5
        # (0.1+0.3)/2 = 0.2
        assert np.isclose(result["centroids"][0][0], 0.2)
        assert np.isclose(result["centroids"][0][1], 0.2)
        assert np.isclose(result["centroids"][0][2], 0.2)
        assert np.isclose(result["centroids"][0][3], 0.2)
        assert np.isclose(result["centroids"][0][4], 0.2)


def test_run_spatial_engine_empty():
    # Execute the function with empty comments
    mock_engine_instance = MagicMock()
    mock_engine_instance.run.remote.return_value = {"comments": [], "centroids": {}}
    
    with patch("spatial_engine.SpatialEngine") as mock_engine_cls:
        mock_engine_cls.return_value = mock_engine_instance
        updated_comments, centroids = run_spatial_engine([])
        
        assert updated_comments == []
        assert centroids == {}
