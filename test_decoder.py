import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError

from schema import Comment, NarrativeCluster
from decoder import decode_labels, _generate_label_with_retry

@pytest.fixture
def mock_clusters():
    c1 = Comment(comment_id="1", author="alice", timestamp=123.0, text="I love this product")
    c2 = Comment(comment_id="2", author="bob", timestamp=124.0, text="This is amazing")
    c3 = Comment(comment_id="3", author="charlie", timestamp=125.0, text="Absolutely fantastic")
    c4 = Comment(comment_id="4", author="dave", timestamp=126.0, text="Best thing ever")
    c5 = Comment(comment_id="5", author="eve", timestamp=127.0, text="I highly recommend it")
    c6 = Comment(comment_id="6", author="frank", timestamp=128.0, text="Will buy again")
    
    cluster1 = NarrativeCluster(cluster_id=1, comments=[c1, c2, c3, c4, c5, c6])
    cluster2 = NarrativeCluster(cluster_id=2, comments=[])
    
    return [cluster1, cluster2]

@patch("decoder._get_openai_client")
def test_decode_labels_success(mock_get_client, mock_clusters):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Positive Product Reviews"
    mock_client.chat.completions.create.return_value = mock_response
    
    result = decode_labels(mock_clusters)
    
    assert len(result) == 2
    assert result[0].label == "Positive Product Reviews"
    assert result[1].label == "No comments"
    
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"

@patch("decoder._get_openai_client")
@patch("time.sleep", return_value=None)
def test_decode_labels_retry_on_error(mock_sleep, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Retried Label"
    
    # Fail twice, then succeed
    mock_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limited", response=MagicMock(), body=MagicMock()),
        Exception("Transient error"),
        mock_response
    ]
    
    c1 = Comment(comment_id="1", author="alice", timestamp=123.0, text="Test comment")
    clusters = [NarrativeCluster(cluster_id=1, comments=[c1])]
    
    result = decode_labels(clusters)
    
    assert result[0].label == "Retried Label"
    assert mock_client.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2

@patch("decoder._get_openai_client")
@patch("time.sleep", return_value=None)
def test_decode_labels_max_retries_exceeded(mock_sleep, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_client.chat.completions.create.side_effect = RateLimitError("Rate limited", response=MagicMock(), body=MagicMock())
    
    c1 = Comment(comment_id="1", author="alice", timestamp=123.0, text="Test comment")
    clusters = [NarrativeCluster(cluster_id=1, comments=[c1])]
    
    with pytest.raises(RateLimitError):
        decode_labels(clusters)
        
    assert mock_client.chat.completions.create.call_count == 5
    assert mock_sleep.call_count == 4
