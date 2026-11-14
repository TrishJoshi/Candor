import pytest
import json
from unittest.mock import patch, Mock
from requests.exceptions import RequestException

from ingest import ingest_url, get_mock_comments, parse_facebook_comment
from schema import Comment, FacebookPost, FacebookComment, Author

def test_get_mock_comments():
    comments = get_mock_comments()
    assert len(comments) == 50
    assert all(isinstance(c, Comment) for c in comments)
    assert comments[0].comment_id == "mock_comment_0"
    assert comments[49].comment_id == "mock_comment_49"

def test_parse_facebook_comment_flat():
    fb_comment = FacebookComment(
        comment_id="c1",
        author=Author(name="Alice"),
        message="Hello world",
        reaction_count=5,
        created_time="2026-11-13T10:19:00+08:00"
    )
    comments = parse_facebook_comment(fb_comment)
    assert len(comments) == 1
    assert comments[0].comment_id == "c1"
    assert comments[0].author == "Alice"
    assert comments[0].text == "Hello world"
    assert comments[0].timestamp > 0

def test_parse_facebook_comment_nested():
    reply = FacebookComment(
        comment_id="r1",
        author=Author(name="Bob"),
        message="Reply to Alice",
        reaction_count=2,
        created_time="2026-11-13T10:25:00+08:00"
    )
    fb_comment = FacebookComment(
        comment_id="c1",
        author=Author(name="Alice"),
        message="Hello world",
        reaction_count=5,
        created_time="2026-11-13T10:19:00+08:00",
        replies=[reply]
    )
    comments = parse_facebook_comment(fb_comment)
    assert len(comments) == 2
    assert comments[0].comment_id == "c1"
    assert comments[1].comment_id == "r1"
    assert comments[1].text == "Reply to Alice"
    assert comments[1].author == "Bob"

@patch("ingest.requests.post")
def test_ingest_url_success(mock_post):
    mock_response = Mock()
    # Provide a minimal valid FacebookPost dict
    mock_response.json.return_value = {
        "post_id": "p1",
        "author": {"name": "PageAdmin"},
        "source": {"type": "page", "id": "PageAdmin", "name": "PageAdmin"},
        "reactions": {"total": 10},
        "comments": [
            {
                "comment_id": "c1",
                "author": {"name": "Alice"},
                "message": "First comment",
                "reaction_count": 0,
                "created_time": "2026-11-13T10:19:00+08:00",
                "replies": []
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    comments = ingest_url("http://example.com/post")
    assert len(comments) == 1
    assert comments[0].comment_id == "c1"
    assert comments[0].text == "First comment"
    
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.context.dev/v1/scrape"
    assert kwargs["json"] == {"url": "http://example.com/post"}

@patch("ingest.requests.post")
def test_ingest_url_request_exception(mock_post):
    mock_post.side_effect = RequestException("Connection error")
    with pytest.raises(RuntimeError, match="Failed to scrape URL"):
        ingest_url("http://example.com/post")

def test_ingest_url_with_sample_json():
    # Integration style test with the actual JSON output file if it exists
    try:
        with open("Facebook_JSON_Output.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        pytest.skip("Facebook_JSON_Output.json not found")
        
    with patch("ingest.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        comments = ingest_url("http://example.com/post")
        # We don't know exact count without counting, but it should be > 0
        assert len(comments) > 0
        assert all(isinstance(c, Comment) for c in comments)
