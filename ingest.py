import requests
from typing import List
from datetime import datetime
from schema import Comment, FacebookPost, FacebookComment

def parse_facebook_comment(fb_comment: FacebookComment) -> List[Comment]:
    """Recursively flattens a nested Facebook comment tree into a list of normalized Comments."""
    comments = []
    
    # Parse timestamp
    timestamp = 0.0
    if fb_comment.created_time:
        try:
            dt = datetime.fromisoformat(fb_comment.created_time)
            timestamp = dt.timestamp()
        except ValueError:
            pass
            
    # Create normalized comment
    comment = Comment(
        comment_id=fb_comment.comment_id,
        author=fb_comment.author.name if fb_comment.author else "Unknown",
        account_age_days=0,  # Would require secondary API call to populate
        timestamp=timestamp,
        text=fb_comment.message or ""
    )
    comments.append(comment)
    
    # Recursively parse replies
    for reply in fb_comment.replies:
        comments.extend(parse_facebook_comment(reply))
        
    return comments

def ingest_url(url: str) -> List[Comment]:
    """Scrapes URL and returns a flat list of Comments."""
    api_url = "https://api.context.dev/v1/scrape"
    payload = {"url": url}
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to scrape URL {url}: {e}")
        
    fb_post = FacebookPost(**data)
    
    all_comments = []
    for fb_comment in fb_post.comments:
        all_comments.extend(parse_facebook_comment(fb_comment))
        
    return all_comments

def get_mock_comments() -> List[Comment]:
    """Returns 50 fake Comment objects so Track 2 and 3 can start immediately."""
    comments = []
    for i in range(50):
        comments.append(
            Comment(
                comment_id=f"mock_comment_{i}",
                author=f"mock_author_{i % 10}",
                account_age_days=100 + i,
                timestamp=1700000000.0 + i * 3600,
                text=f"This is mock comment {i} about the topic.",
            )
        )
    return comments
