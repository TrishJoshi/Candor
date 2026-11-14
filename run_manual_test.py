import sys
from unittest.mock import MagicMock

# Modal is throwing an incompatibility error with Python 3.14+ in our environment.
# Since we just want to test the orchestration flow manually without executing the GPU models natively right now,
# we will mock out the modal dependency globally before importing the pipeline.
sys.modules["modal"] = MagicMock()
sys.modules["modal"].function = lambda **kwargs: lambda f: f

from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch
import json

client = TestClient(app)

# We'll intercept the network calls (Track 1 and Track 4) so we don't hit real APIs during a manual test
@patch("ingest.requests.post")
@patch("decoder._get_openai_client")
def run_test(mock_openai, mock_requests_post):
    print("🚀 Running the Candor Pipeline (FastAPI Orchestration)...\n")
    
    # 1. Mock the Context.dev response for Ingestion (Track 1)
    print("📡 Mocking Context.dev scraper response...")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "post_id": "test_post",
        "author": {"name": "Host"},
        "source": {"type": "page", "id": "1", "name": "Host"},
        "reactions": {"total": 0},
        "comments": [
            {
                "comment_id": "c1", "author": {"name": "Alice"}, "message": "I love datacenters!", 
                "reaction_count": 0, "created_time": "2026-11-13T10:00:00+08:00", "replies": []
            },
            {
                "comment_id": "c2", "author": {"name": "Alice"}, "message": "More datacenters!", 
                "reaction_count": 0, "created_time": "2026-11-13T10:00:01+08:00", "replies": []
            },
            {
                "comment_id": "c3", "author": {"name": "Bob"}, "message": "Datacenters are bad.", 
                "reaction_count": 0, "created_time": "2026-11-13T10:00:00+08:00", "replies": []
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None
    mock_requests_post.return_value = mock_resp

    # 2. Mock OpenAI/NVIDIA API (Track 4)
    print("🤖 Mocking NVIDIA Nemotron LLM labels...")
    class MockMessage: content = "Mocked LLM Label"
    class MockChoice: message = MockMessage()
    class MockResponse: choices = [MockChoice()]
    class MockCompletions:
        def create(self, **kwargs): return MockResponse()
    class MockChat: completions = MockCompletions()
    class MockOpenAI: chat = MockChat()
    mock_openai.return_value = MockOpenAI()

    # We will also mock run_spatial_engine because we don't have sentence-transformers downloaded locally and modal is disabled
    with patch("main.run_spatial_engine") as mock_spatial:
        from schema import Comment
        # Force the 3 comments into two clusters
        clustered = [
            Comment(comment_id="c1", author="Alice", text="I love datacenters!", timestamp=1.0, account_age_days=0, cluster_id=1),
            Comment(comment_id="c2", author="Alice", text="More datacenters!", timestamp=2.0, account_age_days=0, cluster_id=1),
            Comment(comment_id="c3", author="Bob", text="Datacenters are bad.", timestamp=1.0, account_age_days=0, cluster_id=2)
        ]
        centroids = {1: [0.1, 0.1], 2: [0.9, 0.9]}
        mock_spatial.return_value = (clustered, centroids)
        
        print("\n▶️ Executing: GET /analyze?url=http://reddit.com/r/datacenter")
        response = client.get("/analyze?url=http://reddit.com/r/datacenter")
        
        if response.status_code == 200:
            print("\n✅ Success! Pipeline returned:\n")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed: {response.text}")

if __name__ == "__main__":
    run_test()
