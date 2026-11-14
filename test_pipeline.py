from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

client = TestClient(app)

@patch("main.ingest.ingest_url") if "ingest_url" in globals() else patch("ingest.ingest_url") # Handle different ways it might be patched
@patch("spatial_engine.run_spatial_engine")
@patch("decoder._get_openai_client")
def run_test(mock_openai, mock_spatial, mock_ingest):
    print("🚀 Initializing FastAPI Test Client...")
    
    # 1. Mock Ingestion to return fake comments
    from ingest import get_mock_comments
    mock_comments = get_mock_comments()
    mock_ingest.return_value = mock_comments
    print(f"✅ Mocked Track 1 (Ingestion) to return {len(mock_comments)} comments.")
    
    # 2. Mock Spatial Engine to simulate Modal GPU grouping
    for i, c in enumerate(mock_comments):
        c.cluster_id = i % 3 # distribute into 3 clusters
    mock_spatial.return_value = (mock_comments, {0: [0.1, 0.1], 1: [0.2, 0.2], 2: [0.3, 0.3]})
    print("✅ Mocked Track 2 (Spatial Engine) to avoid Modal GPU invocation.")
    
    # 3. Track 3 (Scorer) runs normally natively!
    print("✅ Track 3 (Authenticity Engine) will execute native numpy calculations.")
    
    # 4. Mock OpenAI client to avoid hitting rate limits
    class MockMessage: content = "Mocked LLM Summary"
    class MockChoice: message = MockMessage()
    class MockResponse: choices = [MockChoice()]
    class MockCompletions:
        def create(self, **kwargs): return MockResponse()
    class MockChat: completions = MockCompletions()
    class MockOpenAI: chat = MockChat()
    
    mock_openai.return_value = MockOpenAI()
    print("✅ Mocked Track 4 (Decoder) NVIDIA API calls.")
    
    # Run the pipeline!
    print("\nExecuting: GET /analyze?url=http://reddit.com/r/test")
    response = client.get("/analyze?url=http://reddit.com/r/test")
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- Final Pipeline Output ---")
        import json
        print(json.dumps(data, indent=2))
        print("\n🎉 Pipeline executed successfully end-to-end!")
    else:
        print(f"❌ Failed with status {response.status_code}: {response.text}")

if __name__ == "__main__":
    run_test()
