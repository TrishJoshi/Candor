from fastapi.testclient import TestClient
from main import app
from schema import NarrativeCluster, Comment

client = TestClient(app)

def main():
    print("🚀 Initializing FastAPI Orchestration Pipeline...")
    print("This will execute GET /analyze by sending test data to test_main.py orchestration.\n")
    
    # We can invoke the pytest for test_main to see the orchestrated mock behavior
    import pytest
    pytest.main(["-v", "test_main.py::test_analyze_url_success"])

if __name__ == "__main__":
    main()
