Universal Data Contract (The Core Schema)
Before assigning tasks, initialize a shared schema.py (using Pydantic) that all agents will use.

Python
from pydantic import BaseModel
from typing import List, Optional

class Comment(BaseModel):
    comment_id: str
    author: str
    account_age_days: int
    timestamp: float
    text: str
    cluster_id: Optional[int] = None
    account_authenticity_score: Optional[float] = None

class NarrativeCluster(BaseModel):
    cluster_id: int
    label: Optional[str] = None
    cluster_score: Optional[float] = None
    comments: List[Comment]
Track 1: Ingestion Service (Context.dev Track)
Assign to: Agent 1 or Team Member A Agent Objective: Build a robust web scraper that takes a Reddit URL and returns a flat list of Comment objects adhering to the strict schema.

Inputs: A target Reddit URL string.


Outputs: A List[Comment] object (or JSON equivalent).


Implementation Steps:


Write a script (ingest.py) that queries the Context.dev API using their scraping endpoint.


Write a recursive parsing function that flattens the nested Reddit comment tree.


Extract the required metadata. For account_age_days, you may need a secondary Context.dev call to the author's profile page if it's not in the primary payload.


Mocking Requirement: First, generate a hardcoded mock function get_mock_comments() that returns 50 fake Comment objects so Track 2 and 3 can start immediately.


Track 2: The Spatial Engine (Modal Track)
Assign to: Agent 2 or Team Member B Agent Objective: Run an embedding model and clustering algorithm on a Modal GPU container to group text into narratives.

Inputs: List[Comment] (from Track 1 or mock data).


Outputs: The exact same List[Comment], but with the cluster_id field populated.


Implementation Steps:


Create spatial_engine.py and decorate the main function with @modal.function(gpu="T4").


Initialize the all-MiniLM-L6-v2 model using sentence-transformers to generate text embeddings.


Pass the embeddings into UMAP (reduce to 5 dimensions).


Pass the UMAP output into HDBSCAN (set min_cluster_size=3).


Extract the resulting cluster labels, attach them to the respective Comment objects, and return the updated list.


Calculate and return a dictionary of mathematical centroids for each valid cluster ID.


Track 3: Authenticity & Feedback Engine (Core IP)
Assign to: Agent 3 or Team Member C Agent Objective: Apply behavioral heuristics to the clustered comments to detect coordinated swarms, and maintain a stateful memory of astroturfed clusters.

Inputs: List[Comment] (with cluster_id populated by Track 2) and the centroids dictionary.


Outputs: A populated List[NarrativeCluster] with account_authenticity_score and cluster_score calculated.


Implementation Steps:


Create scorer.py. Group the input comments by author.


Velocity Math: Calculate the variance in timestamp for authors with multiple comments.


n-Gram Math: Use sklearn.feature_extraction.text.TfidfVectorizer (char level) to find pairwise cosine similarities between different authors in the same cluster.


Combine these signals into a 0-100 account_authenticity_score.


Aggregate these into a cluster_score using the median author score.


Feedback Loop: If a cluster_score < 30, append its centroid to a local blacklist.npy file. On future runs, deduct 50 points from any account whose comment lands within a tight cosine distance to any blacklisted centroid.


Track 4: Decoder Service (NVIDIA Track)
Assign to: Agent 4 or Team Member D Agent Objective: Prompt an LLM to generate plain-language UI labels for the mathematical clusters.

Inputs: A list of NarrativeCluster objects (with scored comments).


Outputs: The same list, but with the label string populated (e.g., "Anti-Datacenter Environmentalism").


Implementation Steps:


Create decoder.py using the openai Python SDK, but point the base_url to [https://integrate.api.nvidia.com/v1](https://integrate.api.nvidia.com/v1).


Select the top 5 comments per cluster (either randomly, or preferably the ones with embeddings mathematically closest to the centroid).


Send a strict prompt to nvidia/nemotron-3.5-lightning-30b-a3b: "Read these 5 comments. Summarize their core shared opinion in 3 to 5 words. Output only the label."


Implement a retry block with exponential backoff to handle the API's rate limits (40 req/min).


Track 5: Orchestration & Deployment (Hostinger Track)
Assign to: Agent 5 or Team Lead Agent Objective: Wire the modular functions together into a REST API and containerize the application for a VPS.

Inputs: The functions from Tracks 1–4.


Outputs: A live, accessible public URL returning the final JSON payload.


Implementation Steps:


Create main.py using FastAPI.


Define GET /analyze?url=<reddit_url>.


Write the sequential pipeline: comments = ingest(url) -> clustered = spatial_engine(comments) -> scored = scorer(clustered) -> final = decoder(scored).


Write a simple index.html frontend using vanilla JS/HTML to display the JSON output as red/green narrative cards.


Write the Dockerfile. Use the Hostinger Connector (or standard SSH) to deploy the Docker image to the Hostinger VPS.


