from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os

from schema import NarrativeCluster

app = FastAPI(
    title="Project Candor Orchestration",
    description="API for orchestrating the Candor narrative mapping pipeline."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analyze", response_model=List[NarrativeCluster])
def analyze_url(url: str) -> List[NarrativeCluster]:
    """
    Orchestrates the pipeline: Ingestion -> Spatial Engine -> Authenticity Engine -> Decoder.
    Returns the final scored and labeled clusters.
    """
    try:
        # Import locally to allow concurrent development of other tracks
        from ingest import ingest_url
        from spatial_engine import run_spatial_engine
        from scorer import score_clusters
        from decoder import decode_labels
        
        # 1. Ingestion
        comments = ingest_url(url)
        
        # 2. Spatial Engine
        clustered_comments, centroids = run_spatial_engine(comments)
        
        # 3. Authenticity Engine
        scored_clusters = score_clusters(clustered_comments, centroids)
        
        # 4. Decoder
        final_clusters = decode_labels(scored_clusters)
        
        return final_clusters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def serve_index():
    """Serves the frontend index.html file."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="index.html not found")
