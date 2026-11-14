import asyncio
from ingest import get_mock_comments
from spatial_engine import run_spatial_engine
from scorer import score_clusters
from decoder import decode_labels
import json

def main():
    print("🚀 Running Candor Pipeline with mock data...")
    
    # 1. Ingestion (Mock)
    print("\n--- Track 1: Ingestion ---")
    comments = get_mock_comments()
    print(f"✅ Generated {len(comments)} mock comments.")
    print(f"Sample: {comments[0].text}")

    # 2. Spatial Engine
    print("\n--- Track 2: Spatial Engine ---")
    clustered_comments, centroids = run_spatial_engine(comments)
    print(f"✅ Clustered into {len(centroids)} narratives.")
    
    # 3. Authenticity Engine
    print("\n--- Track 3: Authenticity Engine ---")
    scored_clusters = score_clusters(clustered_comments, centroids)
    print(f"✅ Scored {len(scored_clusters)} clusters.")
    if scored_clusters:
        print(f"Sample Cluster Score: {scored_clusters[0].cluster_score}")
    
    # 4. Decoder Service
    print("\n--- Track 4: Decoder Service ---")
    # For local testing without burning real API credits, we'll just mock the label assignment here.
    # If you have an NVIDIA API key exported as NVIDIA_API_KEY, you can uncomment the real call below.
    # final_clusters = decode_labels(scored_clusters)
    for cluster in scored_clusters:
        cluster.label = "Mocked LLM Narrative Label"
    print("✅ Assigned plain-language labels.")

    # Summary
    print("\n--- Final Output ---")
    output = []
    for c in scored_clusters:
        output.append({
            "cluster_id": c.cluster_id,
            "label": c.label,
            "cluster_score": c.cluster_score,
            "comment_count": len(c.comments)
        })
    print(json.dumps(output, indent=2))
    print("\n🎉 Pipeline execution complete!")

if __name__ == "__main__":
    main()
