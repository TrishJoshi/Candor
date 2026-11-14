import random
import matplotlib.pyplot as plt
from schema import Comment
from spatial_engine import run_spatial_engine

def main():
    comments = []
    # 1. Generate Dummy Data
    topics = [
        ("Datacenters are great because they create so many local jobs and boost the economy.", 50),
        ("Datacenters use way too much water and energy! They are destroying our environment.", 50),
        ("AI is the future and we need all the compute power we can get to stay ahead.", 50),
        ("I just had a really good pizza for lunch today.", 50) # Random noise
    ]
    
    cid_counter = 0
    for text_base, count in topics:
        for _ in range(count):
            variations = [" I agree.", " True.", " Yes.", " Exactly.", " Indeed.", " For sure."]
            text = text_base + random.choice(variations)
            comments.append(Comment(
                comment_id=str(cid_counter),
                author=f"user_{cid_counter}",
                account_age_days=10,
                timestamp=1000.0,
                text=text
            ))
            cid_counter += 1
            
    print(f"Generated {len(comments)} dummy comments.")
    
    # 2. Run Spatial Engine locally
    print("Running Local Spatial Engine...")
    updated_comments, centroids = run_spatial_engine(comments)
    
    # 3. Plot with Matplotlib
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    plotted = 0
    for c in updated_comments:
        cluster_id = c.cluster_id
        vec = c.vector
        if vec:
            color = colors[cluster_id % len(colors)] if cluster_id != -1 else 'gray'
            plt.scatter(vec[0], vec[1], color=color, alpha=0.7, edgecolors='none')
            plotted += 1
            
    plt.title("Spatial Engine - Narrative Clusters (Local Execution)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # Save the plot
    plt.savefig("local_plot.png")
    print(f"Successfully plotted {plotted} comments and saved to local_plot.png!")

if __name__ == "__main__":
    main()
