import os
import random
import time
from typing import List
from openai import OpenAI, RateLimitError

from schema import NarrativeCluster

def _get_openai_client() -> OpenAI:
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ.get("NVIDIA_API_KEY", "dummy-api-key")
    )

def _generate_label_with_retry(comments_texts: List[str], max_retries: int = 5) -> str:
    client = _get_openai_client()
    
    prompt = "Read these 5 comments. Summarize their core shared opinion in 3 to 5 words. Output only the label.\n\nComments:\n"
    for i, text in enumerate(comments_texts, 1):
        prompt += f"{i}. {text}\n"
    
    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="nvidia/nemotron-3.5-lightning-30b-a3b",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))

def decode_labels(clusters: List[NarrativeCluster]) -> List[NarrativeCluster]:
    """
    Generates plain-language labels for each NarrativeCluster.
    Returns the same list of clusters with the `label` string populated.
    """
    for cluster in clusters:
        num_to_select = min(5, len(cluster.comments))
        
        if num_to_select == 0:
            cluster.label = "No comments"
            continue
            
        selected_comments = random.sample(cluster.comments, num_to_select)
        comments_texts = [c.text for c in selected_comments]
        
        label = _generate_label_with_retry(comments_texts)
        cluster.label = label
        
    return clusters
