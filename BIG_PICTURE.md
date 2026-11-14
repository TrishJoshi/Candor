### SYSTEM CONTEXT: PROJECT "CANDOR" (Junction Hackathon 2026)

**The Grand Vision**
We are building a system to visualize how online narratives emerge, grow, evolve, split, and decay across platforms like Reddit. The goal is to detect Coordinated Inauthentic Behavior (CIB) and astroturfing.

**The Technical Philosophy (No LLM Wrappers)**
Instead of checking IP addresses or mouse clicks, we analyze the actual content and temporal spread of specific talking points. We are completely bypassing the standard "LLM Wrapper" approach. We do not pass raw text to an LLM and ask "Is this a bot?" Instead, we map the discussion into a high-dimensional embedding space and cluster these embeddings to find distinct "memes" or narratives. We then mathematically evaluate the behavioral coordination of the accounts inside those specific clusters.

**The Logic & Core IP**

1. **Narrative Buckets:** Extract distinct opinions and narratives from comment sections using vector embeddings, UMAP, and HDBSCAN.


2. **Authenticity Scoring:** We score individual accounts based on age, recent activity spikes, posting volume, and topical hyper-focus, alongside temporal burstiness and n-gram stylistic overlaps.


3. **Author-Level Aggregation:** We roll account scores up into a final narrative score to determine if a specific talking point is organic or a coordinated swarm.
4. **The Causal Feedback Loop:** If a high percentage (e.g., 80%) of accounts pushing a narrative are inauthentic, the narrative is flagged as astroturfing. Other accounts pushing the same narrative then receive an increased inauthenticity score. Strict thresholds are required to prevent a runaway loop, ensuring real users who simply fall for propaganda are not mislabeled as bots.



**The Architecture (How the Modules Connect)**
You are working on a highly modular backend. The pipeline executes strictly in this order:

1. **Ingestion (Context.dev API):** A Reddit URL is scraped into a flat list of `Comment` objects.


2. **Spatial Engine (Modal GPUs):** Comments are embedded (`all-MiniLM-L6-v2`), reduced (UMAP), and clustered (HDBSCAN) to identify distinct narratives.
3. **Authenticity Engine (Core IP):** Behavioral math (Scikit-learn, Numpy) is applied to the clusters to generate an Authenticity Score for the accounts and the narrative.
4. **Decoder (NVIDIA Nemotron API):** The mathematical center of each cluster is sent to an LLM to generate a plain-language UI label (e.g., "Anti-Datacenter Environmentalism").


5. **Orchestration (FastAPI & Hostinger):** The modules are chained together into a REST API and deployed live on a VPS using Docker.



**Your Directive**
As a coding agent, you will be assigned to build one specific module of this pipeline. You must strictly adhere to the shared `schema.py` (using Pydantic) for all data passing in and out of your module. Do not attempt to build the entire pipeline—focus solely on making your assigned module fast, mathematically rigorous, and perfectly typed.