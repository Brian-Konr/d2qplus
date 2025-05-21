# reward.py
#!/usr/bin/env python3

from typing import List
import torch
from sentence_transformers import SentenceTransformer

# — Load precomputed topic centroids & weights —
# Save these tensors during your preprocessing:
#   torch.save(topic_vecs, "topic_vecs.pt")
#   torch.save(topic_ws,   "topic_ws.pt")
topic_vecs = torch.load("topic_vecs.pt")      # shape [K, d]
topic_ws    = torch.load("topic_ws.pt")        # shape [K]
tau         = 0.35

# — Embedder for queries —
embed_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device="cuda"
)

SEP_TOKEN = "<SEP>"

def compute_coverage(
    query_embs: torch.Tensor,
    topic_vecs: torch.Tensor,
    topic_ws: torch.Tensor,
    tau: float = 0.35
) -> float:
    # query_embs: [N, d], topic_vecs: [K, d], topic_ws: [K]
    sims    = topic_vecs @ query_embs.T               # [K, N]
    covered = (sims.max(dim=1).values >= tau)         # [K] bool
    # weighted recall
    return float((covered.float() * topic_ws).sum())

def reward_fn(outs: List[str]) -> List[float]:
    """
    outs: list of generated strings containing N queries separated by SEP_TOKEN
    returns: list of coverage scores
    """
    rewards = []
    for out in outs:
        queries = [q.strip() for q in out.split(SEP_TOKEN) if q.strip()]
        if not queries:
            rewards.append(0.0)
            continue
        # embed all queries in one batch
        embs   = embed_model.encode(queries, convert_to_tensor=True)
        score  = compute_coverage(embs, topic_vecs, topic_ws, tau)
        rewards.append(score)
    return rewards
