# reward.py
#!/usr/bin/env python3

from typing import Any, Dict, List
import torch, torch.nn as nn

class CoverageRewardModel(nn.Module):
    def __init__(self, topic_vecs, embed_model, tau=0.35):
        super().__init__()
        self.topic_vecs = topic_vecs
        self.embed_model = embed_model
        self.tau = tau

    def compute_coverage(self,
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
    
    def forward(self, samples: Dict[str, Any], responses: List[str]) -> List[float]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rewards = []
        for topic_ids, topic_weights, resp in zip(samples["topic_ids"], samples["topic_weights"], responses):
            t_embs = torch.stack([self.topic_vecs[t] for t in topic_ids]).to(device)
            t_weights = torch.tensor(topic_weights, dtype=torch.float).to(device)

            queries = [q.strip() for q in resp.split("\n") if q.strip()]
            if not queries:
                rewards.append(0.0); continue
            
            q_embs = self.embed_model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
            score = self.compute_coverage(q_embs, t_embs, t_weights, tau)
            rewards.append(score)
        return rewards