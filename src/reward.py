# reward.py
#!/usr/bin/env python3

import torch
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from collections import Counter


class Reward:
    """
    Base class for all reward modules.
    """
    def evaluate(
        self,
        queries: List[str],
        q_embs: torch.Tensor,
        context: Dict[str, Any]
    ) -> float:
        raise NotImplementedError


class FormatReward(Reward):
    def __init__(self, expected_n: int, weight: float = 0.1):
        self.expected_n = expected_n
        self.intro_pattern = re.compile(r"^\s*(here|these|okay|hi|dear)\b", re.I)
        self.weight = weight

    def evaluate(
        self,
        queries: List[str],
        q_embs: torch.Tensor,
        context: Dict[str, Any]
    ) -> float:
        # Check intro text in first query
        first = queries[0]
        if self.intro_pattern.search(first):
            return 0.0
        # Ensure no empty queries and correct count
        if len(queries) != self.expected_n or any(len(q.strip()) == 0 for q in queries):
            return 0.0
        return 1.0 * self.weight


class TopicCoverageReward(Reward):
    def __init__(
        self,
        topic_vecs_path: str,
        tau: float = 0.35,
        weight: float = 1.0
    ):
        self.topic_vecs = torch.load(topic_vecs_path)
        self.tau = tau
        self.weight = weight

    def evaluate(
        self,
        queries: List[str],
        q_embs: torch.Tensor,
        context: Dict[str, Any]
    ) -> float:
        # context must contain 'topic_ids' and 'topic_weights'
        t_ids = context['topic_ids']
        t_ws = context['topic_weights']
        device = q_embs.device
        t_embs = torch.stack([self.topic_vecs[i] for i in t_ids]).to(device)
        w_embs = torch.tensor(t_ws, dtype=torch.float, device=device)
        sims = t_embs @ q_embs.T  # [K, N]
        covered = sims.max(dim=1).values >= self.tau
        score = (covered.float() * w_embs).sum().item()
        return score * self.weight


class DiversityReward(Reward):
    def __init__(self, weight: float = 0.2):
        self.weight = weight

    def evaluate(
        self,
        queries: List[str],
        q_embs: torch.Tensor,
        context: Dict[str, Any]
    ) -> float:
        n = q_embs.size(0)
        if n < 2:
            return 0.0
        sims = q_embs @ q_embs.T
        sum_sim = sims.triu(diagonal=1).sum()
        num_pairs = n * (n - 1) / 2
        avg_sim = sum_sim / num_pairs
        diversity = (1.0 - avg_sim.item()) * self.weight
        return diversity


class KeywordCoverageReward(Reward):
    def __init__(
        self,
        bm25: BM25Okapi,
        weight: float = 0.4
    ):
        self.idf = bm25.idf
        self.avgdl = bm25.avgdl
        self.k1 = bm25.k1
        self.b = bm25.b
        self.weight = weight

    def evaluate(
        self,
        queries: List[str],
        q_embs: torch.Tensor,
        context: Dict[str, Any]
    ) -> float:
        keywords = context['keywords']
        tokens = " ".join(queries).lower().split()
        tf = Counter(tokens)
        dl = len(tokens)
        kw_tokens = [tok for kw in keywords for tok in kw.lower().split()]
        idf_sum = sum(self.idf.get(tok, 0.0) for tok in kw_tokens) or 1.0
        norm_factor = (1 - self.b + self.b * dl / self.avgdl)

        score = 0.0
        for tok in kw_tokens:
            idf_val = self.idf.get(tok, 0.0)
            freq = tf.get(tok, 0)
            num = freq * (self.k1 + 1)
            den = freq + self.k1 * norm_factor
            score += idf_val * (num / den if den > 0 else 0.0)

        return (score / idf_sum) * self.weight


class CombinedReward:
    """
    Combines multiple Reward modules in one pass, embedding once.
    """
    def __init__(
        self,
        embed_model,
        topic_vecs_path: str,
        doc_vecs_path: str,
        bm25: BM25Okapi,
        expected_n: int,
        tau: float = 0.35,
        weights: Dict[str, float] = None
    ):
        # Initialize reward modules
        w = weights or {}
        self.modules: List[Reward] = [
            FormatReward(expected_n, weight=w.get('format', 0.1)),
            TopicCoverageReward(topic_vecs_path, tau, weight=w.get('coverage', 0.4)),
            DiversityReward(weight=w.get('diversity', 0.2)),
            KeywordCoverageReward(bm25, weight=w.get('keyword', 0.4)),
            RelevanceReward(doc_vectors_path=doc_vecs_path, weight=w.get('relevance', 0.4))
        ]
        self.embed_model = embed_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(
        self,
        completions: List[List[Dict[str, Any]]],
        topic_ids: List[List[int]],
        topic_weights: List[List[float]],
        keywords_list: List[List[str]],
        doc_id: List[str],
        **kwargs
    ) -> List[float]:
        all_scores: List[float] = []
        for comp, t_ids, t_ws, keywords, docid in zip(
            completions, topic_ids, topic_weights, keywords_list, doc_id
        ):
            text = comp[0]['content'].strip()
            queries = [q.strip() for q in text.split("\n") if q.strip()]
            if not queries:
                all_scores.append(0.0)
                continue

            # embed queries once
            q_embs = self.embed_model.encode(
                queries,
                convert_to_tensor=True,
                normalize_embeddings=True
            ).to(self.device)

            context = {
                'topic_ids': t_ids,
                'topic_weights': t_ws,
                'keywords': keywords,
                'doc_id': str(docid)
            }

            # sum module scores
            total = sum(mod.evaluate(queries, q_embs, context) for mod in self.modules)
            all_scores.append(total)

        return all_scores
    

class RelevanceReward:
    """
    Bi-encoder relevance: average cosine similarity between each query embedding and its document embedding.
    Assumes `doc_vectors` is a Tensor[num_docs, dim]. The average similarity is normalized to [0, 1] range and multiplied by a weight.
    """
    def __init__(self, doc_vectors_path: str, weight: float = 0.3):
        self.doc_vectors = torch.load(doc_vectors_path)  # pre-loaded Tensor
        self.weight = weight

    def evaluate(
        self,
        queries: List[str],
        q_embs: torch.Tensor,          # [N, dim], normalized
        context: Dict[str, Any]
    ) -> float:
        # context must contain the integer 'doc_id'
        doc_id = context['doc_id']
        # select the right document embedding and move to the same device
        doc_emb = self.doc_vectors[doc_id].to(q_embs.device)  # [dim]
        # compute cosines: (q_embs @ doc_emb) gives [N]
        sims = q_embs @ doc_emb
        avg_sim = sims.mean().item() if sims.numel() > 0 else 0.0 # avg_sim is in [0, 1]
        return avg_sim * self.weight