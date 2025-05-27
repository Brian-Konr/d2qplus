# reward_v2.py
#!/usr/bin/env python3

import torch
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from collections import Counter


class SharedComputationCache:
    """
    Shared cache for expensive computations like embeddings.
    """
    def __init__(self):
        self.cache = {}
        
    def get_or_compute(self, key: str, compute_fn):
        if key not in self.cache:
            self.cache[key] = compute_fn()
        return self.cache[key]
    
    def clear(self):
        self.cache.clear()


class BaseReward:
    """
    Base class for all reward modules with shared computation support.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.__name__ = self.__class__.__name__  # For wandb logging
        
    def __call__(
        self,
        completions: List[List[Dict[str, Any]]],
        topic_ids: List[List[int]],
        topic_weights: List[List[float]],
        keywords_list: List[List[str]],
        doc_id: List[str],
        shared_cache: Optional[SharedComputationCache] = None,
        **kwargs
    ) -> List[float]:
        """
        Main entry point that handles batch processing and caching.
        """
        if shared_cache is None:
            shared_cache = SharedComputationCache()
            
        all_scores = []
        
        for i, (comp, t_ids, t_ws, keywords, docid) in enumerate(zip(
            completions, topic_ids, topic_weights, keywords_list, doc_id
        )):
            # Parse queries from completion
            text = comp[0]['content'].strip()
            queries = [q.strip() for q in text.split("\n") if q.strip()]
            
            if not queries:
                all_scores.append(0.0)
                continue
                
            # Get or compute query embeddings (shared across all rewards)
            cache_key = f"q_embs_{i}"
            q_embs = shared_cache.get_or_compute(
                cache_key, 
                lambda: self._get_embeddings(queries)
            )
            
            context = {
                'topic_ids': t_ids,
                'topic_weights': t_ws,
                'keywords': keywords,
                'doc_id': str(docid),
                'queries': queries,
                'q_embs': q_embs,
                'shared_cache': shared_cache,
                'sample_idx': i
            }
            
            score = self.evaluate(context)
            all_scores.append(score * self.weight)
            
        return all_scores
    
    def _get_embeddings(self, queries: List[str]) -> torch.Tensor:
        """Override this in subclasses that need embeddings."""
        return None
        
    def evaluate(self, context: Dict[str, Any]) -> float:
        """Override this method in subclasses."""
        raise NotImplementedError


class FormatReward(BaseReward):
    def __init__(self, expected_n: int, weight: float = 0.1):
        super().__init__(weight)
        self.expected_n = expected_n
        self.intro_pattern = re.compile(r"^\s*(here|these|okay|hi|dear)\b", re.I)

    def evaluate(self, context: Dict[str, Any]) -> float:
        queries = context['queries']
        
        if not queries:
            return 0.0
            
        # Check intro text in first query
        if self.intro_pattern.search(queries[0]):
            return 0.0
            
        # Ensure no empty queries and correct count
        if len(queries) != self.expected_n or any(len(q.strip()) == 0 for q in queries):
            return 0.0
            
        return 1.0


class TopicCoverageReward(BaseReward):
    def __init__(self, embed_model, topic_vecs_path: str, tau: float = 0.35, weight: float = 1.0):
        super().__init__(weight)
        self.embed_model = embed_model
        self.topic_vecs = torch.load(topic_vecs_path)
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_embeddings(self, queries: List[str]) -> torch.Tensor:
        return self.embed_model.encode(
            queries,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(self.device)

    def evaluate(self, context: Dict[str, Any]) -> float:
        q_embs = context['q_embs']
        t_ids = context['topic_ids']
        t_ws = context['topic_weights']
        
        if q_embs is None or not t_ids:
            return 0.0
            
        t_embs = torch.stack([self.topic_vecs[i] for i in t_ids]).to(q_embs.device)
        w_embs = torch.tensor(t_ws, dtype=torch.float, device=q_embs.device)
        
        sims = t_embs @ q_embs.T  # [K, N]
        covered = sims.max(dim=1).values >= self.tau
        score = (covered.float() * w_embs).sum().item()
        
        return score


class DiversityReward(BaseReward):
    def __init__(self, embed_model, weight: float = 0.2):
        super().__init__(weight)
        self.embed_model = embed_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_embeddings(self, queries: List[str]) -> torch.Tensor:
        return self.embed_model.encode(
            queries,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(self.device)

    def evaluate(self, context: Dict[str, Any]) -> float:
        q_embs = context['q_embs']
        
        if q_embs is None:
            return 0.0
            
        n = q_embs.size(0)
        if n < 2:
            return 0.0
            
        sims = q_embs @ q_embs.T
        sum_sim = sims.triu(diagonal=1).sum()
        num_pairs = n * (n - 1) / 2
        avg_sim = sum_sim / num_pairs
        diversity = 1.0 - avg_sim.item()
        
        return diversity


class KeywordCoverageReward(BaseReward):
    def __init__(self, bm25: BM25Okapi, weight: float = 0.4):
        super().__init__(weight)
        self.idf = bm25.idf
        self.avgdl = bm25.avgdl
        self.k1 = bm25.k1
        self.b = bm25.b

    def evaluate(self, context: Dict[str, Any]) -> float:
        queries = context['queries']
        keywords = context['keywords']
        
        tokens = " ".join(queries).lower().split()
        tf = Counter(tokens)
        dl = len(tokens)
        kw_tokens = [tok for kw in keywords for tok in kw.lower().split()]
        
        if not kw_tokens:
            return 0.0
            
        idf_sum = sum(self.idf.get(tok, 0.0) for tok in kw_tokens) or 1.0
        norm_factor = (1 - self.b + self.b * dl / self.avgdl)

        score = 0.0
        for tok in kw_tokens:
            idf_val = self.idf.get(tok, 0.0)
            freq = tf.get(tok, 0)
            num = freq * (self.k1 + 1)
            den = freq + self.k1 * norm_factor
            score += idf_val * (num / den if den > 0 else 0.0)

        return score / idf_sum


class RelevanceReward(BaseReward):
    def __init__(self, embed_model, doc_vectors_path: str, weight: float = 0.3):
        super().__init__(weight)
        self.embed_model = embed_model
        self.doc_vectors = torch.load(doc_vectors_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_embeddings(self, queries: List[str]) -> torch.Tensor:
        return self.embed_model.encode(
            queries,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(self.device)

    def evaluate(self, context: Dict[str, Any]) -> float:
        q_embs = context['q_embs']
        doc_id = context['doc_id']
        
        if q_embs is None:
            return 0.0
            
        try:
            doc_emb = self.doc_vectors[doc_id].to(q_embs.device)
            sims = q_embs @ doc_emb
            avg_sim = sims.mean().item() if sims.numel() > 0 else 0.0
            return avg_sim
        except (ValueError, IndexError):
            return 0.0


def create_reward_functions(
    embed_model,
    topic_vecs_path: str,
    doc_vecs_path: str,
    bm25: BM25Okapi,
    expected_n: int,
    tau: float = 0.35,
    weights: Dict[str, float] = None
) -> List[BaseReward]:
    """
    Factory function to create all reward functions.
    """
    w = weights or {}
    
    return [
        FormatReward(expected_n, weight=w.get('format', 0.1)),
        TopicCoverageReward(embed_model, topic_vecs_path, tau, weight=w.get('coverage', 0.4)),
        DiversityReward(embed_model, weight=w.get('diversity', 0.2)),
        KeywordCoverageReward(bm25, weight=w.get('keyword', 0.4)),
        RelevanceReward(embed_model, doc_vecs_path, weight=w.get('relevance', 0.4))
    ]
