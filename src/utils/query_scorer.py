from typing import List, Dict
import numpy as np
import torch
from collections import Counter
from sacrebleu import BLEU
from rank_bm25 import BM25Okapi

class QueryScorer:
    """
    Collection of static methods for scoring query sets.
    Each method takes minimal inputs and returns a scalar score.
    """
    @staticmethod
    def self_bleu_diversity_score(queries: List[str]) -> float:
        """
        Compute diversity using Self-BLEU (1 - average Self-BLEU).
        
        Args:
            queries: List of query strings
            
        Returns:
            Diversity score based on Self-BLEU
        """
        n = len(queries)
        if n < 2:
            return 0.0

        scores = []

        bleu = BLEU(effective_order=True)
        
        for i, hyp in enumerate(queries):
            refs = [queries[j] for j in range(n) if j != i]
            sb = bleu.sentence_score(hyp, refs).score / 100.0
            scores.append(sb)

        avg_sb = float(np.mean(scores))
        return 1.0 - avg_sb
    
    @staticmethod
    def topic_coverage_score(
        queries: List[str], 
        topic_ids: List[int],
        topic_weights: List[float],
        topic_vecs: Dict[int, torch.Tensor],
        embed_model,
        tau: float = 0.6
    ) -> float:
        """
        Compute topic coverage score.
        
        Args:
            queries: List of query strings
            topic_ids: List of topic IDs for this document
            topic_weights: Weights for each topic
            topic_vecs: Tensor containing all topic vectors [num_topics, embed_dim]
            embed_model: Embedding model
            tau: Similarity threshold for coverage
            
        Returns:
            Weighted coverage score
        """
        if not queries:
            return 0.0
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get query embeddings
        q_embs = embed_model.encode(
            queries,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(device)
        
        # Get topic embeddings
        t_embs = torch.stack([topic_vecs[i] for i in topic_ids]).to(device)
        w_embs = torch.tensor(topic_weights, dtype=torch.float, device=device)
        
        # Compute similarities and coverage
        sims = t_embs @ q_embs.T  # [K, N]
        covered = sims.max(dim=1).values >= tau
        score = (covered.float() * w_embs).sum().item()
        
        return score
    
    @staticmethod
    def jaccard_keyword_coverage_score(queries: List[str], keywords: List[str]) -> float:
        """
        Compute keyword coverage using Jaccard similarity.
        
        Args:
            queries: List of query strings
            keywords: List of keywords
            
        Returns:
            Jaccard similarity between query tokens and keyword tokens
        """
        if not keywords:
            return 0.0
            
        # Tokenize and create sets
        query_tokens = {tok for q in queries for tok in q.lower().split()}
        keyword_tokens = {tok for kw in keywords for tok in kw.lower().split()}

        if not keyword_tokens:
            return 0.0

        intersection = query_tokens & keyword_tokens
        union = query_tokens | keyword_tokens

        return len(intersection) / len(union)
    
    @staticmethod
    def bm25_keyword_coverage_score(queries: List[str], keywords: List[str], bm25: BM25Okapi) -> float:
        """
        Compute keyword coverage using BM25 scoring.
        
        Args:
            queries: List of query strings
            keywords: List of keywords
            bm25: Pre-trained BM25 model
            
        Returns:
            Normalized BM25 score for keyword coverage
        """
        if not keywords:
            return 0.0
            
        tokens = " ".join(queries).lower().split()
        tf = Counter(tokens)
        dl = len(tokens)
        kw_tokens = [tok for kw in keywords for tok in kw.lower().split()]
        
        if not kw_tokens:
            return 0.0
            
        idf_sum = sum(bm25.idf.get(tok, 0.0) for tok in kw_tokens) or 1.0
        norm_factor = (1 - bm25.b + bm25.b * dl / bm25.avgdl)

        score = 0.0
        for tok in kw_tokens:
            idf_val = bm25.idf.get(tok, 0.0)
            freq = tf.get(tok, 0)
            num = freq * (bm25.k1 + 1)
            den = freq + bm25.k1 * norm_factor
            score += idf_val * (num / den if den > 0 else 0.0)

        return score / idf_sum
    
    @staticmethod
    def relevance_score(
        queries: List[str], 
        doc_id: str, 
        doc_vectors: Dict[str, torch.Tensor], 
        embed_model,
        aggregate: str = "max", # 'mean' | 'max' | 'topk'
        topk: int = 3
    ) -> float:
        """
        Compute relevance score between queries and document.
        
        Args:
            queries: List of query strings
            doc_id: Document ID
            doc_vectors: Dictionary mapping doc_id to document vectors
            embed_model: Embedding model
            aggregate: Aggregation method for similarity ('mean', 'max', 'topk')  
            topk is used to control how many top similarities to consider
            
        Returns:
            Average similarity between queries and document
        """
        if not queries:
            return 0.0
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get query embeddings
        q_embs = embed_model.encode(
            queries,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(device)
        
        try:
            doc_emb = doc_vectors[doc_id].to(device)
            sims = q_embs @ doc_emb

            if aggregate == "mean":
                return sims.mean().item()
            elif aggregate == "max":
                return sims.max().item()
            elif aggregate == "topk":
                k = min(topk, len(sims))
                return sims.topk(k).values.mean().item()
        except (ValueError, KeyError):
            return 0.0

    def consistency_pass(queries, doc_id):
        """
        考慮 round-trip consistency (BM25 能不能用這個 query 找到 doc_id)
        要先建好 QG expanded corpus 的 BM25
        """
        raise NotImplementedError("Consistency pass scoring not implemented yet.")
