#!/usr/bin/env python3
"""
Simplified scoring functions for query filtering.
These functions take simple inputs and return scalar scores.
"""

import torch
import re
import numpy as np
from typing import List, Dict, Any, Optional
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

        bleu = BLEU(effective_order=True)
        scores = []
        
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
        topic_vecs: torch.Tensor,
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
        if not queries or not topic_ids:
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
    def relevance_score(queries: List[str], doc_id: str, doc_vectors: Dict[str, torch.Tensor], embed_model) -> float:
        """
        Compute relevance score between queries and document.
        
        Args:
            queries: List of query strings
            doc_id: Document ID
            doc_vectors: Dictionary mapping doc_id to document vectors
            embed_model: Embedding model
            
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
            avg_sim = sims.mean().item() if sims.numel() > 0 else 0.0
            return avg_sim
        except (ValueError, KeyError):
            return 0.0


class QueryFilterScorer:
    """
    High-level scorer that combines multiple scoring functions for filtering.
    """
    
    def __init__(
        self,
        embed_model,
        topic_vecs: Optional[torch.Tensor] = None,
        doc_vectors: Optional[Dict[str, torch.Tensor]] = None,
        bm25: Optional[BM25Okapi] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the scorer with models and weights.
        
        Args:
            embed_model: Embedding model
            topic_vecs: Topic vectors tensor
            doc_vectors: Document vectors dictionary
            bm25: BM25 model for keyword scoring
            weights: Weights for different scoring components
        """
        self.embed_model = embed_model
        self.topic_vecs = topic_vecs
        self.doc_vectors = doc_vectors
        self.bm25 = bm25
        
        # Default weights
        default_weights = {
            'format': 0.1,
            'diversity': 0.2,
            'topic_coverage': 0.4,
            'keyword_coverage': 0.4,
            'relevance': 0.3
        }
        self.weights = weights or default_weights
    
    def score_queries(
        self,
        queries: List[str],
        expected_n: Optional[int] = None,
        topic_ids: Optional[List[int]] = None,
        topic_weights: Optional[List[float]] = None,
        keywords: Optional[List[str]] = None,
        doc_id: Optional[str] = None,
        components: Optional[List[str]] = None
    ) -> float:
        """
        Compute overall score for a set of queries.
        
        Args:
            queries: List of query strings
            expected_n: Expected number of queries (for format check)
            topic_ids: Topic IDs for coverage calculation
            topic_weights: Topic weights for coverage calculation
            keywords: Keywords for coverage calculation
            doc_id: Document ID for relevance calculation
            components: List of components to include in scoring
            
        Returns:
            Weighted overall score
        """
        if not queries:
            return 0.0
            
        # Default to all components if not specified
        if components is None:
            components = ['diversity', 'keyword_coverage']
            if expected_n is not None:
                components.append('format')
            if topic_ids is not None and self.topic_vecs is not None:
                components.append('topic_coverage')
            if doc_id is not None and self.doc_vectors is not None:
                components.append('relevance')
        
        total_score = 0.0
        total_weight = 0.0
        
        # Topic coverage score
        if 'topic_coverage' in components and topic_ids and self.topic_vecs is not None:
            if topic_weights is None:
                topic_weights = [1.0] * len(topic_ids)
            score = QueryScorer.topic_coverage_score(
                queries, topic_ids, topic_weights, self.topic_vecs, self.embed_model
            )
            weight = self.weights.get('topic_coverage', 0.4)
            total_score += score * weight
            total_weight += weight
        
        # Keyword coverage score
        if 'keyword_coverage' in components and keywords:
            if self.bm25 is not None:
                score = QueryScorer.bm25_keyword_coverage_score(queries, keywords, self.bm25)
            else:
                score = QueryScorer.jaccard_keyword_coverage_score(queries, keywords)
            weight = self.weights.get('keyword_coverage', 0.4)
            total_score += score * weight
            total_weight += weight
        
        # Relevance score
        if 'relevance' in components and doc_id and self.doc_vectors is not None:
            score = QueryScorer.relevance_score(queries, doc_id, self.doc_vectors, self.embed_model)
            weight = self.weights.get('relevance', 0.3)
            total_score += score * weight
            total_weight += weight
        
        # Return normalized score
        return total_score / total_weight if total_weight > 0 else 0.0
