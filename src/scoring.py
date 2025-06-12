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
import json
from sentence_transformers import SentenceTransformer
from utils.data import combine_topic_info


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


def main():
    topic_dir = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0612-01"
    enhanced_topic_info_path = f"{topic_dir}/topic_info_dataframe_enhanced.pkl"
    doc_topics_path = f"{topic_dir}/doc_topics.jsonl"
    core_phrases_path = f"{topic_dir}/keywords.jsonl"
    corpus_path = "/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl"

    enhanced_corpus = combine_topic_info(enhanced_topic_info_path, doc_topics_path, corpus_path, core_phrases_path)
    
    import pandas as pd
    enhanced_topics = pd.read_pickle(enhanced_topic_info_path)
    topic_ids = enhanced_topics["Topic"].tolist()
    enhanced_topics = enhanced_topics["Enhanced_Topic"].tolist() # natural language topic label for each topic

    embed_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", device="cuda")
    embs = embed_model.encode(enhanced_topics, convert_to_tensor=True, normalize_embeddings=True)
    topic_lookup = {int(tid): emb.cpu() for tid, emb in zip(topic_ids, embs)}

    queries_file = "/home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus/with_topic_0612-01_Llama-3.1-8B-Instruct_20250612-202217.jsonl"
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [json.loads(line).get("predicted_queries", [[]]) for line in f if line.strip()]
    
    print(len(queries))
    print(queries[0])

    """
    Two Stage: 
    1. set-level query filtering (對一個 query set 評分，並保留 top K% 的 query set)
    2. query-level: greedy select B queries from the query sets obtained from stage 1 
    """

    ### First Stage: Set-level Query Filtering
    

if __name__ == "__main__":
    main()