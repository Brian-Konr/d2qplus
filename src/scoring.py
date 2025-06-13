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
from tqdm import tqdm
from utils.query_scorer import QueryScorer
import torch
from typing import List, Dict, Set

def simple_tokenize(text: str) -> Set[str]:
    """
    Simple whitespace tokenizer that returns a set of tokens.
    """
    return set(re.findall(r'\b\w+\b', text.lower()))  # basic word tokenization

class QuerySelector:
    def __init__(
        self,
        embed_model,
        topic_vecs: Dict[int, torch.Tensor],  # Changed type hint
        topic_ids: List[int],
        topic_weights: List[float],
        keywords: List[str],
        softmax_tau: float = 0.07,
        lambda_tc: float = 1.0,      # weight for topic coverage
        topic_coverage_metric: str = "f1",  # 'precision' | 'recall' | 'f1'
        lambda_kw: float = 1.0,      # weight for keyword coverage
        lambda_div: float = 1.3,     # weight for diversity
        lambda_relevance: float = 1.0,  # weight for relevance
        similarity_threshold: float = 0.7,  # threshold for topic similarity
    ):
        self.embed_model  = embed_model
        self.topic_vecs   = topic_vecs          # Dict mapping topic_id to embedding
        self.topic_ids    = topic_ids
        self.topic_weights= topic_weights
        self.keywords     = keywords
        self.lambda_tc    = lambda_tc
        self.lambda_kw    = lambda_kw
        self.lambda_div   = lambda_div
        self.lambda_relevance = lambda_relevance
        self.softmax_tau  = softmax_tau
        self.topic_coverage_metric = topic_coverage_metric
        self.similarity_threshold = similarity_threshold

        # 先處理關鍵詞 tokens
        self.kw_tokens: Set[str] = {
            tok for kw in keywords for tok in simple_tokenize(kw)
        }

    # ----- Topic mask helper -----
    def _topic_mask(self, queries: List[str]) -> torch.BoolTensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Handle case where there are no topics for this document
        if not self.topic_ids:
            return torch.zeros(0, dtype=torch.bool, device=device)
        
        if not queries:
            return torch.zeros(len(self.topic_ids), dtype=torch.bool, device=device)
            
        q_embs = self.embed_model.encode(
            queries, convert_to_tensor=True, normalize_embeddings=True
        ).to(device)
        
        # Get topic embeddings from the dictionary
        t_embs = torch.stack([self.topic_vecs[tid].to(device) for tid in self.topic_ids])
        sims = t_embs @ q_embs.T                 # [K, |Q|]
        return sims.max(dim=1).values >= self.similarity_threshold
    
    def _topic_softmax_assignment(self, queries: List[str], softmax_tau: float = 0.07) -> List[int]:
        """
        perform softmax assignment of queries to topics based on similarity.
        """ 
        if not queries:
            return []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_embs = self.embed_model.encode(
            queries, convert_to_tensor=True, normalize_embeddings=True
        ).to(device)
        centroids = torch.stack([self.topic_vecs[tid] for tid in self.topic_ids])
        centroids = centroids.to(device)
        centroids = torch.nn.functional.normalize(centroids, dim=1)

        sims = centroids @ q_embs.T  # [num_queries, num_topics]
        softmax_scores = torch.softmax(sims / softmax_tau, dim=1)  # [num_queries, num_topics]
        assignments = torch.argmax(softmax_scores, dim=1).tolist()  # Assign each query to the most probable topic
        return assignments

    def _measure_topic_coverage(self, query_topics: List[int], weighted=True) -> Dict[str, float]:
        """
        Measure topic coverage based on the selected query topics. Return precision, recall, and F1 score.
        """
        if not query_topics:
            return 0.0
        Q = set(query_topics)
        D = set(self.topic_ids)
        W = dict(zip(self.topic_ids, self.topic_weights))

        if weighted:
            # Weighted coverage
            covered = sum(W[tid] for tid in Q if tid in D)
            total_weight = sum(W[tid] for tid in D)
            nQ = len(Q)

            P_w = covered / nQ # weighted-precision
            R_w = covered / total_weight if total_weight > 0 else 0.0 # weighted-recall
            F_w = (2 * P_w * R_w) / (P_w + R_w) if (P_w + R_w) > 0 else 0.0
            return {"precision": P_w, "recall": R_w, "f1": F_w}
        
        P = len(Q & D) / len(Q) if Q else 0.0  # precision
        R = len(Q & D) / len(D) if D else 0.0  # recall
        F = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0  # F1 score
        return {"precision": P, "recall": R, "f1": F}
    
    def compute_single_queryset_score(self, queries: List[str]) -> Dict[str, Any]:
        """
        Compute scores for a single query set.
        Args:
            queries: List of queries in the set.
        Returns:
            Dictionary containing scores:
                - "topic_coverage": topic coverage score
                - "diversity": diversity score
                - "kw_coverage": keyword coverage score
                - "relevance": relevance score
                - "agg_score": aggregated score based on weights
        """
        query_topics = self._topic_softmax_assignment(queries)
        topic_cov_metrics = self._measure_topic_coverage(query_topics) # {"precision": P, "recall": R, "f1": F}
        topic_coverage = topic_cov_metrics[self.topic_coverage_metric]

        # word use diversity using self-BLEU
        diversity = QueryScorer.self_bleu_diversity_score(queries)

        # keyword coverage
        kw_coverage = QueryScorer.jaccard_keyword_coverage_score(queries, self.keywords)
        
        # relevance score
        relevance = QueryScorer.relevance_score(queries, self.embed_model, self.topic_vecs, self.topic_ids, tau=self.similarity_threshold)
        
        # aggregate score
        agg_score = (
            self.lambda_tc * topic_coverage +
            self.lambda_kw * kw_coverage +
            self.lambda_div * diversity +
            self.lambda_relevance * relevance
        )
        
        return {
            "topic_coverage": topic_coverage,
            "diversity": diversity,
            "kw_coverage": kw_coverage,
            "relevance": relevance,
            "agg_score": agg_score,
        }

def main():
    
    # Set environment variables to avoid tokenizer warnings
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("🚀 Starting greedy query selection...")
    
    topic_dir = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0612-01"
    enhanced_topic_info_path = f"{topic_dir}/topic_info_dataframe_enhanced.pkl"
    doc_topics_path = f"{topic_dir}/doc_topics.jsonl"
    core_phrases_path = f"{topic_dir}/keywords.jsonl"
    corpus_path = "/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl"

    print("📂 Loading enhanced corpus...")
    enhanced_corpus = combine_topic_info(enhanced_topic_info_path, doc_topics_path, corpus_path, core_phrases_path)
    print(f"✅ Loaded {len(enhanced_corpus)} documents")
    
    import pandas as pd
    print("📊 Loading topic information...")
    enhanced_topics = pd.read_pickle(enhanced_topic_info_path)
    topic_ids = enhanced_topics["Topic"].tolist()
    enhanced_topics = enhanced_topics["Enhanced_Topic"].tolist() # natural language topic label for each topic
    print(f"✅ Loaded {len(topic_ids)} topics")

    print("🤖 Loading embedding model...")
    embed_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", device="cuda")
    print("🔢 Encoding topic embeddings...")
    embs = embed_model.encode(enhanced_topics, convert_to_tensor=True, normalize_embeddings=True)
    topic_lookup = {int(tid): emb.cpu() for tid, emb in zip(topic_ids, embs)}
    print("✅ Topic embeddings ready")

    queries_file = "/home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus/promptagator_Llama-3.1-8B-Instruct_20q.jsonl"
    print("📋 Loading candidate queries...")
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]
        docid2queries = {(d["id"]): d["predicted_queries"] for d in queries}
    print(f"✅ Loaded queries for {len(docid2queries)} documents")
    
    print("🎯 Starting greedy selection for each document...")
    
    
    new_queries = []
    scored_queries = []

    B = 6 # number of query set to keep
    for doc in enhanced_corpus:
        doc_id = doc["doc_id"]
        selector = QuerySelector(
            embed_model=embed_model,
            topic_vecs=topic_lookup,
            topic_ids=topic_ids,
            topic_weights=doc["topic_weights"],
            keywords=doc["keywords"],
            softmax_tau=0.07,
            lambda_tc=1.0,  # weight for topic coverage
            topic_coverage_metric="f1",  # 'precision' | 'recall' | 'f1'
            lambda_kw=1.0,  # weight for keyword coverage
            lambda_div=1.0,  # weight for diversity
            lambda_relevance=1.0,  # weight for relevance
            similarity_threshold=0.7,  # threshold for topic similarity
        )
        print(f"Processing document {doc_id} with {len(docid2queries[doc_id])} candidate queries...")
        query_sets = docid2queries[doc_id]

        scores = []
        for query_set in tqdm(query_sets, desc=f"Scoring queries for {doc_id}"):
            score = selector.compute_single_queryset_score(query_set)
            scores.append((query_set, score))
        
        # Sort queries by aggregated score
        scores.sort(key=lambda x: x[1]["agg_score"], reverse=True)

        scored_queries.append({
            "doc_id": doc_id,
            "title": doc["title"],
            "text": doc["text"],
            "scores": scores,  # Store all scores for analysis
        })
        new_queries.append({
            "doc_id": doc_id,
            "title": doc["title"],
            "text": doc["text"],
            "queries": [q for q, score in scores[:B]],  # Keep top B query set
        })
    
    output_file = "/home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus/greed_select_10.jsonl"
    print(f"💾 Saving results to {output_file}")
    with open(output_file, "w") as f:
        for item in new_queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("🎉 Greedy selection completed successfully!")


if __name__ == "__main__":
    main()