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


# greedy_selector.py  (drop this next to your scorer module)
import torch
from typing import List, Dict, Set
from copy import deepcopy

def simple_tokenize(text: str) -> Set[str]:
    """
    Simple whitespace tokenizer that returns a set of tokens.
    """
    return set(re.findall(r'\b\w+\b', text.lower()))  # basic word tokenization

# greedy_selector_v2.py
import torch, math
from sacrebleu import BLEU
from typing import List, Dict, Set
from copy import deepcopy


_bleu = BLEU(effective_order=True)            # 全域共用，省時間

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")   # 只保留字母/數字

def simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

def self_bleu_against_set(q: str, ref_set: List[str]) -> float:
    """Self-BLEU of q vs. existing set (0~1). If set 空則回 0."""
    if not ref_set:
        return 0.0
    score = _bleu.sentence_score(q, ref_set).score / 100.0
    return score  # 高 = 重複

class GreedyQuerySelector:
    def __init__(
        self,
        embed_model,
        topic_vecs: Dict[int, torch.Tensor],  # Changed type hint
        topic_ids: List[int],
        topic_weights: List[float],
        keywords: List[str],
        tau: float = 0.6,
        lambda_tc: float = 1.0,
        lambda_kw: float = 1.0,
        lambda_div: float = 0.3,     # 多樣性懲罰係數
    ):
        self.embed_model  = embed_model
        self.topic_vecs   = topic_vecs          # Dict mapping topic_id to embedding
        self.topic_ids    = topic_ids
        self.topic_weights= topic_weights
        self.keywords     = keywords
        self.tau          = tau
        self.lambda_tc    = lambda_tc
        self.lambda_kw    = lambda_kw
        self.lambda_div   = lambda_div

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
        return sims.max(dim=1).values >= self.tau

    # ----- Greedy selection -----
    def select(self, candidates: List[str], B: int) -> List[str]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selected, remaining = [], deepcopy(candidates)
        cur_topic_mask = self._topic_mask([])        # Will handle empty topic_ids
        cur_kw_tokens  = set()                       # 已覆蓋 KW
        cur_jaccard    = 0.0

        # Add progress bar for greedy selection if there are many candidates
        pbar = tqdm(total=min(B, len(candidates)), desc="Selecting queries", leave=False, disable=len(candidates) < 50)
        
        while remaining and len(selected) < B:
            best_q, best_gain = None, -1e9

            for q in remaining:
                # --- ΔTopic coverage ---
                if self.topic_ids:  # Only compute topic coverage if topics exist
                    q_tmask   = self._topic_mask([q])
                    delta_tc  = ((~cur_topic_mask) & q_tmask).float() \
                                * torch.tensor(self.topic_weights, device=device)
                    delta_tc  = delta_tc.sum().item()     # scalar
                else:
                    delta_tc = 0.0

                # --- ΔJaccard keyword coverage ---
                q_tokens  = set(simple_tokenize(q)) & self.kw_tokens
                new_union = cur_kw_tokens | q_tokens
                new_inter = new_union & self.kw_tokens
                new_jacc  = len(new_inter) / len(new_union) if new_union else 0.0
                delta_jac = new_jacc - cur_jaccard

                # --- Diversity penalty via Self-BLEU ---
                redun = self_bleu_against_set(q, selected)  # 0~1

                score = ( self.lambda_tc  * delta_tc +
                          self.lambda_kw  * delta_jac -
                          self.lambda_div * redun )

                if score > best_gain:
                    best_q, best_gain  = q, score
                    if self.topic_ids:
                        best_tmask = q_tmask
                    best_tokens = q_tokens
                    best_new_jacc = new_jacc
                    best_redun    = redun

            if best_q is None or best_gain <= 0:
                break   # 沒有正增益就停

            # commit
            selected.append(best_q)
            remaining.remove(best_q)
            if self.topic_ids:
                cur_topic_mask |= best_tmask
            cur_kw_tokens  |= best_tokens
            cur_jaccard     = best_new_jacc
            
            pbar.update(1)

        pbar.close()
        return selected


def process_document_batch(args):
    """
    Process a batch of documents for concurrent execution using ProcessPoolExecutor.
    Args is a tuple to work with ProcessPoolExecutor.
    
    Args:
        args: Tuple containing (doc_batch, model_name, topic_lookup, docid2queries, 
              tau, lambda_tc, lambda_kw, lambda_div, B, batch_id)
        
    Returns:
        List of results for this batch
    """
    import os
    import torch
    (doc_batch, model_name, topic_lookup, docid2queries, 
     tau, lambda_tc, lambda_kw, lambda_div, B, batch_id) = args
    
    # Clear CUDA cache and set device for this process
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Use CPU for multiprocessing to avoid CUDA conflicts
        device = "cpu"
    else:
        device = "cpu"
    
    # Initialize embedding model in each process
    embed_model = SentenceTransformer(model_name, device=device)
    
    results = []
    
    # Create progress bar for this batch
    pbar = tqdm(doc_batch, desc=f"Process-{batch_id:02d}", position=batch_id, leave=True)
    
    for doc in pbar:
        doc_id = doc["doc_id"]
        cand_queries = docid2queries.get(doc_id, [])
        
        # Update progress bar description with current doc
        pbar.set_postfix(
            doc_id=doc_id[:10], 
            queries=len(cand_queries) if cand_queries else 0,
            pid=os.getpid()
        )
        
        if not cand_queries:
            results.append({
                "id": doc["doc_id"],
                "predicted_queries": []
            })
            continue
            
        selector = GreedyQuerySelector(
            embed_model     = embed_model,
            topic_vecs      = topic_lookup,
            topic_ids       = [topic['topic_id'] for topic in doc["topics"]],
            topic_weights   = [topic['weight'] for topic in doc["topics"]],
            keywords        = [kw['phrase'] for kw in doc['core_phrases']],
            tau             = tau,
            lambda_tc       = lambda_tc,
            lambda_kw       = lambda_kw,
            lambda_div      = lambda_div
        )
        
        selected_queries = selector.select(cand_queries, B=B)
        results.append({
            "id": doc["doc_id"],
            "predicted_queries": selected_queries
        })
    
    pbar.close()
    return results


def create_document_batches(documents, batch_size=None, num_workers=None):
    """
    Split documents into batches for concurrent processing.
    
    Args:
        documents: List of documents
        batch_size: Size of each batch (if None, auto-calculate based on workers)
        num_workers: Number of worker processes (if None, use CPU count)
        
    Returns:
        List of document batches
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid too many processes
        
    if batch_size is None:
        batch_size = max(1, len(documents) // (num_workers * 2))  # 2 batches per worker
        
    batches = []
    for i in range(0, len(documents), batch_size):
        batches.append(documents[i:i + batch_size])
        
    return batches, num_workers


def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
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
    
    # Create document batches for concurrent processing
    batches, num_workers = create_document_batches(enhanced_corpus, num_workers=16)
    print(f"🔄 Processing {len(enhanced_corpus)} documents in {len(batches)} batches using {num_workers} workers")
    
    new_queries = []
    
    # Use ProcessPoolExecutor for CPU-intensive tasks to bypass GIL
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        
        # Prepare arguments for each batch
        batch_args = []
        for i, batch in enumerate(batches):
            args = (
                batch, 
                model_name, 
                topic_lookup, 
                docid2queries,
                0.8,    # tau
                0.5,    # lambda_tc
                0.5,    # lambda_kw
                0.5,    # lambda_div
                10,     # B
                i       # batch_id
            )
            batch_args.append(args)
        
        # Submit all batches
        futures = [executor.submit(process_document_batch, args) for args in batch_args]
        
        print(f"\n🔥 Started {len(futures)} processes, each showing progress...")
        print("=" * 80)
        
        # Collect results with progress bar
        completed_batches = 0
        for future in as_completed(futures):
            batch_results = future.result()
            new_queries.extend(batch_results)
            completed_batches += 1
            print(f"\n✅ Completed batch {completed_batches}/{len(futures)}")
        
        print("\n" + "=" * 80)
    
    output_file = "/home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus/greed_select_10.jsonl"
    print(f"💾 Saving results to {output_file}")
    with open(output_file, "w") as f:
        for item in new_queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("🎉 Greedy selection completed successfully!")


if __name__ == "__main__":
    main()