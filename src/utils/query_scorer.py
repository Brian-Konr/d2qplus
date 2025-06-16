from typing import List, Dict
import numpy as np
import torch
from collections import Counter
from sacrebleu import BLEU
from rank_bm25 import BM25Okapi
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
    
    @staticmethod
    def distinct_n_score(queries: List[str], n: int = 2) -> float:
        """
        Compute Distinct-n score for a set of queries.
        Distinct-n = number of unique n-grams / total number of n-grams
        
        Args:
            queries: List of query strings
            n: n-gram size (1 for unigrams, 2 for bigrams, etc.)
            
        Returns:
            Distinct-n score (0.0 to 1.0)
        """
        if not queries:
            return 0.0
            
        all_ngrams = []
        for query in queries:
            tokens = query.lower().split()
            if len(tokens) < n:
                continue
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
        
        if not all_ngrams:
            return 0.0
            
        unique_ngrams = set(all_ngrams)
        return len(unique_ngrams) / len(all_ngrams)
    
    @staticmethod
    def type_token_ratio_score(queries: List[str]) -> float:
        """
        Compute Type-Token Ratio (TTR) for a set of queries.
        TTR = number of unique tokens / total number of tokens
        
        Args:
            queries: List of query strings
            
        Returns:
            TTR score (0.0 to 1.0)
        """
        if not queries:
            return 0.0
            
        all_tokens = []
        for query in queries:
            tokens = query.lower().split()
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return 0.0
            
        unique_tokens = set(all_tokens)
        return len(unique_tokens) / len(all_tokens)

def distinct2_novelty_scores(queries: List[str]) -> List[float]:
    """
    For each query, compute:
        novelty_i = (# bigrams unique to query_i) / (total bigrams in query_i)
    """
    # 1) build a global count of all bigrams across queries
    global_counts = Counter()
    all_bigrams = []
    for q in queries:
        toks = q.lower().split()
        bigrams = list(zip(toks, toks[1:]))
        all_bigrams.append(bigrams)
        global_counts.update(bigrams)
    
    # 2) for each query, count how many of its bigrams have global count == 1
    novelty_scores = []
    for bigrams in all_bigrams:
        if not bigrams:
            novelty_scores.append(0.0)
            continue
        unique_cnt = sum(1 for bg in bigrams if global_counts[bg] == 1)
        novelty_scores.append(unique_cnt / len(bigrams))
    return novelty_scores

def filter_top_queries_distinct2(
    queries: List[str],
    k_percent: float = 0.5,
    keywords: List[str] = None
) -> List[str]:
    if not queries:
        return []
    if len(queries) == 1:
        return queries

    # default keywords = all tokens
    if keywords is None:
        keywords = list({tok for q in queries for tok in q.lower().split()})

    # 1) Distinct-2 novelty per query
    novelty_scores = distinct2_novelty_scores(queries)

    # 2) Jaccard coverage per query
    keyword_tokens = set(tok for kw in keywords for tok in kw.lower().split())
    jaccard_scores = []
    for q in queries:
        qtoks = set(q.lower().split())
        if keyword_tokens:
            inter = qtoks & keyword_tokens
            union = qtoks | keyword_tokens
            jaccard_scores.append(len(inter) / len(union))
        else:
            jaccard_scores.append(0.0)

    # 3) Combine and pick top k%
    combined = [
        (0.5 * novelty + 0.5 * jaccard, q)
        for novelty, jaccard, q in zip(novelty_scores, jaccard_scores, queries)
    ]
    combined.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(len(queries) * k_percent))
    return [q for _, q in combined[:k]]

def filter_top_queries(queries: List[str], k_percent: float = 0.5, keywords: List[str] = None) -> List[str]:
    """
    Filter top k% of queries using weighted Jaccard similarity and Self-BLEU scoring.
    
    Args:
        queries: List of query strings
        k_percent: Percentage of top queries to keep (0.0 to 1.0)
        keywords: List of keywords for Jaccard similarity (if None, uses all query tokens)
        
    Returns:
        Filtered list of top k% queries
    """
    if not queries:
        return []
    
    if len(queries) == 1:
        return queries
    
    # If no keywords provided, use all unique tokens from queries as keywords
    if keywords is None:
        keywords = list(set(tok for q in queries for tok in q.lower().split()))
    
    # Calculate individual scores
    self_bleu_score = QueryScorer.self_bleu_diversity_score(queries)
    jaccard_score = QueryScorer.jaccard_keyword_coverage_score(queries, keywords)
    
    # Combined score with equal weights
    combined_score = 0.5 * self_bleu_score + 0.5 * jaccard_score
    
    # Calculate how many queries to keep
    k_count = max(1, int(len(queries) * k_percent))
    
    # For simplicity, we'll rank by individual query scores
    # Calculate per-query diversity and coverage scores
    query_scores = []
    for i, query in enumerate(queries):
        # Self-BLEU for individual query against others
        other_queries = [queries[j] for j in range(len(queries)) if j != i]
        if other_queries:
            from sacrebleu import BLEU
            bleu = BLEU(effective_order=True)
            individual_bleu = bleu.sentence_score(query, other_queries).score / 100.0
            diversity = 1.0 - individual_bleu
        else:
            diversity = 1.0
        
        # Jaccard similarity for individual query
        query_tokens = set(query.lower().split())
        keyword_tokens = set(tok for kw in keywords for tok in kw.lower().split())
        if keyword_tokens:
            intersection = query_tokens & keyword_tokens
            union = query_tokens | keyword_tokens
            jaccard = len(intersection) / len(union) if union else 0.0
        else:
            jaccard = 0.0
        
        # Combined score for this query
        score = 0.5 * diversity + 0.5 * jaccard
        query_scores.append((score, query))
    
    # Sort by score descending and take top k%
    query_scores.sort(key=lambda x: x[0], reverse=True)
    top_queries = [query for _, query in query_scores[:k_count]]
    
    return top_queries


if __name__ == "__main__":
    import json
    gen_q_path = "/home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus/top50_queries_t5.jsonl"
    # save_path = "/home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus/4shot_5perdoc_10beam_50total_1_filtered.jsonl"
    keywords_path = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0606-biobert-mnli-reduce-outlier-for-scoring/keywords.jsonl"

    with open(keywords_path, "r") as f:
        keywords = [json.loads(line) for line in f.readlines()]
        docid2kws = {doc["doc_id"]: doc["keywords"] for doc in keywords}
    
    with open(gen_q_path, "r") as f:
        gen_q = [json.loads(line) for line in f.readlines()]

    distinct1 = []
    distinct2 = []
    ttr = []
    jaccard = []
    for doc in tqdm(gen_q, desc="Computing scores"):
        doc_id = doc["id"]
        pred_queries = doc["predicted_queries"]
        # if pred_queries is a list of list, flatten it
        if isinstance(pred_queries[0], list):
            pred_queries = [q for sublist in pred_queries for q in sublist]
        # Compute distinct-1 and distinct-2 scores
        distinct1_score = QueryScorer.distinct_n_score(pred_queries, n=1)
        distinct2_score = QueryScorer.distinct_n_score(pred_queries, n=2)
        ttr_score = QueryScorer.type_token_ratio_score(pred_queries)
        jaccard_score = QueryScorer.jaccard_keyword_coverage_score(pred_queries, docid2kws.get(doc_id, []))
        
        distinct1.append(distinct1_score)
        distinct2.append(distinct2_score)
        ttr.append(ttr_score)
        jaccard.append(jaccard_score)
        
        # Update the document with scores
        doc["distinct_1"] = distinct1_score
        doc["distinct_2"] = distinct2_score
        doc["ttr"] = ttr_score
        doc["jaccard"] = jaccard_score
    
    print(f"Average Distinct-1 score: {sum(distinct1) / len(distinct1):.4f}")
    print(f"Average Distinct-2 score: {sum(distinct2) / len(distinct2):.4f}")
    print(f"Average Type-Token Ratio score: {sum(ttr) / len(ttr):.4f}")
    print(f"Average Jaccard score: {sum(jaccard) / len(jaccard):.4f}")
    
    # compute distinct-1, distinct-2
    # for doc in tqdm(gen_q, desc="Filtering queries"):
    #     doc_id = doc["id"]
    #     pred_queries = doc["predicted_queries"]
    #     # if pred_queries is a list of list, flatten it
    #     if isinstance(pred_queries[0], list):
    #         pred_queries = [q for sublist in pred_queries for q in sublist]
        
    #     filtered_queries = filter_top_queries_distinct2(pred_queries, k_percent=0.6, keywords=docid2kws[doc_id] if docid2kws.get(doc_id, []) else None)
    #     doc["predicted_queries"] = filtered_queries
    # with open(save_path, "w") as f:
    #     for doc in gen_q:
    #         f.write(json.dumps(doc) + "\n")
    # print(f"Filtered queries saved to {save_path}")