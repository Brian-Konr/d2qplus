#!/usr/bin/env python
import os
import json
import argparse
import logging
import re
import pandas as pd
import pyterrier as pt
from pyterrier_dr import BGEM3, FlexIndex
from pyterrier.measures import *


if not pt.started():
    pt.init(version='5.11', helper_version='0.0.7')

def sanitize(q):
    q = q.lower()
    q = re.sub(r'[^a-z0-9\s]', ' ', q)
    return " ".join(q.split())

def load_corpus(jsonl_path: str) -> pd.DataFrame:
    """Load JSONL with fields: id, text, predicted_queries."""
    docs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            e = json.loads(line)
            docs.append({
                'docno': e['id'],
                'text': e['text'],
                'pred_queries': e.get('predicted_queries', [])
            })
    return pd.DataFrame(docs)


def build_indexes(df: pd.DataFrame, base_dir: str, batch_size: int, max_length: int):
    """Create 4 on-disk indices: sparse/dense × text/augmented."""
    os.makedirs(base_dir, exist_ok=True)
    # 1. Prepare augmented text
    df['aug_text'] = df['text'] + " " + df['pred_queries'].apply(lambda qs: " ".join(qs) if isinstance(qs, list) else qs)
    
    # 2. Sparse (BM25) indices via IterDictIndexer :contentReference[oaicite:4]{index=4}
    idx_text_dir = os.path.join(base_dir, "text_bm25")
    idx_aug_dir  = os.path.join(base_dir, "aug_bm25")

    # Text-only BM25
    if os.path.isdir(idx_text_dir):
        idx_text = pt.IndexFactory.of(idx_text_dir)
    else:
        itext = pt.IterDictIndexer(idx_text_dir, blocks=False, overwrite=True, meta={'docno': 40}) # meta docno=40 means the _id field is 40 chars
        idx_text = itext.index(df[['docno','text']].to_dict('records'))
    
    # Text+Doc2Query BM25
    if os.path.isdir(idx_aug_dir):
        idx_aug = pt.IndexFactory.of(idx_aug_dir)
    else:
        iaug  = pt.IterDictIndexer(idx_aug_dir, blocks=False, overwrite=True, meta={'docno': 40})
        idx_aug  = iaug.index(df[['docno','aug_text']].rename(columns={'aug_text':'text'}).to_dict('records'))
    
    # 3. Dense (BGE-M3) indices via FlexIndex :contentReference[oaicite:5]{index=5}
    factory = BGEM3(batch_size=batch_size, max_length=max_length, verbose=True, device="cuda")
    doc_enc = factory.doc_encoder()
    idx_td_dir = os.path.join(base_dir, "text_dense")
    idx_ad_dir = os.path.join(base_dir, "aug_dense")

    # Text-only Dense
    if os.path.isdir(idx_td_dir):
        idx_text_dense = FlexIndex(idx_td_dir, verbose=True)
    else:
        idx_td_dir = os.path.join(base_dir, "text_dense")
        idx_text_dense = FlexIndex(idx_td_dir, verbose=True)
        (doc_enc >> idx_text_dense).index(df[['docno','text']].to_dict('records'))
    
    # Text+Doc2Query Dense
    if os.path.isdir(idx_ad_dir):
        idx_aug_dense = FlexIndex(idx_ad_dir, verbose=True)
    else:
        idx_aug_dense  = FlexIndex(idx_ad_dir, verbose=True)
        (doc_enc >> idx_aug_dense).index(df[['docno','aug_text']].rename(columns={'aug_text':'text'}).to_dict('records'))

    return idx_text, idx_aug, idx_text_dense, idx_aug_dense, factory

def run_experiment(idxs, factory, queries, qrels, k: int, batch_size: int, metrics):
    """Build retrievers, run and evaluate."""
    # Sparse BM25 :contentReference[oaicite:6]{index=6}
    bm25_text = pt.BatchRetrieve(idxs[0], wmodel="BM25")
    bm25_aug  = pt.BatchRetrieve(idxs[1], wmodel="BM25")
    # Dense BGE-M3 (HNSW) :contentReference[oaicite:7]{index=7}
    qenc = factory.query_encoder()

    # dense_text = qenc >> idxs[2].faiss_flat_retriever(gpu=True) >> transformer.limit(k)
    # dense_aug = qenc >> idxs[3].faiss_flat_retriever(gpu=True) >> transformer.limit(k)

    dense_text = qenc >> idxs[2].torch_retriever(num_results=k, device="cuda", qbatch=batch_size)    
    dense_aug = qenc >> idxs[3].torch_retriever(num_results=k, device="cuda", qbatch=batch_size) 

    systems = [bm25_text, bm25_aug, dense_text, dense_aug]
    names   = ["BM25_text", "BM25_text+DQ", "Dense_text", "Dense_text+DQ"]
    
    # Experiment: computes Reciprocal Rank, nDCG@10, Recall@10, Recall@100 :contentReference[oaicite:8]{index=8}
    exp = pt.Experiment(
        systems, queries, qrels,
        eval_metrics=metrics,
        names=names,
        filter_by_qrels=True
    )
    return exp

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sparse vs. dense × text vs. text+Doc2Query"
    )
    parser.add_argument("--corpus",    required=True, help="JSONL input")
    parser.add_argument("--queries",    required=True, help="JSONL queries file")
    parser.add_argument("--qrels",     required=True, help="TREC qrels file")
    parser.add_argument("--index-dir", default="indices", help="where to store indices")
    parser.add_argument("--output",    default="results.csv", help="where to write metrics")
    parser.add_argument("--k",         type=int, default=100, help="top-k for retrieval")
    parser.add_argument("--batch-size",type=int, default=64, help="BGE-M3 encoder batch size")
    parser.add_argument("--max-length",type=int, default=1024, help="BGE-M3 max sequence length")
    args = parser.parse_args()


    # Load data
    corpus = load_corpus(args.corpus)
    
    with open(args.queries, 'r') as f:
        queries = [json.loads(line) for line in f]
        queries = pd.DataFrame(queries)
        queries['qid'] = queries['_id']
        queries['query'] = queries['text'].apply(sanitize)
        queries = queries[['qid', 'query']]

    qrels  = pt.io.read_qrels(args.qrels)

    # Build indices
    idx_text, idx_aug, idx_td, idx_ad, factory = build_indexes(
        corpus, args.index_dir, args.batch_size, args.max_length
    )

    # Define metrics
    metrics = ["recip_rank", "ndcg_cut_10", "recall_20", "recall_40", "recall_60", "recall_80", "recall_100"]
    # Run and evaluate
    results = run_experiment(
        (idx_text, idx_aug, idx_td, idx_ad),
        factory, queries, qrels, args.k, args.batch_size, metrics
    )

    # Persist
    print(results)
    results.to_csv(args.output, index=False)
    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()
