#!/usr/bin/env python
import os
import json
import argparse
import logging
import re
from typing import Optional
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
                'orig_id': e['id'], # keep original id for remapping later
                'text': e['text'],
                'pred_queries': e.get('predicted_queries', [])
            })
    df = pd.DataFrame(docs)
    df['docno'] = df.index.astype(str)
    return df


def build_indexes(df: pd.DataFrame, base_dir: str, index_name: str, batch_size: int, max_length: int, overwrite_text_dense_index: bool, overwrite_aug_dense_index: bool):
    """Create 4 on-disk indices: sparse/dense × text/augmented."""
    os.makedirs(base_dir, exist_ok=True)
    if index_name:
        aug_base_dir = os.path.join(base_dir, index_name)
        os.makedirs(aug_base_dir, exist_ok=True)

    # 1. Prepare augmented text
    df['aug_text'] = df['text'] + " " + df['pred_queries'].apply(lambda qs: " ".join(qs) if isinstance(qs, list) else qs)
    
    # 2. Sparse (BM25) indices via IterDictIndexer :contentReference[oaicite:4]{index=4}
    idx_text_dir = os.path.join(base_dir, "text_bm25")
    idx_aug_dir  = os.path.join(aug_base_dir, "aug_bm25")

    # Text-only BM25
    itext = pt.IterDictIndexer(idx_text_dir, blocks=False, overwrite=True)
    idx_text = itext.index(df[['docno','text']].to_dict('records'))
    
    # Text+Doc2Query BM25
    iaug  = pt.IterDictIndexer(idx_aug_dir, blocks=False, overwrite=True)
    idx_aug  = iaug.index(df[['docno','aug_text']].rename(columns={'aug_text':'text'}).to_dict('records'))
    
    # 3. Dense (BGE-M3) indices via FlexIndex :contentReference[oaicite:5]{index=5}
    factory = BGEM3(batch_size=batch_size, max_length=max_length, verbose=True, device="cuda")
    doc_enc = factory.doc_encoder()
    idx_td_dir = os.path.join(base_dir, "text_dense")
    idx_ad_dir = os.path.join(aug_base_dir, "aug_dense")

    # Text-only Dense
    idx_text_dense = FlexIndex(idx_td_dir, verbose=True)
    if overwrite_text_dense_index:
        print(f"Building text dense index at {idx_td_dir}")
        text_dense_indexer = idx_text_dense.indexer(mode="overwrite")
        (doc_enc >> text_dense_indexer).index(df[['docno','text']].to_dict('records'))    
        print("Text dense index built successfully.")
    
    # Text+Doc2Query Dense
    idx_aug_dense = FlexIndex(idx_ad_dir, verbose=True)
    if overwrite_aug_dense_index:
        print(f"Building augmented dense index at {idx_ad_dir}")
        aug_dense_indexer = idx_aug_dense.indexer(mode="overwrite")
        (doc_enc >> aug_dense_indexer).index(df[['docno','aug_text']].rename(columns={'aug_text':'text'}).to_dict('records'))
        print("Augmented dense index built successfully.")

    return idx_text, idx_aug, idx_text_dense, idx_aug_dense, factory

def run_experiment(idxs, factory, queries, qrels, k: int, batch_size: int, metrics):
    """Build retrievers, run and evaluate."""
    # Sparse BM25 :contentReference[oaicite:6]{index=6}
    bm25_text = pt.BatchRetrieve(idxs[0], wmodel="BM25")
    bm25_aug  = pt.BatchRetrieve(idxs[1], wmodel="BM25")
    # Dense BGE-M3 (HNSW) :contentReference[oaicite:7]{index=7}
    qenc = factory.query_encoder()

    dense_text = qenc >> idxs[2].torch_retriever(num_results=k, device="cuda", qbatch=batch_size)    
    dense_aug = qenc >> idxs[3].torch_retriever(num_results=k, device="cuda", qbatch=batch_size) 

    systems = [bm25_text, bm25_aug, dense_text, dense_aug]
    names   = ["BM25_text", "BM25_text+DQ", "Dense_text", "Dense_text+DQ"]
    
    exp = pt.Experiment(
        systems, queries, qrels,
        eval_metrics=[m['metric'] for m in metrics],
        round={m['metric']: m['round'] for m in metrics},
        names=names,
        filter_by_qrels=True
    )
    return exp

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sparse vs. dense × text vs. text+Doc2Query"
    )
    parser.add_argument("--corpus",    required=True, help="JSONL input for generated corpus file with fields: id, text, predicted_queries")
    parser.add_argument("--queries",    required=True, help="JSONL queries file")
    parser.add_argument("--qrels",     required=True, help="TREC qrels file")

    parser.add_argument("--index-base-dir", default="indices", help="where to store indices")
    parser.add_argument("--overwrite-text-dense-index", action='store_true', default=False, help="Overwrite existing dense indices")
    parser.add_argument("--overwrite-aug-dense-index", action='store_true', default=False, help="Overwrite existing dense indices")
    parser.add_argument("--index-name", default="", help="Name of the directory to store indices, if not specified, will use index_dir")

    parser.add_argument("--output",    default="results.csv", help="where to write metrics")

    parser.add_argument("--k",         type=int, default=300, help="top-k for retrieval")
    parser.add_argument("--batch-size",type=int, default=64, help="BGE-M3 encoder batch size")
    parser.add_argument("--max-length",type=int, default=2048, help="BGE-M3 max sequence length")
    args = parser.parse_args()


    # Load data
    corpus = load_corpus(args.corpus)

    id_map = dict(zip(corpus['orig_id'], corpus['docno']))
    
    with open(args.queries, 'r') as f:
        queries = [json.loads(line) for line in f]
        queries = pd.DataFrame(queries)
        queries['qid'] = queries['_id']
        queries['query'] = queries['text'].apply(sanitize)
        queries = queries[['qid', 'query']]

    qrels  = pt.io.read_qrels(args.qrels)
    qrels['docno'] = qrels['docno'].map(id_map)

    # Build indices
    idx_text, idx_aug, idx_td, idx_ad, factory = build_indexes(
        df=corpus,
        base_dir=args.index_base_dir, 
        index_name=args.index_name, 
        batch_size=args.batch_size, 
        max_length=args.max_length,
        overwrite_text_dense_index=args.overwrite_text_dense_index,
        overwrite_aug_dense_index=args.overwrite_aug_dense_index, 
    )

    # Define metrics
    metrics = [
        {"metric": "recip_rank", "round": 4}, # MRR
        {"metric": "map", "round": 4},
        {"metric": "ndcg_cut_10", "round": 4},
        {"metric": "ndcg_cut_20", "round": 4},
        {"metric": "recall_50", "round": 4},
        {"metric": "recall_100", "round": 4},
        {"metric": "recall_150", "round": 4},
        {"metric": "recall_200", "round": 4},
        {"metric": "recall_250", "round": 4},
        {"metric": "recall_300", "round": 4}
    ]
    # Run and evaluate
    results = run_experiment(
        (idx_text, idx_aug, idx_td, idx_ad),
        factory, queries, qrels, args.k, args.batch_size, metrics
    )

    # if args.output dir does not exist, create it
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(results)
    results.to_csv(args.output, index=False)
    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()
