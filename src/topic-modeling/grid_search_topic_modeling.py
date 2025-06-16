#!/usr/bin/env python3
"""

Run BERTopic on a JSONL corpus with either:
  • sentence-level chunks  (chunk_mode="sentence")
  • sliding paragraph windows (chunk_mode="window")
Outputs a JSONL file mapping each doc_id to its topic distribution.
"""

import argparse, json, collections, random, os, sys
import itertools
import nltk; nltk.download("punkt", quiet=True)
from nltk import sent_tokenize
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer

random.seed(42)


def sliding_windows(sentences, size=4, step=2):
    """Yield overlapping sentence windows."""
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + size]).strip()
        if chunk:
            yield chunk
        i += step


def chunk_document(text, mode="sentence", win_size=4, win_step=2):
    """
    Break `text` into chunks according to `mode`.
    Returns a list of string chunks.
    """
    sents = sent_tokenize(text)
    if mode == "sentence":
        return [s.strip() for s in sents if s.strip()]
    elif mode == "window":
        return list(sliding_windows(sents, size=win_size, step=win_step))
    else:
        raise ValueError(f"Unknown chunk_mode: {mode}")


def run_topic_modeling(
    corpus_path,
    base_output_dir, 
    chunk_mode, 
    win_size, 
    win_step, 
    embed_model, 
    embed_device, 
    n_neighbors,
    n_components,
    min_topic_size,
    args # Additional BERTopic parameters
):
    """
    Fit BERTopic on the corpus and write per-document topic distributions.
    """
    random.seed(42)

    # Build chunks
    chunks, chunk_owner = [], []
    doc2chunk_idx = collections.defaultdict(list)

    if not os.path.isfile(corpus_path):
        print(f"Error: '{corpus_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    corpus_topics_out_path = os.path.join(base_output_dir, "doc_topics.jsonl")
    topic_info_dataframe_out = os.path.join(base_output_dir, "topic_info_dataframe.pkl")
    topic_info_csv_out = os.path.join(base_output_dir, "topic_info_dataframe.csv")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id, text = obj.get("_id"), obj.get("text", "")
            if doc_id is None or not isinstance(text, str):
                continue
            doc_chunks = chunk_document(text, mode=chunk_mode, win_size=win_size, win_step=win_step)
            for ch in doc_chunks:
                idx = len(chunks)
                chunks.append(ch)
                chunk_owner.append(doc_id)
                doc2chunk_idx[doc_id].append(idx)

    print(f"[{chunk_mode.upper()}] Built {len(chunks):,} chunks from {len(doc2chunk_idx):,} documents.")

    # Initialize embedding model (ensure device is correct)
    try:
        embedder = SentenceTransformer(embed_model, device=embed_device)
    except Exception as e:
        print(f"Error loading embedder '{embed_model}' on '{embed_device}': {e}", file=sys.stderr)
        sys.exit(1)

    pos_patterns = [
        [{'POS': 'ADJ'},  {'POS': 'NOUN'}],   # adjective-noun, e.g. "gene expression"
        [{'POS': 'NOUN'}],                    # single noun,  e.g. "oncogene"
        [{'POS': 'PROPN'}]                    # proper noun  e.g. "CRISPR"
    ]

    pos = PartOfSpeech("en_core_sci_sm", pos_patterns=pos_patterns, top_n_words=200)
    keybert = KeyBERTInspired(nr_candidate_words=100, nr_repr_docs=5, top_n_words=25)
    mmr = MaximalMarginalRelevance(diversity=0.6, top_n_words=args.top_n_words)

    representation_chain = [pos, keybert, mmr]

    # Initialize BERTopic with grid search parameters
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=UMAP(n_neighbors=n_neighbors, n_components=n_components),
        min_topic_size=min_topic_size,
        representation_model=representation_chain,
        verbose=True
    )

    # Fit and transform
    topics, probs = topic_model.fit_transform(chunks)

    os.makedirs(os.path.dirname(corpus_topics_out_path) or ".", exist_ok=True)
    with open(corpus_topics_out_path, "w", encoding="utf-8") as fout:
        for doc_id, idx_list in doc2chunk_idx.items():
            freq = collections.Counter(topics[i] for i in idx_list if topics[i] != -1)
            total = sum(freq.values())
            topic_entries = []
            for tid, cnt in freq.items():
                topic_entries.append({"topic_id": int(tid), "weight": round(cnt / total, 4)})
            fout.write(json.dumps({"doc_id": doc_id, "topics": topic_entries}, ensure_ascii=False) + "\n")

    print(f"Wrote document-topic distributions to '{corpus_topics_out_path}'")

    # Save topic model information
    topic_df = topic_model.get_topic_info()
    os.makedirs(os.path.dirname(topic_info_dataframe_out) or ".", exist_ok=True)
    topic_df.to_csv(topic_info_csv_out, index=False)
    print(f"Wrote topic model info to '{topic_info_csv_out}'")
    topic_df.to_pickle(topic_info_dataframe_out)
    print(f"Wrote topic model info to '{topic_info_dataframe_out}'")

    # Save topic centroids as embeddings
    import torch
    centroids = torch.tensor(topic_model.topic_embeddings_, dtype=torch.float32)
    centroids = centroids[1:]  # Remove outlier topic (topic -1)
    
    # Save centroids to .pt file
    centroids_path = os.path.join(base_output_dir, "topic_centroids.pt")
    torch.save(centroids, centroids_path)
    print(f"Saved topic centroids to '{centroids_path}'")

    # Optionally save the BERTopic model
    if args.save_topic_model:
        topic_model_out = os.path.join(base_output_dir, "bertopic_model")
        os.makedirs(os.path.dirname(topic_model_out) or ".", exist_ok=True)
        topic_model.save(topic_model_out)
        print(f"Saved BERTopic model to '{topic_model_out}'")

    return topic_model


def run_grid_search(
    corpus_path,
    base_output_dir,
    chunk_mode,
    win_size,
    win_step,
    embed_model,
    embed_device,
    n_neighbors_list,
    n_components_list,
    min_topic_size_list,
    args
):
    """
    Run grid search over UMAP and BERTopic parameters.
    """
    print(f"Starting grid search with:")
    print(f"  n_neighbors: {n_neighbors_list}")
    print(f"  n_components: {n_components_list}")
    print(f"  min_topic_size: {min_topic_size_list}")
    print(f"  Total combinations: {len(n_neighbors_list) * len(n_components_list) * len(min_topic_size_list)}")
    
    results = []
    
    for i, (n_neighbors, n_components, min_topic_size) in enumerate(
        itertools.product(n_neighbors_list, n_components_list, min_topic_size_list)
    ):
        print(f"\n{'='*60}")
        print(f"Running configuration {i+1}: n_neighbors={n_neighbors}, n_components={n_components}, min_topic_size={min_topic_size}")
        print(f"{'='*60}")
        
        # Create output directory for this configuration
        config_dir = f"neighbors_{n_neighbors}_components_{n_components}_mintopic_{min_topic_size}"
        output_dir = os.path.join(base_output_dir, config_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Run topic modeling for this configuration
            topic_model = run_topic_modeling(
                corpus_path=corpus_path,
                base_output_dir=output_dir,
                chunk_mode=chunk_mode,
                win_size=win_size,
                win_step=win_step,
                embed_model=embed_model,
                embed_device=embed_device,
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_topic_size=min_topic_size,
                args=args
            )
            
            # Save configuration parameters
            config_params = {
                "n_neighbors": n_neighbors,
                "n_components": n_components,
                "min_topic_size": min_topic_size,
                "chunk_mode": chunk_mode,
                "win_size": win_size,
                "win_step": win_step,
                "embed_model": embed_model,
                "embed_device": embed_device,
                "top_n_words": args.top_n_words,
                "save_topic_model": args.save_topic_model
            }
            
            config_path = os.path.join(output_dir, "grid_search_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_params, f, indent=2, ensure_ascii=False)
            
            # Get topic model metrics
            topic_info = topic_model.get_topic_info()
            num_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
            
            result = {
                "config_dir": config_dir,
                "n_neighbors": n_neighbors,
                "n_components": n_components,
                "min_topic_size": min_topic_size,
                "num_topics": num_topics,
                "status": "success"
            }
            results.append(result)
            
            print(f"✅ SUCCESS: Generated {num_topics} topics")
            print(f"   Output directory: {output_dir}")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            result = {
                "config_dir": config_dir,
                "n_neighbors": n_neighbors,
                "n_components": n_components,
                "min_topic_size": min_topic_size,
                "num_topics": 0,
                "status": "failed",
                "error": str(e)
            }
            results.append(result)
    
    # Save grid search summary
    summary_path = os.path.join(base_output_dir, "grid_search_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Grid search completed! Summary saved to: {summary_path}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\nGrid Search Results Summary:")
    print(f"{'Config':<40} {'Topics':<8} {'Status':<10}")
    print("-" * 60)
    for result in results:
        config_name = result['config_dir']
        num_topics = result['num_topics']
        status = result['status']
        print(f"{config_name:<40} {num_topics:<8} {status:<10}")


def main():
    parser = argparse.ArgumentParser(description="Run grid search BERTopic on a JSONL corpus")
    parser.add_argument("--corpus_path", required=True, help="Path to input JSONL (with '_id' and 'text').")
    parser.add_argument("--base_output_dir", required=True, help="Base directory for output files.")

    parser.add_argument("--chunk_mode", choices=["sentence", "window"], default="sentence", help="Chunking mode.")
    parser.add_argument("--win_size", type=int, default=4, help="Window size if chunk_mode='window'.")
    parser.add_argument("--win_step", type=int, default=2, help="Window step if chunk_mode='window'.")
    
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
    parser.add_argument("--embed_device", default="cpu", help="Device for embeddings (e.g., 'cpu' or 'cuda:0').")

    # Grid search parameters
    parser.add_argument("--n_neighbors", nargs="+", type=int, default=[10, 15, 30], help="List of n_neighbors values for UMAP.")
    parser.add_argument("--n_components", nargs="+", type=int, default=[5, 10, 15], help="List of n_components values for UMAP.")
    parser.add_argument("--min_topic_size", nargs="+", type=int, default=[5, 10, 15], help="List of min_topic_size values for BERTopic.")

    # BERTopic parameters
    parser.add_argument("--top_n_words", type=int, default=10, help="Number of top words per topic to extract.")
    parser.add_argument("--save_topic_model", action="store_true", default=False, help="Save the BERTopic model to disk.")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.base_output_dir, exist_ok=True)

    # Run grid search
    run_grid_search(
        corpus_path=args.corpus_path,
        base_output_dir=args.base_output_dir,
        chunk_mode=args.chunk_mode,
        win_size=args.win_size,
        win_step=args.win_step,
        embed_model=args.embed_model,
        embed_device=args.embed_device,
        n_neighbors_list=args.n_neighbors,
        n_components_list=args.n_components,
        min_topic_size_list=args.min_topic_size,
        args=args
    )

if __name__ == "__main__":
    main()
