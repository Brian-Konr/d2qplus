#!/usr/bin/env python3
"""

Run BERTopic on a JSONL corpus with either:
  • sentence-level chunks  (chunk_mode="sentence")
  • sliding paragraph windows (chunk_mode="window")
Outputs a JSONL file mapping each doc_id to its topic distribution.
"""

import argparse, json, collections, random, os, sys
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

    pos = PartOfSpeech("en_core_sci_sm", pos_patterns=pos_patterns, top_n_words=250)
    keybert = KeyBERTInspired(nr_candidate_words=100, nr_repr_docs=5, top_n_words=25)
    mmr = MaximalMarginalRelevance(diversity=0.6, top_n_words=args.top_n_words)

    representation_chain = [pos, keybert, mmr]

    vectorizer = CountVectorizer(stop_words="english", ngram_range=args.n_gram_range, min_df=5)

    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=UMAP(n_components=5, metric="cosine"),
        min_topic_size=5,
        vectorizer_model=vectorizer,
        representation_model=representation_chain,
        verbose=True
    )

    # Fit and transform
    topics, probs = topic_model.fit_transform(chunks)

    if args.reduce_outliers:
        # Reduce outliers if specified
        print("Reducing outliers...")
        topics = topic_model.reduce_outliers(chunks, topics, probabilities=probs, strategy="probabilities")
        topic_model.update_topics(chunks, topics=topics)

    # Aggregate per document and write output
    os.makedirs(os.path.dirname(corpus_topics_out_path) or ".", exist_ok=True)
    with open(corpus_topics_out_path, "w", encoding="utf-8") as fout:
        for doc_id, idx_list in doc2chunk_idx.items():
            freq = collections.Counter(topics[i] for i in idx_list if topics[i] != -1)
            total = sum(freq.values())
            topic_entries = []
            for tid, cnt in freq.items():
                topic_entries.append({"topic_id": int(tid), "weight": round(cnt / total, 6)})
            fout.write(json.dumps({"doc_id": doc_id, "topics": topic_entries}, ensure_ascii=False) + "\n")

    print(f"Wrote document-topic distributions to '{corpus_topics_out_path}'")

    # Save topic model information
    topic_df = topic_model.get_topic_info()
    os.makedirs(os.path.dirname(topic_info_dataframe_out) or ".", exist_ok=True)
    topic_df.to_csv(topic_info_csv_out, index=False)
    print(f"Wrote topic model info to '{topic_info_csv_out}'")
    topic_df.to_pickle(topic_info_dataframe_out)
    print(f"Wrote topic model info to '{topic_info_dataframe_out}'")

    # Optionally save the BERTopic model
    if args.save_topic_model:
        topic_model_out = os.path.join(base_output_dir, "bertopic_model")
        os.makedirs(os.path.dirname(topic_model_out) or ".", exist_ok=True)
        topic_model.save(topic_model_out)
        print(f"Saved BERTopic model to '{topic_model_out}'")

    return topic_model


def main():
    parser = argparse.ArgumentParser(description="Run sentence- or window-based BERTopic on a JSONL corpus")
    parser.add_argument("--corpus_path", required=True, help="Path to input JSONL (with '_id' and 'text').")
    parser.add_argument("--base_output_dir", required=True, help="Base directory for output files.")

    parser.add_argument("--chunk_mode", choices=["sentence", "window"], default="sentence", help="Chunking mode.")
    parser.add_argument("--win_size", type=int, default=4, help="Window size if chunk_mode='window'.")
    parser.add_argument("--win_step", type=int, default=2, help="Window step if chunk_mode='window'.")
    
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
    parser.add_argument("--embed_device", default="cpu", help="Device for embeddings (e.g., 'cpu' or 'cuda:0').")

    # - BERTopic parameters
    parser.add_argument("--min_topic_size", type=int, default=3, help="Minimum topic size for BERTopic.")
    parser.add_argument("--top_n_words", type=int, default=10, help="Number of top words per topic to extract.")
    parser.add_argument("--n_gram_range", type=lambda x: tuple(map(int, x.split(','))), default="1,3", help="Range of n-grams to consider for topic keywords (min,max)")
    parser.add_argument("--reduce_outliers", action="store_true", help="Reduce outliers in topic model.")
    parser.add_argument("--save_topic_model", action="store_true", default=False, help="Save the BERTopic model to disk.")


    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.base_output_dir, exist_ok=True)

    # output every arguments to parameters.txt
    params_path = os.path.join(args.base_output_dir, "parameters.txt")
    with open(params_path, "w", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"Parameters written to '{params_path}'")

    run_topic_modeling(
        corpus_path=args.corpus_path,
        base_output_dir=args.base_output_dir,
        chunk_mode=args.chunk_mode,
        win_size=args.win_size,
        win_step=args.win_step,
        embed_model=args.embed_model,
        embed_device=args.embed_device,
        args=args
    )


if __name__ == "__main__":
    main()
