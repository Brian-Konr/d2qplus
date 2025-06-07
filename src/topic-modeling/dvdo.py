#!/usr/bin/env python3
"""
Optimised BERTopic with sentence‑level chunking
==============================================
This single script implements the DVDO + silhouette/coherence search from
Sawant et al. (2022) while **fixing the variable‑scope bug** the user hit.
The changes are:

1. All references inside the min_cluster_size sweep now use the local
   variable **`tm`** (the temporary BERTopic instance) instead of the
   undefined `topic_model`.
2. The best model is captured in `best_tm` and returned after the grid
   search.
3. The per‑document aggregation stage uses `best_tm.transform(...)` and
   therefore runs only once – no wasted compute.

Run with e.g.::

    python optimise_sentence_bertopic.py \
        --corpus docs.jsonl \
        --out_dir output/ \
        --dims 5 8 10 12 15 \
        --min_cluster_sizes 5 10 20 30
"""
import argparse, json, random, collections, re, os
from pathlib import Path

import nltk; nltk.download("punkt", quiet=True)
from nltk import sent_tokenize

import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_sentence_chunks(jsonl_path):
    """Return (chunks, chunk_owner, doc2idx) from a JSONL corpus."""
    chunks, chunk_owner = [], []
    doc2idx = collections.defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id, text = obj.get("_id"), obj.get("text", "")
            if not doc_id or not isinstance(text, str):
                continue
            for sent in (s.strip() for s in sent_tokenize(text) if s.strip()):
                idx = len(chunks)
                chunks.append(sent)
                chunk_owner.append(doc_id)
                doc2idx[doc_id].append(idx)
    print(f"Built {len(chunks):,} sentence chunks from {len(doc2idx):,} docs")
    return chunks, chunk_owner, doc2idx


def tokenise_texts(texts):
    return [re.findall(r"\b\w+\b", t.lower()) for t in texts]

def flatten_ngrams(word_scores, max_words=10):
    """Convert a list[(token, score)] to a list[str] of **unique unigrams**.

    Keeps order of first occurrence and stops when `max_words` unigrams are
    collected.  Example: [("credit card", 0.9), ("late", 0.5)] →
    ["credit", "card", "late"]
    """
    seen, out = set(), []
    for w, _ in word_scores[:max_words]:
        for unigram in w.split():
            if unigram not in seen:
                seen.add(unigram)
                out.append(unigram)
            if len(out) >= max_words:
                return out
    return out

def coherence_cv(topics, tokenised_docs):
    """Compute C_V coherence for a list[list[str]]."""
    if not topics:
        return 0.0
    dictionary = Dictionary(tokenised_docs)
    cm = CoherenceModel(topics=topics, texts=tokenised_docs,
                        dictionary=dictionary, coherence="c_v")
    return cm.get_coherence()

# ---------------------------------------------------------------------------
# Main optimisation routine
# ---------------------------------------------------------------------------

def optimise_bertopic_sentence(corpus_path: str, out_dir: str, *,
                               dims=(5, 8, 10, 12, 15),
                               min_sizes=(5, 10, 20, 30),
                               alpha=0.6,
                               embed_model="all-MiniLM-L6-v2",
                               device="cpu"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # -------- Stage 0: Sentence chunks & embeddings ------------------------
    chunks, chunk_owner, doc2idx = build_sentence_chunks(corpus_path)
    embedder = SentenceTransformer(embed_model, device=device)
    E = embedder.encode(chunks, show_progress_bar=True)

    # -------- Stage A: DVDO (best UMAP dimension) --------------------------
    best_dim, best_Z, best_sil = None, None, -1
    for d in dims:
        Z = UMAP(n_components=d, metric="cosine").fit_transform(E)
        k = max(2, int(np.sqrt(len(chunks))))
        sil = silhouette_score(Z, KMeans(k, random_state=42).fit_predict(Z))
        print(f"UMAP dim={d:<2}  silhouette={sil:.4f}")
        if sil > best_sil:
            best_dim, best_Z, best_sil = d, Z, sil
    print(f"★ DVDO selected n_components={best_dim} (silhouette={best_sil:.4f})")

    # Prepare tokenised docs once for coherence reuse
    tokenised_docs = tokenise_texts(chunks)

    # -------- Stage B: min_cluster_size sweep -----------------------------
    best_score, best_cfg = -1, None
    for m in min_sizes:
        hdb_tmp = HDBSCAN(min_cluster_size=m, metric="euclidean").fit(best_Z)
        labels = hdb_tmp.labels_
        if len(np.unique(labels)) < 2:
            continue  # degenerate
        sil = silhouette_score(best_Z, labels)

        tm = BERTopic(embedding_model=embedder,
                      umap_model=UMAP(n_components=best_dim, metric="cosine"),
                      hdbscan_model=HDBSCAN(min_cluster_size=m, metric="euclidean", prediction_data=True),
                      calculate_probabilities=False,
                      top_n_words=10,
                      verbose=True)
        tm.fit(chunks)
        
        top_words = []
        for tid in tm.get_topics().keys():
            if tid == -1:
                continue
            ws = tm.get_topic(tid)
            if not ws:
                continue
            unigrams = flatten_ngrams(ws)        
            if len(unigrams) >= 2:
                top_words.append(unigrams)

        coh = coherence_cv(top_words, tokenised_docs)
        score = alpha*coh + (1-alpha)*sil
        print(f"   min_cluster={m:<2}  C_V={coh:.4f}  sil={sil:.4f}  → score={score:.4f}")
        if score > best_score:
            best_score, best_cfg = score, (m, tm, coh, sil)

    if best_cfg is None:
        raise RuntimeError("All min_cluster_size settings degenerated – try different grid.")

    m_best, best_tm, coh_best, sil_best = best_cfg
    print(f"✔ Selected model: min_cluster_size={m_best}, C_V={coh_best:.3f}, sil={sil_best:.3f}")

    # -------- Stage C: Aggregate per‑document -----------------------------
    topic_ids, _ = best_tm.transform(chunks)
    doc_topics_path = Path(out_dir) / "doc_topics.jsonl"
    with open(doc_topics_path, "w", encoding="utf-8") as fout:
        for doc_id, idxs in doc2idx.items():
            cnt = collections.Counter(topic_ids[i] for i in idxs if topic_ids[i] != -1)
            total = sum(cnt.values()) or 1
            vec = [{"topic_id": int(t), "weight": round(v/total, 6)} for t, v in cnt.items()]
            fout.write(json.dumps({"doc_id": doc_id, "topics": vec}) + "\n")
    print(f"✔ Wrote per‑doc distributions → {doc_topics_path}")

    # -------- Stage D: Persist topic info & model -------------------------
    best_tm.get_topic_info().to_pickle(Path(out_dir) / "topic_info_dataframe.pkl")
    best_tm.save(Path(out_dir) / "bertopic_model")
    print("✔ Saved topic_info_dataframe and full BERTopic model")

    return best_tm

# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="Path to input JSONL with _id + text")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dims", nargs="*", type=int, default=[5,8,10,12,15])
    p.add_argument("--min_cluster_sizes", nargs="*", type=int, default=[5,10,15,20])
    p.add_argument("--alpha", type=float, default=0.6, help="Weight for coherence vs silhouette")
    p.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    optimise_bertopic_sentence(args.corpus, args.out_dir,
                               dims=args.dims,
                               min_sizes=args.min_cluster_sizes,
                               alpha=args.alpha,
                               embed_model=args.embed_model,
                               device=args.device)
