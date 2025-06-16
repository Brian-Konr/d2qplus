import pickle
import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract keywords using KeyBERT")
    parser.add_argument("--corpus_path", type=str, required=True, 
                       help="Path to corpus.jsonl file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for keyword files")
    parser.add_argument("--model_name", type=str, required=True,
                       help="SentenceTransformer model name")
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="Device to use (cuda:0, cuda:1, cpu)")
    parser.add_argument("--top_n_candidates", type=int, default=10,
                       help="Number of top candidate keywords to extract")
    parser.add_argument("--ngram_min", type=int, default=1,
                       help="Minimum n-gram size")
    parser.add_argument("--ngram_max", type=int, default=2,
                       help="Maximum n-gram size")
    parser.add_argument("--diversity", type=float, default=0.6,
                       help="Diversity parameter for MMR (0.0-1.0)")
    parser.add_argument("--use_mmr", action="store_true", default=True,
                       help="Use Maximal Marginal Relevance")
    parser.add_argument("--output_name", type=str, default="candidate_keywords_scibert.pkl",
                       help="Output pickle file name")
    
    args = parser.parse_args()
    
    # Initialize KeyBERT with specified model
    print(f"Initializing KeyBERT model: {args.model_name}")
    embedder = SentenceTransformer(args.model_name, device=args.device)
    kw_model = KeyBERT(model=embedder)

    # Load corpus
    print(f"Loading corpus from: {args.corpus_path}")
    with open(args.corpus_path, "r") as f:
        corpus = [json.loads(line) for line in f]

    print(f"Loaded {len(corpus)} documents")

    # Parameters for keyword extraction
    keyphrase_ngram_range = (args.ngram_min, args.ngram_max)
    
    # Extract candidate keywords for all documents
    candidate_keywords = {}
    print("Extracting candidate keywords...")

    for doc in tqdm(corpus, desc="Processing documents"):
        doc_id = doc["_id"]
        doc_text = doc["text"]
        
        # Extract candidate phrases
        candidate_phrases = kw_model.extract_keywords(
            doc_text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words='english',
            top_n=args.top_n_candidates,
            use_mmr=args.use_mmr,
            diversity=args.diversity,
            use_maxsum=False,
            nr_candidates=args.top_n_candidates * 2
        )
        
        candidate_keywords[doc_id] = candidate_phrases

    # Save to pickle file
    os.makedirs(args.output_dir, exist_ok=True)
    pickle_path = os.path.join(args.output_dir, args.output_name)

    with open(pickle_path, "wb") as f:
        pickle.dump(candidate_keywords, f)

    print(f"Saved candidate keywords to {pickle_path}")
    print(f"Total documents processed: {len(candidate_keywords)}")

    # Show example results
    print("\nExample results:")
    sample_doc_ids = list(candidate_keywords.keys())[:3]
    for doc_id in sample_doc_ids:
        print(f"\nDocument {doc_id}:")
        keywords = candidate_keywords[doc_id][:5]  # Show top 5
        for phrase, score in keywords:
            print(f"  - '{phrase}' (score: {score:.4f})")

if __name__ == "__main__":
    main()