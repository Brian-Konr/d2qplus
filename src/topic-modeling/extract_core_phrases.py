import numpy as np
import json
import argparse
from keybert import KeyBERT
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm import tqdm

class CorePhraseExtractor:
    """
    Extract core phrases using the CCQGen methodology with existing topic assignments.
    Core phrases are relevant to the document but distinctive within its topic group.
    """
    
    def __init__(self, embedding_model="allenai/scibert_scivocab_uncased", device="cuda:2"):
        """Initialize the extractor with KeyBERT and embedding model."""
        embedder = SentenceTransformer(embedding_model, device=device)
        self.kw_model = KeyBERT(model=embedder)
        
    def extract_core_phrases(self, corpus, doc_topics, 
                           top_n_candidates=20, 
                           selection_ratio=0.2,
                           min_phrases_per_doc=1,
                           max_phrases_per_doc=8,
                           keyphrase_ngram_range=(1, 3),
                           use_mmr=True,
                           diversity=0.7,
                           candidate_keywords_path=None
                           ):
        """
        Extract core phrases for each document using the CCQGen distinctiveness score.
        
        Args:
            corpus: List of documents with '_id' and 'text' keys
            doc_topics: List of topic assignments with 'doc_id' and 'topics' keys
            top_n_candidates: Number of candidate phrases to extract initially
            selection_ratio: Ratio of candidates to select as core phrases (0.2 = top 20%)
            min_phrases_per_doc: Minimum number of phrases per document
            max_phrases_per_doc: Maximum number of phrases per document
            keyphrase_ngram_range: N-gram range for phrase extraction
            use_mmr: Use Maximal Marginal Relevance for diversity
            diversity: Diversity parameter for MMR (higher = more diverse)
        """
        
        if candidate_keywords_path:
            import pandas as pd
            cand_keywords = pd.read_pickle(candidate_keywords_path)
        # Create document lookup
        doc_lookup = {doc['_id']: doc for doc in corpus}
        topic_assignment = {item['doc_id']: item['topics'] for item in doc_topics}
        
        # Group documents by their assigned topics (weighted by topic strength)
        print("Grouping documents by topics...")
        topic_to_docs = self._group_docs_by_topics(topic_assignment)
        
        # Create BM25 models for each topic
        print("Creating BM25 models for each topic...")
        topic_bm25_models = self._create_topic_bm25_models(topic_to_docs, doc_lookup)
        
        # Extract core phrases for each document
        print("Extracting core phrases...")
        doc_core_phrases = {}
        
        for doc_item in tqdm(doc_topics, desc="Processing documents"):
            doc_id = doc_item['doc_id']
            doc_topics_list = doc_item['topics']
            
            if doc_id not in doc_lookup:
                continue
                
            doc_text = doc_lookup[doc_id]['text']
            
            # Extract candidate phrases with improved KeyBERT parameters
            candidate_phrases = []
            if candidate_keywords_path:
                candidate_phrases = cand_keywords[doc_id]
            else:
                candidate_phrases = self.kw_model.extract_keywords(
                    doc_text,
                    keyphrase_ngram_range=keyphrase_ngram_range,
                    stop_words='english',
                    top_n=top_n_candidates,
                    use_mmr=use_mmr,
                    diversity=diversity,  # Higher diversity to avoid redundant phrases
                    use_maxsum=False,     # MMR is generally better than MaxSum
                    nr_candidates=top_n_candidates * 2  # More candidates for MMR to choose from
                )
            
            if not candidate_phrases:
                doc_core_phrases[doc_id] = []
                continue
            
            # Calculate distinctiveness scores for each candidate
            distinctive_phrases = self._calculate_distinctiveness_scores(
                candidate_phrases, doc_id, doc_topics_list, 
                topic_to_docs, topic_bm25_models, doc_lookup
            )
            
            # Select top phrases based on distinctiveness
            num_to_select = max(
                min_phrases_per_doc,
                min(max_phrases_per_doc, int(len(distinctive_phrases) * selection_ratio))
            )
            
            distinctive_phrases.sort(key=lambda x: x[1], reverse=True)
            selected_phrases = distinctive_phrases[:num_to_select]
            
            doc_core_phrases[doc_id] = [
                {"phrase": phrase, "distinctiveness_score": score} 
                for phrase, score in selected_phrases
            ]
        
        return doc_core_phrases
    
    def _group_docs_by_topics(self, topic_assignment):
        """Group documents by their assigned topics, considering topic weights."""
        topic_to_docs = defaultdict(list)
        
        for doc_id, topics in topic_assignment.items():
            for topic_info in topics:
                topic_id = topic_info['topic_id']
                weight = topic_info['weight']
                topic_to_docs[topic_id].append({
                    'doc_id': doc_id,
                    'weight': weight
                })
        
        return topic_to_docs
    
    def _create_topic_bm25_models(self, topic_to_docs, doc_lookup):
        """Create BM25 models for each topic."""
        topic_bm25_models = {}
        
        for topic_id, doc_list in topic_to_docs.items():
            # Get document texts for this topic
            docs_in_topic = []
            for doc_info in doc_list:
                doc_id = doc_info['doc_id']
                if doc_id in doc_lookup:
                    docs_in_topic.append(doc_lookup[doc_id]['text'])
            
            if len(docs_in_topic) > 1:  # Need at least 2 docs for BM25 comparison
                # Tokenize documents for BM25
                tokenized_corpus = [doc.lower().split() for doc in docs_in_topic]
                topic_bm25_models[topic_id] = {
                    'model': BM25Okapi(tokenized_corpus),
                    'doc_ids': [doc_info['doc_id'] for doc_info in doc_list if doc_info['doc_id'] in doc_lookup]
                }
        
        return topic_bm25_models
    
    def _calculate_distinctiveness_scores(self, candidate_phrases, current_doc_id, 
                                        doc_topics_list, topic_to_docs, 
                                        topic_bm25_models, doc_lookup):
        """Calculate distinctiveness scores using CCQGen methodology."""
        distinctive_phrases = []
        
        for phrase, relevance_score in candidate_phrases:
            # Calculate distinctiveness across all topics this document belongs to
            total_distinctiveness = 0.0
            total_weight = 0.0
            
            # # sort by weight to prioritize more relevant topics
            # doc_topics_list.sort(key=lambda x: x['weight'], reverse=True)
            
            for topic_info in doc_topics_list:
                topic_id = topic_info['topic_id']
                topic_weight = topic_info['weight']
                
                if topic_id not in topic_bm25_models:
                    # If topic has insufficient docs for BM25, use relevance score only
                    distinctiveness = relevance_score
                else:
                    bm25_info = topic_bm25_models[topic_id]
                    bm25_model = bm25_info['model']
                    topic_doc_ids = bm25_info['doc_ids']
                    
                    # Find current document's position in the topic
                    try:
                        current_doc_idx = topic_doc_ids.index(current_doc_id)
                    except ValueError:
                        # Document not found in topic (shouldn't happen)
                        distinctiveness = relevance_score
                        continue
                    
                    # Calculate BM25 scores for the phrase across all docs in topic
                    tokenized_phrase = phrase.lower().split()
                    all_scores = bm25_model.get_scores(tokenized_phrase)
                    
                    # Current document's BM25 score
                    current_score = all_scores[current_doc_idx]
                    
                    # Sum of exp(BM25) for other documents in topic
                    other_scores = np.concatenate([
                        all_scores[:current_doc_idx], 
                        all_scores[current_doc_idx+1:]
                    ])
                    sum_exp_others = np.sum(np.exp(other_scores))
                    
                    # CCQGen distinctiveness formula
                    distinctiveness = np.exp(current_score) / (1 + sum_exp_others)
                
                total_distinctiveness += distinctiveness * topic_weight
                total_weight += topic_weight
            
            # Weight-averaged distinctiveness score
            final_distinctiveness = total_distinctiveness / total_weight if total_weight > 0 else relevance_score
            distinctive_phrases.append((phrase, final_distinctiveness))
        
        return distinctive_phrases

# Usage example with your data
def run_core_phrase_extraction(args):
    """Run core phrase extraction on your corpus and doc_topics data."""
    CORPUS_PATH = args.corpus_path
    DOC_TOPICS_PATH = args.doc_topics_path

    # Load your corpus
    with open(CORPUS_PATH, "r") as f:
        corpus = [json.loads(line) for line in f]
    
    with open(DOC_TOPICS_PATH, "r") as f:
        doc_topics = [json.loads(line) for line in f]

    print(f"Processing {len(corpus)} documents with {len(doc_topics)} topic assignments")
    
    # Initialize extractor
    extractor = CorePhraseExtractor(
        embedding_model=args.embedding_model,
        device=args.device
    )
    
    # Extract core phrases
    core_phrases = extractor.extract_core_phrases(
        corpus=corpus,
        doc_topics=doc_topics,
        top_n_candidates=args.top_n_candidates,
        selection_ratio=args.selection_ratio,
        min_phrases_per_doc=args.min_phrases,
        max_phrases_per_doc=args.max_phrases,
        keyphrase_ngram_range=(args.min_ngram, args.max_ngram),
        use_mmr=args.use_mmr,
        diversity=args.diversity,
        candidate_keywords_path=args.candidate_keywords_path
    )
    
    # Save results
    output_path = args.output_path
    with open(output_path, "w") as f:
        for doc_id, phrases in core_phrases.items():
            f.write(json.dumps({"doc_id": doc_id, "core_phrases": phrases}) + "\n")
    
    print(f"Core phrases saved to {output_path}")
    
    # Show some examples
    if args.show_examples:
        print("\nExample results:")
        for i, (doc_id, phrases) in enumerate(list(core_phrases.items())[:3]):
            print(f"\nDocument {doc_id}:")
            for phrase_info in phrases:
                print(f"  - '{phrase_info['phrase']}' (score: {phrase_info['distinctiveness_score']:.4f})")
    
    return core_phrases

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract core phrases from documents using CCQGen methodology.')
    
    # Input/output paths
    parser.add_argument('--corpus_path', type=str, 
                        default="/home/guest/r12922050/GitHub/d2qplus/data/CSFCube-1.1/corpus.jsonl",
                        help='Path to the corpus file (JSONL format)')
    parser.add_argument('--doc_topics_path', type=str, 
                        default="/home/guest/r12922050/GitHub/d2qplus/augmented-data/CSFCube-1.1/topics/0609-pritamdeka_scibert-biobert-pos-keybert-mmr/doc_topics.jsonl",
                        help='Path to the document topics file (JSONL format)')
    parser.add_argument('--output_path', type=str, 
                        default="/home/guest/r12922050/GitHub/d2qplus/augmented-data/CSFCube-1.1/keywords/core_phrases_ccqgen.jsonl",
                        help='Path to save the extracted core phrases (JSONL format)')
    parser.add_argument('--candidate_keywords_path', type=str,
                        default=None,
                        help='Path to precomputed candidate keywords (optional, for faster extraction)')
    
    # Model parameters
    parser.add_argument('--embedding_model', type=str, 
                        default="pritamdeka/S-Scibert-snli-multinli-stsb",
                        help='Name or path of the sentence transformer embedding model')
    parser.add_argument('--device', type=str, default="cuda:1",
                        help='Device to use for embedding model (e.g., cuda:0, cpu)')
    
    # Extraction parameters
    parser.add_argument('--top_n_candidates', type=int, default=50,
                        help='Number of candidate phrases to extract initially')
    parser.add_argument('--selection_ratio', type=float, default=0.25,
                        help='Ratio of candidates to select as core phrases')
    parser.add_argument('--min_phrases', type=int, default=2,
                        help='Minimum number of phrases per document')
    parser.add_argument('--max_phrases', type=int, default=6,
                        help='Maximum number of phrases per document')
    parser.add_argument('--min_ngram', type=int, default=1,
                        help='Minimum n-gram length for keyphrases')
    parser.add_argument('--max_ngram', type=int, default=3,
                        help='Maximum n-gram length for keyphrases')
    parser.add_argument('--use_mmr', action='store_true', default=True,
                        help='Use Maximal Marginal Relevance for diversity')
    parser.add_argument('--diversity', type=float, default=0.6,
                        help='Diversity parameter for MMR (0.0=no diversity, 1.0=max diversity)')
    
    # Other parameters
    parser.add_argument('--show_examples', action='store_true', default=True,
                        help='Show example results after extraction')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    core_phrases_results = run_core_phrase_extraction(args)