import numpy as np
from bertopic import BERTopic
from keybert import KeyBERT
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
import logging
from typing import List, Dict

# Download required NLTK data
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RewardModel:
    def __init__(self, bert_model: str = "all-MiniLM-L6-v2", bm25_k1: float = 1.5, bm25_b: float = 0.75):
        """Initialize the reward model with pre-trained models and BM25 parameters."""
        self.sentence_model = SentenceTransformer(bert_model)
        self.kw_model = KeyBERT(model=bert_model)
        self.topic_model = BERTopic(embedding_model=bert_model, verbose=True)
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        logger.info("Reward model initialized with %s", bert_model)

    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize text into words, removing stopwords and punctuation."""
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalnum()]

    def topic_coverage(self, document: str, queries: List[str]) -> float:
        """Compute topic coverage using BERTopic."""
        try:
            # Split document into sentences for topic modeling
            sentences = document.split('. ')
            topics, _ = self.topic_model.fit_transform(sentences)
            topic_info = self.topic_model.get_topic_info()
            doc_topics = set(self.topic_model.get_document_info(sentences)['Topic'])

            # Embed queries and assign to topics
            query_topics = set()
            for query in queries:
                query_topic = self.topic_model.transform([query])[0][0]
                if query_topic != -1:  # Exclude outlier topic
                    query_topics.add(query_topic)

            # Compute coverage as proportion of document topics covered
            if len(doc_topics) == 0:
                logger.warning("No topics detected in document")
                return 0.0
            coverage = len(query_topics.intersection(doc_topics)) / len(doc_topics)
            logger.info("Topic coverage: %.4f", coverage)
            return coverage
        except Exception as e:
            logger.error("Error in topic_coverage: %s", e)
            return 0.0

    def keyword_coverage(self, document: str, queries: List[str], threshold: float = 0.7) -> float:
        """Compute keyword coverage using KeyBERT."""
        try:
            # Extract top-10 keywords
            keywords = self.kw_model.extract_keywords(document, top_n=10)
            keyword_texts = [kw[0] for kw in keywords]
            keyword_embeddings = self.sentence_model.encode(keyword_texts, convert_to_tensor=True)

            # Compute max similarity between each keyword and any query
            query_embeddings = self.sentence_model.encode(queries, convert_to_tensor=True)
            similarities = []
            for kw_emb in keyword_embeddings:
                max_sim = np.max(np.dot(query_embeddings, kw_emb) / 
                                (np.linalg.norm(query_embeddings, axis=1) * np.linalg.norm(kw_emb)))
                similarities.append(max_sim if max_sim >= threshold else 0)

            # Average similarities
            coverage = np.mean(similarities) if similarities else 0.0
            logger.info("Keyword coverage: %.4f", coverage)
            return coverage
        except Exception as e:
            logger.error("Error in keyword_coverage: %s", e)
            return 0.0

    def term_overlap(self, document: str, queries: List[str]) -> float:
        """Compute term-based overlap using BM25."""
        try:
            # Tokenize document and initialize BM25
            doc_tokens = self.preprocess_text(document)
            bm25 = BM25Okapi([doc_tokens], k1=self.bm25_k1, b=self.bm25_b)

            # Compute BM25 score for each query
            scores = []
            for query in queries:
                query_tokens = self.preprocess_text(query)
                score = bm25.get_scores(query_tokens)[0]
                scores.append(score)

            # Normalize by max possible score (approximated as sum of max idf scores)
            max_score = sum(bm25.idf.get(t, 0) for t in doc_tokens)
            normalized_scores = [s / max_score if max_score > 0 else 0 for s in scores]
            overlap = np.mean(normalized_scores) if normalized_scores else 0.0
            logger.info("Term overlap: %.4f", overlap)
            return overlap
        except Exception as e:
            logger.error("Error in term_overlap: %s", e)
            return 0.0

    def semantic_relationship(self, document: str, queries: List[str]) -> float:
        """Compute semantic relationship using cosine similarity."""
        try:
            # Embed document and queries
            doc_embedding = self.sentence_model.encode([document], convert_to_tensor=True)
            query_embeddings = self.sentence_model.encode(queries, convert_to_tensor=True)

            # Compute cosine similarities
            similarities = np.dot(query_embeddings, doc_embedding.T).flatten() / (
                np.linalg.norm(query_embeddings, axis=1) * np.linalg.norm(doc_embedding, axis=1)
            )
            semantic_score = np.mean(similarities) if similarities.size > 0 else 0.0
            logger.info("Semantic relationship: %.4f", semantic_score)
            return semantic_score
        except Exception as e:
            logger.error("Error in semantic_relationship: %s", e)
            return 0.0

    def context_dependency(self, document: str, queries: List[str]) -> float:
        """Compute context dependency by aligning queries with document sentences."""
        try:
            # Split document into sentences and embed
            sentences = document.split('. ')
            sentence_embeddings = self.sentence_model.encode(sentences, convert_to_tensor=True)
            query_embeddings = self.sentence_model.encode(queries, convert_to_tensor=True)

            # Compute max similarity between each query and any sentence
            max_similarities = []
            for q_emb in query_embeddings:
                similarities = np.dot(sentence_embeddings, q_emb) / (
                    np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(q_emb)
                )
                max_similarities.append(np.max(similarities) if similarities.size > 0 else 0)

            context_score = np.mean(max_similarities) if max_similarities else 0.0
            logger.info("Context dependency: %.4f", context_score)
            return context_score
        except Exception as e:
            logger.error("Error in context_dependency: %s", e)
            return 0.0

    def compute_reward(self, document: str, queries: List[str], weights: Dict[str, float] = None) -> float:
        """Compute combined reward for a document and query set."""
        try:
            # Default equal weights
            if weights is None:
                weights = {
                    "topic": 0.2,
                    "keyword": 0.2,
                    "term": 0.2,
                    "semantic": 0.2,
                    "context": 0.2
                }

            # Compute individual rewards
            rewards = {
                "topic": self.topic_coverage(document, queries),
                "keyword": self.keyword_coverage(document, queries),
                "term": self.term_overlap(document, queries),
                "semantic": self.semantic_relationship(document, queries),
                "context": self.context_dependency(document, queries)
            }

            # Combine rewards
            total_reward = sum(weights[k] * v for k, v in rewards.items())
            logger.info("Combined reward: %.4f (components: %s)", total_reward, rewards)
            return total_reward
        except Exception as e:
            logger.error("Error in compute_reward: %s", e)
            return 0.0

# Example usage
if __name__ == "__main__":
    # Sample document and queries
    document = """
    Artificial intelligence (AI) is transforming industries. Machine learning models, 
    such as neural networks, enable tasks like image recognition and natural language 
    processing. Deep learning advancements have improved AI performance significantly.
    """
    queries = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "What are applications of deep learning?",
        "Why is machine learning important?"
    ]

    # Initialize and test reward model
    reward_model = RewardModel()
    reward = reward_model.compute_reward(document, queries)
    print(f"Total Reward: {reward:.4f}")