#!/usr/bin/env python3
"""
Example usage of the simplified scoring functions.
"""

import torch
from sentence_transformers import SentenceTransformer
from src.scoring import QueryScorer, QueryFilterScorer

def example_basic_scoring():
    """Example of basic query scoring without complex setup."""
    
    # Sample queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Applications of neural networks",
        "Difference between AI and ML"
    ]
    
    # Load embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Simple diversity scoring
    diversity = QueryScorer.diversity_score(queries, embed_model)
    print(f"Diversity score: {diversity:.3f}")
    
    # Simple keyword coverage (without BM25)
    keywords = ["machine learning", "deep learning", "neural networks", "artificial intelligence"]
    keyword_score = QueryScorer.bm25_keyword_coverage_score(queries, keywords)
    print(f"Keyword coverage score: {keyword_score:.3f}")


def example_topic_coverage_scoring():
    """Example of topic coverage scoring."""
    
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Applications of neural networks"
    ]
    
    # Load embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Mock topic vectors (in practice, load from file)
    # Assume we have 100 topics with 384-dimensional embeddings
    topic_vecs = torch.randn(100, 384)
    topic_vecs = torch.nn.functional.normalize(topic_vecs, dim=1)
    
    # Document topics and weights
    topic_ids = [5, 23, 67]  # This document relates to topics 5, 23, and 67
    topic_weights = [0.5, 0.3, 0.2]  # Weights for each topic
    
    # Score topic coverage
    coverage_score = QueryScorer.topic_coverage_score(
        queries=queries,
        topic_ids=topic_ids,
        topic_weights=topic_weights,
        topic_vecs=topic_vecs,
        embed_model=embed_model,
        tau=0.35
    )
    print(f"Topic coverage score: {coverage_score:.3f}")


def example_comprehensive_scoring():
    """Example of comprehensive scoring using QueryFilterScorer."""
    
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Applications of neural networks",
        "Difference between AI and ML"
    ]
    
    # Load embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Mock data (in practice, load from files)
    topic_vecs = torch.randn(100, 384)
    topic_vecs = torch.nn.functional.normalize(topic_vecs, dim=1)
    
    doc_vectors = {
        "doc_123": torch.randn(384),
        "doc_456": torch.randn(384)
    }
    
    # Create scorer
    scorer = QueryFilterScorer(
        embed_model=embed_model,
        topic_vecs=topic_vecs,
        doc_vectors=doc_vectors,
        weights={
            'diversity': 0.3,
            'topic_coverage': 0.4,
            'keyword_coverage': 0.3
        }
    )
    
    # Score queries
    overall_score = scorer.score_queries(
        queries=queries,
        topic_ids=[5, 23, 67],
        topic_weights=[0.5, 0.3, 0.2],
        keywords=["machine learning", "deep learning", "neural networks"],
        doc_id="doc_123",
        components=['diversity', 'topic_coverage', 'keyword_coverage', 'relevance']
    )
    
    print(f"Overall score: {overall_score:.3f}")


if __name__ == "__main__":
    print("=== Basic Scoring Example ===")
    example_basic_scoring()
    
    print("\n=== Topic Coverage Example ===")
    example_topic_coverage_scoring()
    
    print("\n=== Comprehensive Scoring Example ===")
    example_comprehensive_scoring()
