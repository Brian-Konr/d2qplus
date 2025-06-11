# Query Scoring for Filtering

This document explains how to use the simplified query scoring functions for filtering purposes.

## Overview

The new scoring system provides simplified interfaces for scoring query sets without the complexity of the RL reward system. There are two main approaches:

1. **Individual Scoring Functions**: Direct access to specific scoring metrics
2. **Comprehensive Scoring**: Combined scoring with customizable weights

## Quick Start

### Basic Usage

```python
from sentence_transformers import SentenceTransformer
from src.reward import score_diversity, score_keyword_coverage

# Your queries
queries = ["What is ML?", "How does AI work?", "Neural networks"]

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Score diversity
diversity = score_diversity(queries, embed_model)

# Score keyword coverage
keywords = ["machine learning", "artificial intelligence", "neural networks"]
keyword_score = score_keyword_coverage(queries, keywords)
```

### One-line Scoring

```python
from src.reward import score_query_set

score = score_query_set(
    queries=queries,
    embed_model=embed_model,
    keywords=keywords,
    components=['diversity', 'keyword_coverage']
)

# Use score for filtering
keep_queries = score >= 0.5
```

## Scoring Components

### 1. Format Score
Checks if queries meet format requirements (count, no intro text).

```python
from src.reward import score_format

score = score_format(queries, expected_n=4)
```

### 2. Diversity Score
Measures semantic diversity between queries (1.0 - average similarity).

```python
from src.reward import score_diversity

score = score_diversity(queries, embed_model)
```

### 3. Topic Coverage Score
Measures how well queries cover document topics.

```python
from src.reward import score_topic_coverage

score = score_topic_coverage(
    queries=queries,
    topic_ids=[5, 23, 67],           # Topic IDs for this document
    topic_weights=[0.5, 0.3, 0.2],   # Importance weights
    topic_vecs=topic_vectors,         # Loaded topic embeddings
    embed_model=embed_model,
    tau=0.35                         # Coverage threshold
)
```

### 4. Keyword Coverage Score
Measures keyword coverage using Jaccard similarity or BM25.

```python
from src.reward import score_keyword_coverage

# Jaccard similarity (default)
score = score_keyword_coverage(queries, keywords)

# BM25 scoring (if BM25 model available)
score = score_keyword_coverage(queries, keywords, bm25_model)
```

### 5. Relevance Score
Measures relevance to source document.

```python
from src.reward import score_relevance

score = score_relevance(queries, doc_id, doc_vectors, embed_model)
```

## Comprehensive Scoring

### Using QueryFilterScorer

```python
from src.scoring import QueryFilterScorer

# Create scorer with models and weights
scorer = QueryFilterScorer(
    embed_model=embed_model,
    topic_vecs=topic_vectors,
    doc_vectors=doc_vectors,
    weights={
        'diversity': 0.3,
        'topic_coverage': 0.4,
        'keyword_coverage': 0.3,
        'relevance': 0.2
    }
)

# Score queries
overall_score = scorer.score_queries(
    queries=queries,
    topic_ids=[5, 23, 67],
    topic_weights=[0.5, 0.3, 0.2],
    keywords=keywords,
    doc_id="doc_123",
    components=['diversity', 'topic_coverage', 'keyword_coverage']
)
```

### Factory Function

```python
from src.reward import create_filter_scorer

# Create scorer from file paths
scorer = create_filter_scorer(
    embed_model=embed_model,
    topic_vecs_path="path/to/topic_vectors.pt",
    doc_vecs_path="path/to/doc_vectors.pt",
    bm25=bm25_model,
    weights={'diversity': 0.4, 'keyword_coverage': 0.6}
)

score = scorer.score_queries(queries, keywords=keywords)
```

## Filtering Workflow

### Basic Filtering

```python
def filter_query_sets(query_sets, threshold=0.5):
    """Filter query sets based on quality scores."""
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    filtered_sets = []
    for queries in query_sets:
        score = score_query_set(
            queries=queries,
            embed_model=embed_model,
            components=['diversity']
        )
        if score >= threshold:
            filtered_sets.append(queries)
    
    return filtered_sets
```

### Advanced Filtering with Topic Coverage

```python
def filter_with_topic_coverage(query_sets, documents_metadata):
    """Filter using topic coverage and keywords."""
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    topic_vecs = torch.load("topic_vectors.pt")
    
    scorer = QueryFilterScorer(
        embed_model=embed_model,
        topic_vecs=topic_vecs,
        weights={
            'diversity': 0.2,
            'topic_coverage': 0.5,
            'keyword_coverage': 0.3
        }
    )
    
    filtered_results = []
    for queries, metadata in zip(query_sets, documents_metadata):
        score = scorer.score_queries(
            queries=queries,
            topic_ids=metadata['topic_ids'],
            topic_weights=metadata['topic_weights'],
            keywords=metadata['keywords']
        )
        
        if score >= 0.6:  # High quality threshold
            filtered_results.append((queries, score))
    
    return filtered_results
```

## Performance Tips

1. **Reuse Embedding Models**: Load embedding models once and reuse them
2. **Batch Processing**: Process multiple query sets in batches when possible
3. **Component Selection**: Only compute required scoring components
4. **Caching**: Cache embeddings for frequently used queries

## Migration from RL Rewards

If you're migrating from the RL reward system:

```python
# Old RL way
reward_functions = create_reward_functions(...)
scores = []
for reward_fn in reward_functions:
    score = reward_fn(completions, topic_ids, topic_weights, keywords, doc_id)
    scores.append(score)

# New filtering way
scorer = create_filter_scorer(...)
score = scorer.score_queries(queries, topic_ids=topic_ids, ...)
```

The new system is much simpler and designed for filtering rather than RL training.
