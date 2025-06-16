# D2Q+ Project Documentation

## 📖 Project Overview

D2Q+ (Document-to-Query Plus) is an advanced document-to-query generation system that combines **topic modeling**, **keyword extraction**, and **reinforcement learning** to generate high-quality pseudo queries for document retrieval enhancement. The system uses BERTopic for topic modeling, KeyBERT for keyword extraction, and various LLMs (including fine-tuned models with RL) for query generation.

## 🏗️ System Architecture

The project consists of four major components:

1. **Topic Modeling & Enhancement** - Identify document topics using BERTopic and enhance with LLM-generated natural language descriptions
2. **Keyword Extraction** - Extract relevant keywords using KeyBERT and topic-aware core phrase extraction  
3. **Query Generation** - Generate pseudo queries using various strategies (topic/keyword-guided, few-shot, etc.)
4. **Evaluation & Analysis** - Comprehensive evaluation framework using PyTerrier with BM25 and dense retrieval

---

## 📂 Directory Structure & Key Files

### `src/` - Core Source Code

#### **Main Entry Points:**
- **`eval.py`** - Main evaluation framework
  - Builds sparse (BM25) and dense (BGE-M3) indexes
  - Runs PyTerrier experiments for retrieval performance
  - Input: JSONL with `id`, `text`, `predicted_queries`
  - Output: Evaluation metrics and index files

- **`generate.py`** - Primary query generation script (most comprehensive)
  - Supports multiple prompt templates and generation strategies
  - Uses VLLM for efficient LLM inference
  - Input: Corpus + topics/keywords
  - Output: Generated queries JSONL

#### **Query Generation Scripts (Multiple Variants):**
- **`few_shot_generate_w_topic_keywords.py`** - Few-shot generation with topics AND keywords
  - Uses topic information and extracted keywords together
  - Supports different prompt configurations (topic-only, keyword-only, no-topic-keywords)
  - Implements token-aware document truncation

- **`few_shot_generate_w_keywords.py`** - Few-shot generation with keywords only
  - Uses extracted keywords without topic information
  - Simpler pipeline for keyword-focused query generation

- **`generate_w_keywords.py`** - Direct keyword-guided generation
  - Uses KeyBERT extracted keywords for query generation
  - No few-shot examples, direct prompting approach

- **`promptagator.py`** - Complete Promptagator method implementation
  - Dynamic few-shot example selection based on token limits
  - Auto-adjusts number of examples to fit model context
  - Token analysis and safety buffer management

- **`generate_t5.py`** - T5-based baseline generation
  - Uses Doc2Query-T5 models for query generation
  - Configurable batch processing and output length

#### **Analysis & Utility Scripts:**
- **`analyze_promptagator_tokens.py`** - Token usage analysis for Promptagator
  - Analyzes few-shot example token consumption
  - Provides recommendations for optimal example selection
  - Statistical analysis of document and prompt token sizes

- **`analyze_token_sizes.py`** - Comprehensive token analysis
  - Analyzes document token distributions across different prompt templates
  - Visualization of token usage patterns
  - Supports multiple model tokenizers and prompt formats

- **`scoring.py`** - Advanced query scoring and selection system
  - QuerySelector class with multiple scoring metrics
  - Topic coverage, keyword inclusion, diversity scoring
  - Query set optimization with configurable weights

- **`dataset_splitter.py`** - Dataset creation and management
  - Creates small corpus with hard negatives for testing
  - Handles query and qrels sampling
  - Supports BM25-based hard negative mining

- **`visualize_topic_keywords.py`** - Topic and keyword visualization
  - Generates example documents with topics and keywords
  - Creates markdown reports for manual inspection

#### **Reinforcement Learning:**
- **`grpo_trainer.py`** - GRPO (Generalized Reward Preference Optimization) trainer
  - Fine-tunes LLMs using multiple reward functions
  - Supports topic coverage, keyword inclusion, and diversity rewards
  - Integration with Weights & Biases for logging

- **`grpo_trainer_unsloth.py`** - Optimized GRPO trainer with Unsloth
  - Memory-efficient training with Unsloth integration
  - LoRA fine-tuning configuration
  - Faster training for resource-constrained environments

- **`trainer.py`** - PPO-based trainer (alternative RL approach)
  - Traditional PPO implementation for query generation
  - Coverage reward model integration

- **`reward.py`** - Comprehensive reward function system
  - Multiple reward types: TopicCoverageReward, KeywordInclusionReward, etc.
  - Shared computation cache for efficiency
  - Configurable reward weights and combinations

#### **Topic Modeling (`src/topic-modeling/`):**
- **`run_topic_modeling.py`** - Main BERTopic pipeline
  - Document chunking (sentence-level or sliding windows)
  - Topic modeling with UMAP + HDBSCAN clustering
  - Output: `doc_topics.jsonl`, `topic_info_dataframe.pkl`

- **`extract_core_phrases.py`** - Advanced keyword extraction using CCQGen
  - Implements CCQGen methodology for distinctiveness scoring
  - Uses BM25 and topic-aware ranking
  - Input: Corpus + topic assignments
  - Output: Core phrases JSONL

- **`extract_keywords_keybert.py`** - Basic KeyBERT keyword extraction
  - Configurable n-gram ranges and diversity parameters
  - MMR (Maximal Marginal Relevance) for diversity
  - Saves candidate keywords as pickle files

- **`grid_search_topic_modeling.py`** - Grid search optimization for topic modeling
  - Systematic parameter tuning for BERTopic
  - Multiple parameter combinations testing
  - Performance evaluation and comparison

- **`keywords_as_pq.py`** - Use topic keywords directly as pseudo-queries
  - Simple baseline using topic keywords as generated queries
  - Configurable number of keywords per topic

- **`llm_extracted_keywords_as_pq.py`** - Use LLM-extracted keywords as pseudo-queries
  - Alternative baseline using LLM-generated keywords
  - Direct keyword-to-query conversion

- **`visualize_topic_and_analyze.py`** - Topic distribution analysis and visualization
  - Statistical analysis of topic assignments
  - Visualization of topic distributions
  - Multi-directory comparison support

- **`dvdo.py`** - DVDO algorithm implementation
  - Alternative topic modeling approach
  - Grid search optimization for clustering parameters

#### **Utilities (`src/utils/`):**
- **`data.py`** - Data processing and prompt preparation
- **`constants.py`** - System prompts and templates
- **`llm_extract_keywords.py`** - LLM-based keyword extraction with JSON schema
- **`get_llm_representation.py`** - Topic enhancement with LLM descriptions
- **`query_scorer.py`** - Modular query scoring functions
  - Self-BLEU diversity scoring
  - Topic coverage calculation
  - Keyword inclusion metrics
- **`append_multiple_set_genq.py`** - Merge multiple query generation outputs
- **`append_multiple_t5_genq.py`** - Merge T5-specific query outputs  
- **`doc_minus_minus_for_t5_genq.py`** - Filter top-k% T5 queries (Doc2Query--)
- **`analyze_topic_keyword_eval_result.py`** - Evaluation result analysis across topics
- **`deepseek.py`** - DeepSeek API integration for LLM calls
- **`text_only.py`** - Generate text-only baselines (no pseudo-queries)

### `scripts/` - Execution Scripts

#### **Topic Modeling (`scripts/topic-modeling/`):**
- **`topic_modeling.sh`** - Complete topic modeling pipeline
- **`get_llm_representation.sh`** - Enhance topics with LLM descriptions
- **`extract_core_phrases.sh`** - Extract distinctive keyphrases
- **`extract_keywords_keybert.sh`** - KeyBERT keyword extraction
- **`llm_extract_keywords.sh`** - LLM-based keyword extraction
- **`all_in_one.sh`** - End-to-end topic modeling + keyword extraction
- **`grid_topic_modeling.sh`** - Grid search for optimal parameters
- **`visualize_and_analyze_topics.sh`** - Topic analysis and visualization

#### **Query Generation (`scripts/query-gen/`):**
- **`d2q_gen_vllm.sh`** - Main VLLM-based query generation (supports multiple job types)
- **`d2q_gen_vllm_gary.sh`** - Alternative VLLM generation script with different configurations
- **`few_shot_gen_w_topic_keywords.sh`** - Few-shot generation with topics + keywords
- **`few_shot_gen_w_keywords.sh`** - Few-shot generation with keywords only
- **`promptagator_gen.sh`** - Promptagator method execution
- **`d2q_gen_t5.sh`** - T5-based baseline generation
- **`doc_minus_minus_for_t5.sh`** - T5 query filtering (Doc2Query--)
- **`multi_setting_gen.sh`** - Multiple generation settings in one script
- **`multi_setting_gen_topics.sh`** - Multiple topic-based generation settings
- **`append_multiple_set_genq.sh`** - Merge multiple generation outputs
- **`score_queries.sh`** - Score generated queries using ELECTRA
- **`text_only.sh`** - Generate text-only baselines
- **`text_add_topic_keywords.sh`** - Add topic keywords to text
- **`text_add_llm_extracted_keywords.sh`** - Add LLM keywords to text
- **`text_add_llm_extracted_keywords_batch.sh`** - Batch LLM keyword addition

#### **Training & Evaluation:**
- **`grpo_trainer.sh`** - RL training script
- **`run_rl_trainer.sh`** - Alternative RL training
- **`run_promptagator.sh`** - Run Promptagator experiments
- **`eval/eval.sh`** - Main evaluation pipeline
- **`eval/eval_from_topic_dir.sh`** - Evaluate from specific topic directory
- **`eval/eval_from_all_topic_dirs.sh`** - Evaluate across multiple topic directories
- **`eval/analyze_topic_eval_result.sh`** - Analyze evaluation results
- **`zero_shot.sh`** - Zero-shot evaluation experiments
- **`analyze_token_sizes.sh`** - Token analysis script
- **`score_query_set.sh`** - Score query sets with multiple metrics

#### **VLLM & API Services:**
- **`vllm_serve.sh`** - Start VLLM inference server
- **`trl_vllm_serve.sh`** - TRL-compatible VLLM server

### `experiments/` - Research & Analysis

#### **Baseline Models & Filtering:**
- **`zero_shot.py`** - Zero-shot query generation experiments
- **`filter_tune_and_test.py`** - Query filtering and BM25 parameter tuning
- **`filter.py`** - Query filtering utilities
- **`score_generator.py`** - Generate relevance scores for queries using ELECTRA

#### **Advanced Features:**
- **`reranking/`** - Neural reranking experiments
  - **`rerank.sh`** - MonoT5 reranking script
  - **`README.md`** - Reranking setup instructions
- **`lsr/`** - Learned sparse retrieval (uniCOIL, DeepImpact)
  - Multiple indexing and query scripts for sparse retrieval
- **`rl/`** - Reinforcement learning experiments (PPO, TRL)
  - **`trl_ppo_d2q.py`** - TRL-based PPO training
  - **`run.py`** - RL model evaluation and generation
- **`analysis/`** - Result analysis and visualization tools
- **`test_rerank.py`** - Testing neural reranking pipelines

### `prompts/` - LLM Templates & Few-shot Examples
- **`user_prompt_template.txt`** - Base query generation template
- **`topic-modeling/`** - Topic enhancement prompts for different datasets
- **`promptagator/`** - Promptagator method templates and few-shot examples
- **`plan-then-write/`** - Multi-step generation prompts
- **`get_few_shot_prompt.ipynb`** - Notebook for creating few-shot examples

### `playground/` - Development & Analysis
- **Jupyter notebooks** for data exploration, topic analysis, and reward testing
- **`topic-modeling/`** - Topic modeling experiments and visualizations
  - **`topic_modeling.ipynb`** - Main topic modeling exploration
  - **`keyword_extract.ipynb`** - Keyword extraction experiments
  - **`topic_coverage_experiment.ipynb`** - Topic coverage analysis
  - **`visualize.ipynb`** - Topic visualization
  - **`dataset_exploring.ipynb`** - Dataset analysis
  - **`lda.ipynb`** - LDA topic modeling comparison
- **`rewards.ipynb`** - Reward function development and testing
- **`querygen.ipynb`** - Query generation experiments
- **`chat_with_llm.ipynb`** - LLM interaction testing

### `data/` - Datasets
- **Multiple IR datasets**: NFCorpus, SciFact, FIQA, MS MARCO, TREC-CAR, CSFCube
- **Standard format**: `corpus.jsonl`, `queries.jsonl`, `qrels/test.trec`
- **Processing utilities**: `utils.py`, `process.ipynb`

### `helper/` - PyTerrier Integration & Evaluation
- **`evaluation.py`** - Comprehensive evaluation functions
- **`util.py`** - Utility functions for indexing and filtering
- **`preprocessing.py`** - Data preprocessing utilities
- **`retrieve_rerank.py`** - Retrieval and reranking pipelines

---

## 🚀 Complete Workflows

### 1. Topic Modeling Pipeline

```bash
# Step 1: Basic topic modeling
cd scripts/topic-modeling
./topic_modeling.sh

# Step 2: Enhance topics with LLM descriptions
./get_llm_representation.sh

# Step 3: Extract distinctive keyphrases using CCQGen
./extract_core_phrases.sh

# Step 4: Extract keywords using KeyBERT
./extract_keywords_keybert.sh

# Step 5: Extract keywords using LLM
./llm_extract_keywords.sh

# Alternative: Run all steps together
./all_in_one.sh

# Grid search for optimal parameters
./grid_topic_modeling.sh

# Analyze and visualize results
./visualize_and_analyze_topics.sh
```

**Outputs:**
- `doc_topics.jsonl` - Document-topic assignments with weights
- `topic_info_dataframe_enhanced.pkl` - Enhanced topic information with LLM descriptions
- `keywords.jsonl` - CCQGen extracted core phrases per document
- `candidate_keywords_*.pkl` - KeyBERT extracted candidate keywords
- `llm_extracted_keywords.jsonl` - LLM-extracted keywords per document

### 2. Query Generation Workflows

#### A. Few-shot with Topics + Keywords (Latest Approach)
```bash
cd scripts/query-gen
# Configure dataset and parameters in the script
./few_shot_gen_w_topic_keywords.sh
```

#### B. Few-shot with Keywords Only
```bash
./few_shot_gen_w_keywords.sh
```

#### C. Promptagator Method
```bash
./promptagator_gen.sh
# Or using the dedicated promptagator script:
../run_promptagator.sh
```

#### D. VLLM-based Generation (Multiple Job Types)
```bash
# Configure job types in d2q_gen_vllm.sh
export JOBS_TO_RUN="base_with_topic base_without_topic d2q-fewshot-topics promptagator"
./d2q_gen_vllm.sh

# Alternative configuration
./d2q_gen_vllm_gary.sh
```

#### E. T5 Baseline Generation
```bash
./d2q_gen_t5.sh
```

#### F. Doc2Query-- (Filtered T5)
```bash
# First generate T5 queries
./d2q_gen_t5.sh

# Score the queries
./score_queries.sh

# Filter top k% queries
./doc_minus_minus_for_t5.sh
```

#### G. Multi-setting Generation
```bash
# Run multiple configurations in sequence
./multi_setting_gen.sh
./multi_setting_gen_topics.sh
```

#### H. Merge Multiple Outputs
```bash
# Combine multiple generation runs
./append_multiple_set_genq.sh
```

### 3. Reinforcement Learning Training

```bash
# Method 1: GRPO Training (Recommended)
./scripts/grpo_trainer.sh

# Method 2: Alternative RL Training
./scripts/run_rl_trainer.sh

# Monitor training via Weights & Biases
# Use trained model for generation:
export MODEL="/path/to/trained/checkpoint"
./scripts/query-gen/d2q_gen_vllm.sh
```

### 4. Evaluation Pipeline

#### A. Single Dataset Evaluation
```bash
cd scripts/eval
# Configure dataset and query names in eval.sh
./eval.sh
```

#### B. Topic-based Evaluation
```bash
# Evaluate from specific topic directory
./eval_from_topic_dir.sh

# Evaluate across all topic directories
./eval_from_all_topic_dirs.sh

# Analyze evaluation results
./analyze_topic_eval_result.sh
```

#### C. Advanced Filtering Experiments
```bash
# Zero-shot evaluation with filtering
../zero_shot.sh

# Comprehensive filtering and tuning
cd ../../
python experiments/filter_tune_and_test.py \
    --scored_file /path/to/scored_queries.jsonl \
    --index_dir /path/to/indexes \
    --eval_dir /path/to/results
```

#### D. Query Scoring
```bash
# Score query sets with multiple metrics
./scripts/score_query_set.sh

# Generate ELECTRA scores
./scripts/query-gen/score_queries.sh
```

### 5. Analysis & Visualization

#### A. Token Analysis
```bash
# Analyze token distributions and prompt sizes
./scripts/analyze_token_sizes.sh

# Specific Promptagator token analysis
python src/analyze_promptagator_tokens.py
```

#### B. Topic Analysis
```bash
# Visualize topics and keywords
python src/visualize_topic_keywords.py

# Comprehensive topic analysis
python src/topic-modeling/visualize_topic_and_analyze.py
```

#### C. Result Analysis
```bash
# Analyze evaluation results across topics
python src/utils/analyze_topic_keyword_eval_result.py
```

---

## 🔧 Key Configuration Files

### Topic Modeling Configuration
- **Embedding Models**: Choose based on domain (SciBERT for scientific, FinBERT for financial)
- **Chunk Mode**: `sentence` vs `window` for document processing
- **Clustering**: Min topic size, UMAP dimensions, HDBSCAN parameters

### Query Generation Configuration
- **Models**: Meta-Llama, T5, trained checkpoints
- **Prompting**: Template selection, few-shot examples, max keywords/topics
- **Sampling**: Temperature, top-k, number of queries per document

### Evaluation Configuration
- **Datasets**: NFCorpus, SciFact, FIQA, MS MARCO, TREC-CAR
- **Metrics**: BM25 parameters, dense retrieval models (BGE-M3)
- **Filtering**: Top-k% query selection thresholds

---

## 📊 Input/Output Formats

### Core Data Formats

#### Corpus (Input)
```json
{"_id": "doc123", "text": "Document content...", "title": "Document Title"}
```

#### Topic Assignments
```json
{"doc_id": "doc123", "topics": [{"topic_id": 5, "weight": 0.75}, {"topic_id": 12, "weight": 0.25}]}
```

#### Generated Queries (Output)
```json
{"id": "doc123", "title": "Document Title", "text": "Document content...", "predicted_queries": ["query1", "query2", "query3"]}
```

#### Enhanced Topic Information
```python
# Pickle format: topic_info_dataframe_enhanced.pkl
{
    "Topic": [0, 1, 2, ...],
    "Representation": [["keyword1", "keyword2"], ...],
    "Enhanced_Topic": ["Natural language description", ...]
}
```

---

## 📚 Key Dependencies

### Core Libraries
- **BERTopic** - Topic modeling framework
- **KeyBERT** - Keyword extraction
- **VLLM** - Efficient LLM inference
- **PyTerrier** - Information retrieval evaluation
- **TRL** - Reinforcement learning for LLMs

### Model Dependencies
- **Sentence Transformers** - Document embeddings
- **Transformers** - HuggingFace model integration
- **UMAP + HDBSCAN** - Dimensionality reduction and clustering

---

## 🔧 Detailed Configuration Examples

### Complete Query Generation Configuration

```yaml
# config.yaml - Comprehensive configuration file
model:
  name: "microsoft/DialoGPT-medium"
  temperature: 0.7
  max_tokens: 150
  top_p: 0.9
  frequency_penalty: 0.1
  presence_penalty: 0.1

query_generation:
  batch_size: 32
  num_queries: 100
  max_length: 512
  seed: 42
  use_keywords: true
  keyword_count: 5
  use_topic_keywords: true
  topic_threshold: 0.3

keywords:
  method: "keybert"  # or "yake", "tfidf"
  top_k: 20
  mmr_diversity: 0.7
  similarity_threshold: 0.5
  use_stemming: true

topic_modeling:
  n_topics: 50
  min_topic_size: 10
  embedding_model: "all-MiniLM-L6-v2"
  umap_n_neighbors: 15
  umap_n_components: 5
  hdbscan_min_cluster_size: 10

evaluation:
  metrics: ["ndcg@10", "map", "recall@100"]
  index_type: "bm25"  # or "dense"
  rerank: false
  batch_size: 1000

paths:
  data_dir: "data/"
  output_dir: "outputs/"
  models_dir: "models/"
  cache_dir: "cache/"
```

### Environment Configuration

```bash
# .env file for environment variables
CUDA_VISIBLE_DEVICES=0,1
JAVA_HOME=/usr/lib/jvm/java-11-openjdk
JVM_PATH=/usr/lib/jvm/java-11-openjdk/lib/server/libjvm.so
TRANSFORMERS_CACHE=/path/to/cache
HF_HOME=/path/to/huggingface/cache
WANDB_PROJECT=d2qplus
WANDB_ENTITY=your_entity
```

### Advanced Model Configuration

```python
# model_config.py - Advanced model settings
VLLM_CONFIG = {
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
    "trust_remote_code": True,
    "dtype": "float16",
    "quantization": "awq"
}

GENERATION_CONFIG = {
    "temperature": [0.5, 0.7, 0.9],
    "top_p": [0.8, 0.9, 0.95],
    "max_tokens": [100, 150, 200],
    "repetition_penalty": [1.0, 1.05, 1.1]
}

KEYWORD_CONFIG = {
    "keybert": {
        "model": "all-MiniLM-L6-v2",
        "keyphrase_ngram_range": (1, 3),
        "stop_words": "english",
        "use_maxsum": True,
        "use_mmr": True,
        "diversity": 0.7
    },
    "yake": {
        "lan": "en",
        "n": 3,
        "dedupLim": 0.7,
        "top": 20
    }
}
```

---

## 📋 Complete Input/Output Formats

### Input Data Formats

#### Document Collections
```json
// JSONL format for document collections
{"docid": "doc_001", "text": "Document content here...", "title": "Document Title"}
{"docid": "doc_002", "text": "Another document...", "title": "Another Title"}
```

#### TSV Format (PyTerrier compatible)
```tsv
docno	text	title
doc_001	Document content here...	Document Title
doc_002	Another document...	Another Title
```

#### Qrels (Relevance Judgments)
```tsv
query_id	iteration	docno	relevance
q001	0	doc_001	2
q001	0	doc_002	1
q002	0	doc_003	3
```

#### Topics/Queries
```tsv
qid	query
q001	machine learning algorithms
q002	natural language processing
```

### Keyword Files
```json
// keywords.json
{
  "doc_001": ["machine learning", "algorithms", "neural networks"],
  "doc_002": ["natural language", "processing", "text analysis"]
}
```

### Topic Keywords Format
```json
// topic_keywords.json
{
  "topic_0": {
    "keywords": ["machine learning", "ai", "neural"],
    "weight": 0.85,
    "documents": ["doc_001", "doc_003"]
  },
  "topic_1": {
    "keywords": ["nlp", "text", "language"],
    "weight": 0.72,
    "documents": ["doc_002", "doc_004"]
  }
}
```

### Output Formats

#### Generated Queries
```json
// generated_queries.jsonl
{"query_id": "gen_001", "query": "What are the latest machine learning algorithms?", "source_doc": "doc_001", "keywords": ["machine learning", "algorithms"], "confidence": 0.87}
{"query_id": "gen_002", "query": "How does natural language processing work?", "source_doc": "doc_002", "keywords": ["nlp", "processing"], "confidence": 0.92}
```

#### Evaluation Results
```json
// evaluation_results.json
{
  "overall_metrics": {
    "ndcg@10": 0.654,
    "map": 0.421,
    "recall@100": 0.789
  },
  "per_query": {
    "gen_001": {"ndcg@10": 0.67, "map": 0.45},
    "gen_002": {"ndcg@10": 0.62, "map": 0.38}
  },
  "metadata": {
    "num_queries": 1000,
    "num_docs": 50000,
    "index_type": "bm25"
  }
}
```

#### Topic Modeling Output
```json
// topic_model_output.json
{
  "topics": {
    "0": {
      "words": [["machine", 0.1], ["learning", 0.08], ["ai", 0.06]],
      "size": 1250,
      "representative_docs": ["doc_001", "doc_015"]
    }
  },
  "document_topics": {
    "doc_001": [{"topic": 0, "probability": 0.85}],
    "doc_002": [{"topic": 1, "probability": 0.72}]
  }
}
```

#### Training Logs
```json
// training_log.jsonl
{"epoch": 1, "step": 100, "loss": 2.34, "reward": 0.67, "kl_div": 0.12, "lr": 5e-5}
{"epoch": 1, "step": 200, "loss": 2.12, "reward": 0.71, "kl_div": 0.15, "lr": 4.8e-5}
```