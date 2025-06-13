#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# Configuration (fill in default values here)
DATASET="fiqa" 
CORPUS_PATH="/home/guest/r12922050/GitHub/d2qplus/data/$DATASET/corpus.jsonl"

# Base directory for output files
PROJECT_BASE_DIR="/home/guest/r12922050/GitHub/d2qplus"

CHUNK_MODE="sentence"          # Options: "sentence" or "window"
WIN_SIZE=2                     # Window size if CHUNK_MODE="window"
WIN_STEP=2                     # Window step if CHUNK_MODE="window"


# Embedding model to use for topic modeling. Should be carefully choose based on the corpus (e.g., BioBERT for biomedical texts, SciBERT for scientific texts)
# EMBED_MODEL="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
# EMBED_MODEL="pritamdeka/S-Scibert-snli-multinli-stsb"
EMBED_MODEL="all-mpnet-base-v2"

EMBED_DEVICE="cuda"        # Device: "cpu" or "cuda:0", "cuda:1", etc.
MIN_TOPIC_SIZE=5              # Minimum topic size for BERTopic
TOP_N_WORDS=10                 # Number of top words per topic to extract
N_GRAM_RANGE="1,2"           # N-gram range for topic keywords (format: min,max)

TOPIC_NAME="0612-${EMBED_MODEL}-topic-size-${MIN_TOPIC_SIZE}"
BASE_OUTPUT_DIR="${PROJECT_BASE_DIR}/augmented-data/$DATASET/topics/${TOPIC_NAME}" 

############## Keyword Extraction Parameters ##############

TOP_N_CANDIDATES=20
SELECTION_RATIO=0.3
MIN_PHRASES=5
MAX_PHRASES=10
MIN_NGRAM=1
MAX_NGRAM=3
USE_MMR=true
DIVERSITY=0.6

###############


# If you want to save the BERTopic model, set --save_topic_model to true
conda activate bertopic

python3 /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/run_topic_modeling.py \
  --corpus_path "$CORPUS_PATH" \
  --base_output_dir "$BASE_OUTPUT_DIR" \
  --chunk_mode "$CHUNK_MODE" \
  --win_size "$WIN_SIZE" \
  --win_step "$WIN_STEP" \
  --embed_model "$EMBED_MODEL" \
  --embed_device "$EMBED_DEVICE" \
  --min_topic_size "$MIN_TOPIC_SIZE" \
  --top_n_words "$TOP_N_WORDS" \
  --n_gram_range "$N_GRAM_RANGE"


conda deactivate
conda activate d2qplus
# Extract core phrases for each topic
python /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/extract_core_phrases.py \
    --corpus_path "$CORPUS_PATH" \
    --doc_topics_path "$BASE_OUTPUT_DIR/doc_topics.jsonl" \
    --output_path "$BASE_OUTPUT_DIR/keywords.jsonl" \
    --embedding_model "$EMBED_MODEL" \
    --device "$DEVICE" \
    --top_n_candidates "$TOP_N_CANDIDATES" \
    --selection_ratio "$SELECTION_RATIO" \
    --min_phrases "$MIN_PHRASES" \
    --max_phrases "$MAX_PHRASES" \
    --min_ngram "$MIN_NGRAM" \
    --max_ngram "$MAX_NGRAM" \
    --use_mmr \
    --diversity "$DIVERSITY" \
    --show_examples

# Get LLM representation for topics

python3 /home/guest/r12922050/GitHub/d2qplus/src/utils/get_llm_representation.py \
    --few_shot_prompt_txt_path "/home/guest/r12922050/GitHub/d2qplus/prompts/topic-modeling/enhance_NL_topic_${DATASET}.txt" \
    --topic_base_dir "$BASE_OUTPUT_DIR"