#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
# Path parameters
DATASET="nfcorpus"
TOPIC_NAME=0612-topic-size-10
CORPUS_PATH="/home/guest/r12922050/GitHub/d2qplus/data/${DATASET}/corpus.jsonl"
OUTPUT_PATH="/home/guest/r12922050/GitHub/d2qplus/augmented-data/${DATASET}/topics/${TOPIC_NAME}/keywords.jsonl"

DOC_TOPICS_PATH="/home/guest/r12922050/GitHub/d2qplus/augmented-data/${DATASET}/topics/${TOPIC_NAME}/doc_topics.jsonl"

# Model parameters
EMBEDDING_MODEL="pritamdeka/S-Scibert-snli-multinli-stsb"
# EMBEDDING_MODEL="all-mpnet-base-v2"
DEVICE="cuda"

# Extraction parameters
TOP_N_CANDIDATES=50
SELECTION_RATIO=0.2
MIN_PHRASES=5
MAX_PHRASES=10
MIN_NGRAM=1
MAX_NGRAM=3
USE_MMR=true
DIVERSITY=0.6

# Run the Python script with all parameters
python /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/extract_core_phrases.py \
    --corpus_path "$CORPUS_PATH" \
    --doc_topics_path "$DOC_TOPICS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --embedding_model "$EMBEDDING_MODEL" \
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
