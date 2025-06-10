#!/usr/bin/env bash
# run_topic_modeling.sh
#
# This script sets parameters internally. Edit the variables below as needed.

export CUDA_VISIBLE_DEVICES=2

# Configuration (fill in default values here)
DATASET="CSFCube-1.1" 
CORPUS_PATH="/home/guest/r13944029/IRLab/d2qplus/data/$DATASET/corpus.jsonl"

# Base directory for output files
BASE_OUTPUT_DIR="/home/guest/r13944029/IRLab/d2qplus/augmented-data/CSFCube-1.1/topics/0609-pritamdeka_scibert-biobert-pos-keybert-mmr" 

CHUNK_MODE="sentence"          # Options: "sentence" or "window"
WIN_SIZE=4                     # Window size if CHUNK_MODE="window"
WIN_STEP=2                     # Window step if CHUNK_MODE="window"

# Embedding model to use for topic modeling. Should be carefully choose based on the corpus (e.g., BioBERT for biomedical texts, SciBERT for scientific texts)
# EMBED_MODEL="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
EMBED_MODEL="pritamdeka/S-Scibert-snli-multinli-stsb"  # Embedding model for topic modeling

EMBED_DEVICE="cuda"        # Device: "cpu" or "cuda:0", "cuda:1", etc.
MIN_TOPIC_SIZE=5               # Minimum topic size for BERTopic
TOP_N_WORDS=10                 # Number of top words per topic to extract
N_GRAM_RANGE="1,2"           # N-gram range for topic keywords (format: min,max)
REDUCE_OUTLIERS_MODE="c-tf-idf"  # Options: "c-tf-idf", "distributions", "embeddings"
REDUCE_OUTLIERS_THRESHOLD=50  # Threshold for outlier reduction (if applicable)


# If you want to save the BERTopic model, set --save_topic_model to true

# Run the Python driver with the above parameters
python3 /home/guest/r13944029/IRLab/d2qplus/src/topic-modeling/run_topic_modeling.py \
  --corpus_path "$CORPUS_PATH" \
  --base_output_dir "$BASE_OUTPUT_DIR" \
  --chunk_mode "$CHUNK_MODE" \
  --win_size "$WIN_SIZE" \
  --win_step "$WIN_STEP" \
  --embed_model "$EMBED_MODEL" \
  --embed_device "$EMBED_DEVICE" \
  --min_topic_size "$MIN_TOPIC_SIZE" \
  --top_n_words "$TOP_N_WORDS" \
  --n_gram_range "$N_GRAM_RANGE" \
  --reduce_outliers "$REDUCE_OUTLIERS_MODE" \
  --reduce_outliers_threshold "$REDUCE_OUTLIERS_THRESHOLD"