#!/usr/bin/env bash
# run_topic_modeling.sh
#
# This script sets parameters internally. Edit the variables below as needed.

export CUDA_VISIBLE_DEVICES=7

# Configuration (fill in default values here)
CORPUS_PATH="/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl"

BASE_OUTPUT_DIR="/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0605-biobert" # Base directory for output files

CHUNK_MODE="sentence"         # Options: "sentence" or "window"
WIN_SIZE=4                     # Window size if CHUNK_MODE="window"
WIN_STEP=2                     # Window step if CHUNK_MODE="window"
EMBED_MODEL="dmis-lab/biobert-v1.1"  # SentenceTransformer model name
EMBED_DEVICE="cuda"        # Device: "cpu" or "cuda:0", "cuda:1", etc.
MIN_TOPIC_SIZE=6               # Minimum topic size for BERTopic
TOP_N_WORDS=20                 # Number of top words per topic to extract
N_GRAM_RANGE="1,2"           # N-gram range for topic keywords (format: min,max)

# Run the Python driver with the above parameters
python3 /home/guest/r12922050/GitHub/d2qplus/src/run_topic_modeling.py \
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
