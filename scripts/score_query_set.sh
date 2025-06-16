#!/bin/bash

# Set script variables
TOPIC_DIR="/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/neighbors_10_components_10_mintopic_10"
CORPUS_PATH="/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl"
QUERIES_FILE="/home/guest/r12922050/GitHub/d2qplus/generated-queries/nfcorpus/docTTTTTquery_generated_queries.jsonl"
OUTPUT_FILE="/home/guest/r12922050/GitHub/d2qplus/selected-queries/nfcorpus/scored_queries.jsonl"

# Model parameters
EMBED_MODEL="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
DEVICE="cuda"

# Selection parameters
NUM_QUERY_SET=5
SOFTMAX_TAU=0.07
SIMILARITY_THRESHOLD=0.7

# Scoring weights
LAMBDA_TC=1.0
LAMBDA_KW=1.0
LAMBDA_DIV=1.0
LAMBDA_RELEVANCE=1.0
TOPIC_COVERAGE_METRIC="f1"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run the scoring script
python /home/guest/r12922050/GitHub/d2qplus/src/scoring.py \
    --topic-dir "$TOPIC_DIR" \
    --corpus-path "$CORPUS_PATH" \
    --queries-file "$QUERIES_FILE" \
    --output-file "$OUTPUT_FILE" \
    --embed-model "$EMBED_MODEL" \
    --device "$DEVICE" \
    --num-query-set "$NUM_QUERY_SET" \
    --softmax-tau "$SOFTMAX_TAU" \
    --similarity-threshold "$SIMILARITY_THRESHOLD" \
    --lambda-tc "$LAMBDA_TC" \
    --lambda-kw "$LAMBDA_KW" \
    --lambda-div "$LAMBDA_DIV" \
    --lambda-relevance "$LAMBDA_RELEVANCE" \
    --topic-coverage-metric "$TOPIC_COVERAGE_METRIC"

echo "🎉 Query scoring completed! Results saved to: $OUTPUT_FILE"
