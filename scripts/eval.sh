#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export JAVA_HOME="$CONDA_PREFIX"
export JVM_PATH="$CONDA_PREFIX/lib/server/libjvm.so"

# Variables for better control
BASE_DIR="/home/guest/r12922050/GitHub/d2qplus"
DATASET="nfcorpus"

# List of query names to iterate through
GEN_QUERY_NAMES=(
    "Llama-3.2-1B-Instruct-promptagator"
)

# Iterate through each query name
for GEN_QUERY_NAME in "${GEN_QUERY_NAMES[@]}"; do
    echo "=========================================="
    echo "Currently running evaluation for: $GEN_QUERY_NAME"
    echo "=========================================="
    
    python3 $BASE_DIR/src/eval.py \
        --overwrite-aug-dense-index \
        --corpus "$BASE_DIR/gen/$DATASET/$GEN_QUERY_NAME.jsonl" \
        --queries $BASE_DIR/data/$DATASET/queries.jsonl \
        --qrels $BASE_DIR/data/$DATASET/qrels/test.trec \
        --index-base-dir $BASE_DIR/built-index/$DATASET \
        --index-name $GEN_QUERY_NAME \
        --output $BASE_DIR/eval/$DATASET/$GEN_QUERY_NAME.csv
    echo "Completed evaluation for: $GEN_QUERY_NAME"
    echo ""
done

echo "All evaluations completed!"