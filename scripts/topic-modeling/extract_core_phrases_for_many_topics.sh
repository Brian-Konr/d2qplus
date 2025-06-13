#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# Base parameters
DATASET="nfcorpus"
BASE_TOPICS_DIR="/home/guest/r12922050/GitHub/d2qplus/augmented-data/${DATASET}/topics"
CORPUS_PATH="/home/guest/r12922050/GitHub/d2qplus/data/${DATASET}/corpus.jsonl"

# Model parameters
EMBEDDING_MODEL="pritamdeka/S-Scibert-snli-multinli-stsb"
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

echo "Scanning for topic directories with write permission in: $BASE_TOPICS_DIR"

# Find all topic directories with write permission
topic_dirs=()
if [ -d "$BASE_TOPICS_DIR" ]; then
    for topic_dir in "$BASE_TOPICS_DIR"/*; do
        if [ -d "$topic_dir" ] && [ -w "$topic_dir" ]; then
            # Check if required files exist
            doc_topics_file="$topic_dir/doc_topics.jsonl"
            if [ -f "$doc_topics_file" ]; then
                topic_name=$(basename "$topic_dir")
                topic_dirs+=("$topic_name")
                echo "✓ Found writable topic dir: $topic_name"
            else
                echo "✗ Skipping $topic_dir: missing doc_topics.jsonl"
            fi
        fi
    done
else
    echo "Error: Base topics directory not found: $BASE_TOPICS_DIR"
    exit 1
fi

echo "Found ${#topic_dirs[@]} topic directories to process"

if [ ${#topic_dirs[@]} -eq 0 ]; then
    echo "No valid topic directories found!"
    exit 1
fi

# Process each topic directory
for topic_name in "${topic_dirs[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing topic: $topic_name"
    echo "=========================================="
    
    # Set paths for this topic
    TOPIC_DIR="$BASE_TOPICS_DIR/$topic_name"
    DOC_TOPICS_PATH="$TOPIC_DIR/doc_topics.jsonl"
    OUTPUT_PATH="$TOPIC_DIR/keywords.jsonl"
    
    # Check if output already exists
    if [ -f "$OUTPUT_PATH" ]; then
        echo "Warning: keywords.jsonl already exists for $topic_name"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping $topic_name"
            continue
        fi
    fi
    
    # Run extraction
    echo "Running core phrase extraction for $topic_name..."
    
    python /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/extract_core_phrases.py \
        --corpus_path "$CORPUS_PATH" \
        --doc_topics_path "$DOC_TOPICS_PATH" \
        --candidate_keywords_path /home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/keywords/candidate_keywords_scibert.pkl \
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
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $topic_name"
    else
        echo "✗ Failed to process $topic_name"
    fi
done

echo ""
echo "=========================================="
echo "Batch processing completed!"
echo "=========================================="

# Summary
success_count=0
for topic_name in "${topic_dirs[@]}"; do
    output_file="$BASE_TOPICS_DIR/$topic_name/keywords.jsonl"
    if [ -f "$output_file" ]; then
        success_count=$((success_count + 1))
        echo "✓ $topic_name: $(wc -l < "$output_file") documents processed"
    else
        echo "✗ $topic_name: failed"
    fi
done

echo "Summary: $success_count/${#topic_dirs[@]} topic directories processed successfully"