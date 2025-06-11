#!/bin/bash
# Promptagator query generation script

# Default paths - modify these according to your setup
ENHANCED_TOPIC_INFO_PKL="/path/to/enhanced_topic_info.pkl"
CORPUS_PATH="/path/to/corpus.jsonl"
CORPUS_TOPICS_PATH="/path/to/corpus_topics.jsonl"
OUTPUT_PATH="./promptagator_queries.jsonl"

# Model settings
MODEL="meta-llama/Llama-3.2-1B-Instruct"
MAX_MODEL_LEN=8192
TARGET_QUERIES=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_FLAG="--test"
            echo "🧪 Running in test mode"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --target_queries)
            TARGET_QUERIES="$2"
            shift 2
            ;;
        --corpus_path)
            CORPUS_PATH="$2"
            shift 2
            ;;
        --enhanced_topic_info_pkl)
            ENHANCED_TOPIC_INFO_PKL="$2"
            shift 2
            ;;
        --corpus_topics_path)
            CORPUS_TOPICS_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --test                        Run in test mode (10 docs)"
            echo "  --model MODEL                 Model to use (default: $MODEL)"
            echo "  --output PATH                 Output path (default: $OUTPUT_PATH)"
            echo "  --target_queries N            Queries per doc (default: $TARGET_QUERIES)"
            echo "  --corpus_path PATH            Corpus path"
            echo "  --enhanced_topic_info_pkl PATH Enhanced topic info path"
            echo "  --corpus_topics_path PATH     Corpus topics path"
            echo "  -h, --help                    Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Starting Promptagator with:"
echo "  Model: $MODEL"
echo "  Target queries per doc: $TARGET_QUERIES"
echo "  Output: $OUTPUT_PATH"

python src/promptagator.py \
    --enhanced_topic_info_pkl "$ENHANCED_TOPIC_INFO_PKL" \
    --corpus_path "$CORPUS_PATH" \
    --corpus_topics_path "$CORPUS_TOPICS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model "$MODEL" \
    --max_model_len "$MAX_MODEL_LEN" \
    --target_queries_per_doc "$TARGET_QUERIES" \
    --auto_select_examples \
    $TEST_FLAG

echo "✅ Promptagator generation complete!"
