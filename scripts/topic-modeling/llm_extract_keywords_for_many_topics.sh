export CUDA_VISIBLE_DEVICES=0,1

MODEL=meta-llama/Llama-3.1-8B-Instruct
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=3000
FINAL_EXTRACT_KEYWORDS=10
MAX_TOKENS=256
GPU_MEMORY_UTILIZATION=0.8
BASE_TOPICS_DIR="/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics"
CORPUS_PATH="/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl"

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
    TOPIC_INFO_PKL="$TOPIC_DIR/topic_info_dataframe.pkl"
    DOC_TOPICS_PATH="$TOPIC_DIR/doc_topics.jsonl"
    OUTPUT_PATH="$TOPIC_DIR/llm_extracted_keywords.txt"
    
    # Check if output already exists
    if [ -f "$OUTPUT_PATH" ]; then
        echo "Warning: llm_extracted_keywords.txt already exists for $topic_name"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping $topic_name"
            continue
        fi
    fi
    
    # Run extraction
    echo "Running core phrase extraction for $topic_name..."
    
    python3 -m src.utils.llm_extract_keywords \
        --topic_info_pkl $TOPIC_INFO_PKL \
        --keywords_path /home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/keywords/candidate_keywords_scibert.pkl \
        --corpus_topics_path $DOC_TOPICS_PATH \
        --corpus_path $CORPUS_PATH \
        --output_path $OUTPUT_PATH \
        --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
        --model $MODEL \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --max_model_len $MAX_MODEL_LEN \
        --max_tokens $MAX_TOKENS \
        --final_extract_keywords_num $FINAL_EXTRACT_KEYWORDS
    
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
