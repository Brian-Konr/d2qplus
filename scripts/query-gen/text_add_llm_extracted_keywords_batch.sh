TOPIC_BASE_DIR=/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics
CORPUS=/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl

# Process all topic directories that contain llm_extracted_keywords.txt
for TOPIC_DIR in "$TOPIC_BASE_DIR"/*; do
    if [ -d "$TOPIC_DIR" ]; then
        TOPIC_NAME=$(basename "$TOPIC_DIR")
        LLM_EXTRACTED_KEYWORDS_PATH="$TOPIC_DIR/llm_extracted_keywords.txt"
        
        # Check if the llm_extracted_keywords.txt file exists
        if [ -f "$LLM_EXTRACTED_KEYWORDS_PATH" ]; then
            echo "Processing topic: $TOPIC_NAME"
            
            # Create gen directory if it doesn't exist
            mkdir -p "$TOPIC_DIR/gen"
            
            OUTPUT="$TOPIC_DIR/gen/text_add_llm_extracted_keywords.jsonl"
            
            python3 /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/llm_extracted_keywords_as_pq.py \
                --llm-extracted-keywords-path "$LLM_EXTRACTED_KEYWORDS_PATH" \
                --corpus "$CORPUS" \
                --output "$OUTPUT"
            
            echo "Completed processing for topic: $TOPIC_NAME"
        else
            echo "Skipping $TOPIC_NAME - llm_extracted_keywords.txt not found"
        fi
    fi
done

echo "All topics processed!"