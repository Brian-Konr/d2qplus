TOPIC_BASE_DIR=/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics
TOPIC_NAME=0606-biobert-mnli-reduce-outlier

LLM_EXTRACTED_KEYWORDS_PATH="${TOPIC_BASE_DIR}/${TOPIC_NAME}/llm_extracted_keywords.txt"
CORPUS=/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl
OUTPUT="${TOPIC_BASE_DIR}/${TOPIC_NAME}/gen/text_add_llm_extracted_keywords.jsonl"

python3 /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/llm_extracted_keywords_as_pq.py \
    --llm-extracted-keywords-path "$LLM_EXTRACTED_KEYWORDS_PATH" \
    --corpus "$CORPUS" \
    --output "$OUTPUT"