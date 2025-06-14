LLM_EXTRACTED_KEYWORDS_PATH=/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0606-biobert-mnli/llm_extracted_keywords.txt
CORPUS=/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl
OUTPUT=/home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus/text_add_llm_extracted_keywords.jsonl

python3 /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/llm_extracted_keywords_as_pq.py \
    --llm-extracted-keywords-path "$LLM_EXTRACTED_KEYWORDS_PATH" \
    --corpus "$CORPUS" \
    --output "$OUTPUT"