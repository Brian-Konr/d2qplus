BASE_DIR=/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0606-biobert-mnli-reduce-outlier-for-scoring/gen
INPUT_FILES=(
    "$BASE_DIR/4shot_1perdoc_20beam_20total_5.jsonl"
    "$BASE_DIR/4shot_1perdoc_20beam_20total_6.jsonl"
    "$BASE_DIR/4shot_1perdoc_20beam_20total_7.jsonl"
)

KEYWORD_PATH=/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0606-biobert-mnli-reduce-outlier-for-scoring/keywords.jsonl

OUTPUT_FILE=/home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus/4shot_1perdoc_20beam_20total_pick50.jsonl

python3 /home/guest/r12922050/GitHub/d2qplus/src/utils/append_multiple_set_genq.py \
    --input_files "${INPUT_FILES[@]}" \
    --output_file "$OUTPUT_FILE" \
    --num_q_to_keep 50 \
    --keyword_path "$KEYWORD_PATH"