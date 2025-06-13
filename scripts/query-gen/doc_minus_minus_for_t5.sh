
PERECENTAGE=50 # top k % queries to keep
DATASET=nfcorpus
INPUT_FILE="/home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus/scores-t5/t5_100q.jsonl" # 要放 scored t5 queries files (先跑 score_queries.sh)
OUTPUT_FILE="/home/guest/r12922050/GitHub/d2qplus/gen/$DATASET/top${PERECENTAGE}_queries_t5.jsonl"


python3 /home/guest/r12922050/GitHub/d2qplus/src/utils/doc_minus_minus_for_t5_genq.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --percentage "$PERECENTAGE"