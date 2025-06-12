# max_input_length 0 means no limit, default to 512
export CUDA_VISIBLE_DEVICES=4

# Generate queries using DocTTTTTQuery

DATASET=fiqa
NUM_GENERATED_QUERIES=20

TOTAL_QUERIES=100
NUM_ITERATIONS=$((TOTAL_QUERIES / NUM_GENERATED_QUERIES))
NUM_ITERATIONS=5

for i in $(seq 1 $NUM_ITERATIONS); do
    python /home/guest/r12922050/GitHub/d2qplus/src/generate_t5.py \
        --model "macavaney/doc2query-t5-base-msmarco" \
        --num_examples "$NUM_GENERATED_QUERIES" \
        --max_input_length 512 \
        --max_output_length 64 \
        --top_k 10 \
        --batch_size 8 \
        --corpus_path "/home/guest/r12922050/GitHub/d2qplus/data/${DATASET}/corpus.jsonl" \
        --save_file "/home/guest/r12922050/GitHub/d2qplus/gen/${DATASET}/t5_${NUM_GENERATED_QUERIES}q_${i}.jsonl"
done

# Merge generated files
INPUT_FILES=""
for file in /home/guest/r12922050/GitHub/d2qplus/gen/${DATASET}/t5_${NUM_GENERATED_QUERIES}q_*.jsonl; do
    if [ -f "$file" ]; then
        INPUT_FILES="$INPUT_FILES $file"
    fi
done

python /home/guest/r12922050/GitHub/d2qplus/gen/append_multiple_t5_genq.py \
    --input_files $INPUT_FILES \
    --output_file "/home/guest/r12922050/GitHub/d2qplus/gen/${DATASET}/t5_${TOTAL_QUERIES}q.jsonl"