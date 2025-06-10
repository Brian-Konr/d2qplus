# max_input_length 0 means no limit, default to 512
export CUDA_VISIBLE_DEVICES=4

# Generate queries using DocTTTTTQuery

DATASET=CSFCube-1.1
NUM_GENERATED_QUERIES=20

python /home/guest/r12922050/GitHub/d2qplus/src/generate_t5.py \
    --model "macavaney/doc2query-t5-base-msmarco" \
    --num_examples "$NUM_GENERATED_QUERIES" \
    --max_input_length 512 \
    --max_output_length 64 \
    --top_k 10 \
    --batch_size 8 \
    --corpus_path "/home/guest/r12922050/GitHub/d2qplus/data/${DATASET}/corpus.jsonl" \
    --save_file "/home/guest/r12922050/GitHub/d2qplus/gen/${DATASET}/t5_${NUM_GENERATED_QUERIES}q.jsonl"
