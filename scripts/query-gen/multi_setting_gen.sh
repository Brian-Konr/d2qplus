#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

CORPUS_PATH="/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl"
TOPIC_BASE_DIR="/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics"
TOPIC_NAME="0606-biobert-mnli-reduce-outlier-for-scoring"
TOPIC_DIR="${TOPIC_BASE_DIR}/${TOPIC_NAME}"

FEW_SHOT_PATH="/home/guest/r12922050/GitHub/d2qplus/prompts/promptagator/few_shot_examples.jsonl"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=4000
GPU_MEMORY_UTILIZATION=0.8
TEMPERATURE=0.8
MAX_TOKENS=256

# Setting 1: query_per_doc=5, num_return=10, few_shot=4
echo "===== Setting 1: query_per_doc=5, num_return=10, few_shot=4 ====="
python3 /home/guest/r12922050/GitHub/d2qplus/src/few_shot_generate_w_keywords.py \
    --corpus_path "$CORPUS_PATH" \
    --topic_dir "$TOPIC_DIR" \
    --few_shot_path "$FEW_SHOT_PATH" \
    --few_shot_num 4 \
    --query_per_doc 5 \
    --num_return_sequences 10 \
    --model "$MODEL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --run 1

# Setting 2: query_per_doc=5, num_return=10, few_shot=6
echo "===== Setting 2: query_per_doc=5, num_return=10, few_shot=6 ====="
python3 /home/guest/r12922050/GitHub/d2qplus/src/few_shot_generate_w_keywords.py \
    --corpus_path "$CORPUS_PATH" \
    --topic_dir "$TOPIC_DIR" \
    --few_shot_path "$FEW_SHOT_PATH" \
    --few_shot_num 6 \
    --query_per_doc 5 \
    --num_return_sequences 10 \
    --model "$MODEL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --run 2

# Setting 3: query_per_doc=3, num_return=10, few_shot=4 (run 2 times)
for i in {1..2}; do
    echo "===== Setting 3: query_per_doc=3, num_return=10, few_shot=4 (run $i/2) ====="
    python3 /home/guest/r12922050/GitHub/d2qplus/src/few_shot_generate_w_keywords.py \
        --corpus_path "$CORPUS_PATH" \
        --topic_dir "$TOPIC_DIR" \
        --few_shot_path "$FEW_SHOT_PATH" \
        --few_shot_num 4 \
        --query_per_doc 3 \
        --num_return_sequences 10 \
        --model "$MODEL" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --temperature "$TEMPERATURE" \
        --max_tokens "$MAX_TOKENS" \
        --run $((2 + i))
done

# Setting 4: query_per_doc=1, num_return=20, few_shot=4 (run 3 times)
for i in {1..3}; do
    echo "===== Setting 4: query_per_doc=1, num_return=20, few_shot=4 (run $i/3) ====="
    python3 /home/guest/r12922050/GitHub/d2qplus/src/few_shot_generate_w_keywords.py \
        --corpus_path "$CORPUS_PATH" \
        --topic_dir "$TOPIC_DIR" \
        --few_shot_path "$FEW_SHOT_PATH" \
        --few_shot_num 4 \
        --query_per_doc 1 \
        --num_return_sequences 20 \
        --model "$MODEL" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --temperature "$TEMPERATURE" \
        --max_tokens "$MAX_TOKENS" \
        --run $((4 + i))
done

echo "🎉 All settings completed!"
