#!/bin/bash
# filepath: /home/guest/r12922050/GitHub/d2qplus/scripts/d2q_gen_vllm.sh
export CUDA_VISIBLE_DEVICES=0,1
# Configuration variables
BASE_PATH="/home/guest/r13944029/IRLab/d2qplus"
DATASET="CFCube-1.1"
DATA_FILE="data_with_prompt_3.jsonl"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
TRAINED_MODEL="/home/guest/r12922050/GitHub/d2qplus/outputs/Llama-3.2-1B-Instruct-GRPO-separate-reward/checkpoint-1798"

TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=8192
TEMPERATURE=0.8
MAX_TOKENS=256
TEST_FLAG="--test"
TOPIC_KEYWORDS_FLAG="--with_topic_keywords"

# Derived paths
INPUT_PATH="${BASE_PATH}/augmented-data/${DATASET}/integrated/${DATA_FILE}"
OUTPUT_DIR="${BASE_PATH}/gen/${DATASET}"

OUTPUT_NAME="Llama-3.2-1B-Instruct-GRPO-separate-reward"


# python3 ${BASE_PATH}/src/generate.py \
#     ${TOPIC_KEYWORDS_FLAG} \
#     --integrated_data_with_prompt_path ${INPUT_PATH} \
#     --output_path "${OUTPUT_DIR}/${OUTPUT_NAME}.jsonl" \
#     --model ${TRAINED_MODEL} \
#     --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
#     --max_model_len ${MAX_MODEL_LEN} \
#     --temperature ${TEMPERATURE} \
#     --max_tokens ${MAX_TOKENS}

# echo "GRPO With Topic Keywords Generation Completed"


# # Job 1: base LLM With topic keywords
python3 ${BASE_PATH}/src/generate.py \
    ${TOPIC_KEYWORDS_FLAG} \
    --integrated_data_with_prompt_path ${INPUT_PATH} \
    --output_path "${OUTPUT_DIR}/with_topic_llama_1b.jsonl" \
    --model ${MODEL} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --max_model_len ${MAX_MODEL_LEN} \
    --temperature ${TEMPERATURE} \
    --max_tokens ${MAX_TOKENS}

echo "With Topic Keywords Generation Completed"

# # Job 2: base LLM Without topic keywords
python3 ${BASE_PATH}/src/generate.py \
    --integrated_data_with_prompt_path ${INPUT_PATH} \
    --output_path "${OUTPUT_DIR}/without_topic_llama_1b.jsonl" \
    --model ${MODEL} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --max_model_len ${MAX_MODEL_LEN} \
    --temperature ${TEMPERATURE} \
    --max_tokens ${MAX_TOKENS}

echo "Without Topic Keywords Generation Completed"