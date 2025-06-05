#!/bin/bash
# filepath: /home/guest/r12922050/GitHub/d2qplus/scripts/d2q_gen_vllm.sh
export CUDA_VISIBLE_DEVICES=2
# Configuration variables
BASE_PATH="/home/guest/r12922050/GitHub/d2qplus"
DATASET="nfcorpus"
DATA_FILE="data_with_prompt_3.jsonl"
MODEL="meta-llama/Llama-3.2-1B-Instruct"

TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=8192

PROMPT_TEMPLATE="promptagator"
FEW_SHOT_EXAMPLES_PATH="${BASE_PATH}/prompts/${PROMPT_TEMPLATE}/few_shot_examples.jsonl"
TEMPERATURE=0.7
MAX_TOKENS=64
NUM_RETURN_SEQUENCES=10

TEST_FLAG="--test"
TOPIC_KEYWORDS_FLAG="--with_topic_keywords"

# Derived paths
INPUT_PATH="${BASE_PATH}/augmented-data/${DATASET}/integrated/${DATA_FILE}"
OUTPUT_DIR="${BASE_PATH}/gen/${DATASET}"

OUTPUT_NAME="Llama-3.2-1B-Instruct-promptagator"


python3 ${BASE_PATH}/src/generate.py \
    --integrated_data_with_prompt_path ${INPUT_PATH} \
    --output_path "${OUTPUT_DIR}/${OUTPUT_NAME}.jsonl" \
    --model ${MODEL} \
    --prompt_template ${PROMPT_TEMPLATE} \
    --few_shot_examples_path ${FEW_SHOT_EXAMPLES_PATH} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --max_model_len ${MAX_MODEL_LEN} \
    --temperature ${TEMPERATURE} \
    --max_tokens ${MAX_TOKENS} \
    --return_sequence_num ${NUM_RETURN_SEQUENCES}

echo "Promptagator Generation Completed"