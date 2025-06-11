#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7


# Job Control - set which jobs to run (space-separated list)
# Options: "base_with_topic", "base_without_topic", "trained_with_topic", plan-then-write-given-topics-plan, plan-then-write-identify-then-plan
JOBS_TO_RUN="promptagator"

# Configuration variables
BASE_PATH="/home/guest/r12922050/GitHub/d2qplus"
DATASET="nfcorpus"
TOPIC_DIR="0606-pritamdeka-biobert-pos-keybert-mmr"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME_FOR_SAVE=${MODEL##*/}  # Extract everything after the last '/'

TRAINED_MODEL="/home/guest/r12922050/GitHub/d2qplus/outputs/Llama-3.2-1B-Instruct-GRPO-separate-reward/checkpoint-1798"

# VLLM Configuration
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4000

# Sampling Parameters
TEMPERATURE=0.8
MAX_TOKENS=64 # need to look at constants.py FIXED_NUMBER_OF_QUERIES to see how many queries model generates at a time
RETURN_SEQUENCE_NUM=20 # actual query generated per document will be FIXED_NUMBER_OF_QUERIES (constants.py) * RETURN_SEQUENCE_NUM
NUM_OF_QUERIES_PER_DOC=5
# TODOs: need to remove FIXED_NUMBER_OF_QUERIES in constants.py and use it here for better control

TOTAL_TARGET_QUERIES=100

# Prompt Parameters
PROMPT_TEMPLATE="d2q"
MAX_KEYWORDS=15
MAX_TOPICS=5



# topic modeling related paths
ENHANCED_TOPIC_INFO_PKL="${BASE_PATH}/augmented-data/${DATASET}/topics/${TOPIC_DIR}/topic_info_dataframe_enhanced.pkl"
CORPUS_TOPICS_PATH="${BASE_PATH}/augmented-data/${DATASET}/topics/${TOPIC_DIR}/doc_topics.jsonl"

CORPUS_PATH="${BASE_PATH}/data/${DATASET}/corpus.jsonl"
OUTPUT_DIR="${BASE_PATH}/gen/${DATASET}"
# only use this for the trained model (because there might be different trained model checkpoints)
OUTPUT_NAME="Llama-3.2-1B-Instruct-GRPO-separate-reward"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Job 1: Trained LLM with topic keywords
if [[ " $JOBS_TO_RUN " =~ " trained_with_topic " ]]; then
    echo "Starting: Trained LLM with topic keywords generation..."
    python3 ${BASE_PATH}/src/generate.py \
        --enhanced_topic_info_pkl ${ENHANCED_TOPIC_INFO_PKL} \
        --corpus_path ${CORPUS_PATH} \
        --corpus_topics_path ${CORPUS_TOPICS_PATH} \
        --output_path "${OUTPUT_DIR}/${OUTPUT_NAME}.jsonl" \
        --model ${TRAINED_MODEL} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --max_model_len ${MAX_MODEL_LEN} \
        --temperature ${TEMPERATURE} \
        --max_tokens ${MAX_TOKENS} \
        --return_sequence_num ${RETURN_SEQUENCE_NUM} \
        --with_topic_keywords \
        --with_topic_weights \
        --prompt_template ${PROMPT_TEMPLATE} \
        --max_keywords ${MAX_KEYWORDS} \
        --max_topics ${MAX_TOPICS}
    echo "Trained LLM with topic keywords generation completed"
    echo "----------------------------------------"
fi

# Job 2: Base LLM with topic keywords
if [[ " $JOBS_TO_RUN " =~ " base_with_topic " ]]; then
    echo "Starting: ${MODEL} with topic keywords generation..."
    python3 ${BASE_PATH}/src/generate.py \
        --enhanced_topic_info_pkl ${ENHANCED_TOPIC_INFO_PKL} \
        --corpus_path ${CORPUS_PATH} \
        --corpus_topics_path ${CORPUS_TOPICS_PATH} \
        --output_path "${OUTPUT_DIR}/with_topic_${MODEL_NAME_FOR_SAVE}.jsonl" \
        --model ${MODEL} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --max_model_len ${MAX_MODEL_LEN} \
        --temperature ${TEMPERATURE} \
        --max_tokens ${MAX_TOKENS} \
        --return_sequence_num ${RETURN_SEQUENCE_NUM} \
        --with_topic_keywords \
        --with_topic_weights \
        --prompt_template ${PROMPT_TEMPLATE} \
        --max_keywords ${MAX_KEYWORDS} \
        --max_topics ${MAX_TOPICS}
    echo "Base LLM with topic keywords generation completed"
    echo "----------------------------------------"
fi

# Job 3: Base LLM without topic keywords
if [[ " $JOBS_TO_RUN " =~ " base_without_topic " ]]; then
    echo "Starting: ${MODEL} without topic keywords generation..."
    python3 ${BASE_PATH}/src/generate.py \
        --enhanced_topic_info_pkl ${ENHANCED_TOPIC_INFO_PKL} \
        --corpus_path ${CORPUS_PATH} \
        --corpus_topics_path ${CORPUS_TOPICS_PATH} \
        --output_path "${OUTPUT_DIR}/without_topic_${MODEL_NAME_FOR_SAVE}.jsonl" \
        --model ${MODEL} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --max_model_len ${MAX_MODEL_LEN} \
        --temperature ${TEMPERATURE} \
        --max_tokens ${MAX_TOKENS} \
        --return_sequence_num ${RETURN_SEQUENCE_NUM} \
        --prompt_template ${PROMPT_TEMPLATE} \
        --max_keywords ${MAX_KEYWORDS} \
        --max_topics ${MAX_TOPICS}
    echo "Base LLM without topic keywords generation completed"
    echo "----------------------------------------"
fi

# Job 3: Base LLM without topic keywords
if [[ " $JOBS_TO_RUN " =~ " plan-then-write-given-topics-plan " ]]; then
    PROMPT_TEMPLATE="plan-then-write-identify-then-plan"
    echo "Starting: ${MODEL} ${PROMPT_TEMPLATE} generation..."
    python3 ${BASE_PATH}/src/generate.py \
        --test \
        --enhanced_topic_info_pkl ${ENHANCED_TOPIC_INFO_PKL} \
        --corpus_path ${CORPUS_PATH} \
        --corpus_topics_path ${CORPUS_TOPICS_PATH} \
        --output_path "${OUTPUT_DIR}/${PROMPT_TEMPLATE}_${MODEL_NAME_FOR_SAVE}.jsonl" \
        --model ${MODEL} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --max_model_len ${MAX_MODEL_LEN} \
        --temperature ${TEMPERATURE} \
        --max_tokens ${MAX_TOKENS} \
        --num_of_queries ${NUM_OF_QUERIES_PER_DOC} \
        --return_sequence_num ${RETURN_SEQUENCE_NUM} \
        --prompt_template ${PROMPT_TEMPLATE} \
        --max_keywords ${MAX_KEYWORDS} \
        --max_topics ${MAX_TOPICS}
    echo "Base LLM ${PROMPT_TEMPLATE} completed"
    echo "----------------------------------------"
fi

if [[ " $JOBS_TO_RUN " =~ " plan-then-write-identify-then-plan " ]]; then
    PROMPT_TEMPLATE="plan-then-write-identify-then-plan"
    echo "Starting: ${MODEL} ${PROMPT_TEMPLATE} generation..."
    python3 ${BASE_PATH}/src/generate.py \
        --test \
        --enhanced_topic_info_pkl ${ENHANCED_TOPIC_INFO_PKL} \
        --corpus_path ${CORPUS_PATH} \
        --corpus_topics_path ${CORPUS_TOPICS_PATH} \
        --output_path "${OUTPUT_DIR}/${PROMPT_TEMPLATE}_${MODEL_NAME_FOR_SAVE}.jsonl" \
        --model ${MODEL} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --max_model_len ${MAX_MODEL_LEN} \
        --temperature ${TEMPERATURE} \
        --max_tokens ${MAX_TOKENS} \
        --num_of_queries ${NUM_OF_QUERIES_PER_DOC} \
        --return_sequence_num ${RETURN_SEQUENCE_NUM} \
        --prompt_template ${PROMPT_TEMPLATE} \
        --max_keywords ${MAX_KEYWORDS} \
        --max_topics ${MAX_TOPICS}
    echo "Base LLM ${PROMPT_TEMPLATE} completed"
    echo "----------------------------------------"
fi

if [[ " $JOBS_TO_RUN " =~ " promptagator " ]]; then
    PROMPT_TEMPLATE="promptagator"
    echo "Starting: ${MODEL} ${PROMPT_TEMPLATE} generation..."
    python3 ${BASE_PATH}/src/generate.py \
        --few_shot_examples_path /home/guest/r12922050/GitHub/d2qplus/prompts/promptagator/few_shot_examples.jsonl \
        --enhanced_topic_info_pkl ${ENHANCED_TOPIC_INFO_PKL} \
        --corpus_path ${CORPUS_PATH} \
        --corpus_topics_path ${CORPUS_TOPICS_PATH} \
        --output_path "${OUTPUT_DIR}/${PROMPT_TEMPLATE}_${MODEL_NAME_FOR_SAVE}.jsonl" \
        --model ${MODEL} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --max_model_len ${MAX_MODEL_LEN} \
        --temperature ${TEMPERATURE} \
        --max_tokens ${MAX_TOKENS} \
        --num_of_queries ${NUM_OF_QUERIES_PER_DOC} \
        --total_target_queries ${TOTAL_TARGET_QUERIES} \
        --return_sequence_num ${RETURN_SEQUENCE_NUM} \
        --prompt_template ${PROMPT_TEMPLATE} \
        --max_keywords ${MAX_KEYWORDS} \
        --max_topics ${MAX_TOPICS}
    echo "Base LLM ${PROMPT_TEMPLATE} completed"
    echo "----------------------------------------"
fi

echo "All selected jobs completed!"