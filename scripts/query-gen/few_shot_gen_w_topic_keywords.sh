#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Default paths and values
DATASET=fiqa-5000
CORPUS_PATH="/home/guest/r12922050/GitHub/d2qplus/data/$DATASET/corpus.jsonl"
CORE_PHRASE_PKL="/home/guest/r12922050/GitHub/d2qplus/augmented-data/$DATASET/keywords/candidate_keywords_finbert.pkl"
FEW_SHOT_PATH=/home/guest/r12922050/GitHub/d2qplus/prompts/promptagator/fiqa_few_shot_examples.jsonl
FEW_SHOT_NUM=4
QUERY_PER_DOC=3
NUM_RETURN_SEQUENCES=10

# Model parameters
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=4000
GPU_MEMORY_UTILIZATION=0.8
TEMPERATURE=0.8
MAX_TOKENS=256

# Required parameter (no default)
TOPIC_BASE_DIR=/home/guest/r12922050/GitHub/d2qplus/augmented-data/$DATASET/topics
TOPIC_NAME=0612-all-mpnet-base-v2-topic-size-10sentence
TOPIC_DIR=$TOPIC_BASE_DIR/$TOPIC_NAME

LLM_KEYWORDS_PATH=$TOPIC_DIR/llm_extracted_keywords.jsonl

TOPIC_NUM=2 # fiqa has less topics, so we can set it to 2 (nfcorpus uses 3)

# Call your Python script with all parameters
# Add --no-topic-keywords flag here if you want to disable topic and keywords
python /home/guest/r12922050/GitHub/d2qplus/src/few_shot_generate_w_topic_keywords.py \
    --corpus_path "$CORPUS_PATH" \
    --core_phrase_pkl "$CORE_PHRASE_PKL" \
    --topic_dir "$TOPIC_DIR" \
    --llm_keywords_path $LLM_KEYWORDS_PATH \
    --few_shot_path "$FEW_SHOT_PATH" \
    --few_shot_num "$FEW_SHOT_NUM" \
    --query_per_doc "$QUERY_PER_DOC" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --model "$MODEL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --topic_num "$TOPIC_NUM" \
    --test