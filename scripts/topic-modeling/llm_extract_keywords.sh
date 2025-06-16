#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

DATASET=fiqa-5000

# Define list of topic names
# TOPIC_NAMES=(
#     "0612-all-mpnet-base-v2-topic-size-10sentence"
# )

TOPIC_NAMES=(
    "0612-all-mpnet-base-v2-topic-size-10sentence"
    "0612-all-mpnet-base-v2-topic-size-10sentence-reduce-outliers"
    "finbert-topic-size-10sentence"
    "finbert-topic-size-10sentence-reduce-outliers"
)

# Common paths and parameters
TOPIC_BASE_DIR=/home/guest/r12922050/GitHub/d2qplus/augmented-data/$DATASET/topics
CORPUS_PATH=/home/guest/r12922050/GitHub/d2qplus/data/$DATASET/corpus.jsonl
KEYWORDS_NAME=candidate_keywords_finbert
KEYWORDS_PATH=/home/guest/r12922050/GitHub/d2qplus/augmented-data/$DATASET/keywords/$KEYWORDS_NAME.pkl

# Model parameters
MODEL=meta-llama/Llama-3.1-8B-Instruct
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=3000
FINAL_EXTRACT_KEYWORDS=10
MAX_TOKENS=256
GPU_MEMORY_UTILIZATION=0.8

# Loop through each topic name
for TOPIC_NAME in "${TOPIC_NAMES[@]}"; do
    echo "Processing topic: $TOPIC_NAME"
    
    TOPIC_DIR=$TOPIC_BASE_DIR/$TOPIC_NAME
    TOPIC_INFO_PKL=$TOPIC_DIR/topic_info_dataframe.pkl
    CORPUS_TOPICS_PATH=$TOPIC_DIR/doc_topics.jsonl
    OUTPUT_PATH=$TOPIC_DIR/llm_extracted_keywords.jsonl
    
    python3 -m src.utils.llm_extract_keywords \
        --topic_info_pkl $TOPIC_INFO_PKL \
        --keywords_path $KEYWORDS_PATH \
        --corpus_topics_path $CORPUS_TOPICS_PATH \
        --corpus_path $CORPUS_PATH \
        --output_path $OUTPUT_PATH \
        --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
        --model $MODEL \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --max_model_len $MAX_MODEL_LEN \
        --max_tokens $MAX_TOKENS \
        --final_extract_keywords_num $FINAL_EXTRACT_KEYWORDS
    
    echo "Completed topic: $TOPIC_NAME"
    echo "---"
done

echo "All topics processed!"