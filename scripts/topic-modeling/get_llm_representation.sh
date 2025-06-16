export CUDA_VISIBLE_DEVICES=5

DATASET=fiqa-5000

TOPIC_NAMES=(
    "0612-all-mpnet-base-v2-topic-size-10sentence-reduce-outliers"
    "finbert-topic-size-10sentence"
    "finbert-topic-size-10sentence-reduce-outliers"
)

for TOPIC_NAME in "${TOPIC_NAMES[@]}"; do
    TOPIC_INFO_BASE_DIR="/home/guest/r12922050/GitHub/d2qplus/augmented-data/${DATASET}/topics/${TOPIC_NAME}"
    python3 /home/guest/r12922050/GitHub/d2qplus/src/utils/get_llm_representation.py \
        --few_shot_prompt_txt_path "/home/guest/r12922050/GitHub/d2qplus/prompts/topic-modeling/enhance_NL_topic_${DATASET}.txt" \
        --topic_base_dir "$TOPIC_INFO_BASE_DIR"
    
    echo "Completed topic: $TOPIC_NAME"
    echo "---"
done

echo "All topics processed!"