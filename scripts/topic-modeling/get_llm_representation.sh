export CUDA_VISIBLE_DEVICES=1

DATASET="CSFCube-1.1"
TOPIC_MODELING_NAME="0609-pritamdeka_scibert-biobert-pos-keybert-mmr"
TOPIC_INFO_BASE_DIR="/home/guest/r12922050/GitHub/d2qplus/augmented-data/${DATASET}/topics/${TOPIC_MODELING_NAME}"

python3 /home/guest/r12922050/GitHub/d2qplus/src/utils/get_llm_representation.py \
    --few_shot_prompt_txt_path "/home/guest/r12922050/GitHub/d2qplus/prompts/topic-modeling/enhance_NL_topic_${DATASET}.txt" \
    --topic_base_dir "$TOPIC_INFO_BASE_DIR"