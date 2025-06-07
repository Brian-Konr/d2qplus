export CUDA_VISIBLE_DEVICES=1

TOPIC_INFO_BASE_DIR="/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0606-pritamdeka-biobert-pos-keybert-mmr"

python3 /home/guest/r12922050/GitHub/d2qplus/src/utils/get_llm_representation.py \
    --few_shot_prompt_txt_path /home/guest/r12922050/GitHub/d2qplus/prompts/enhance_NL_topic.txt \
    --topic_base_dir "$TOPIC_INFO_BASE_DIR"