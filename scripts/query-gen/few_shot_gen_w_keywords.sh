export CUDA_VISIBLE_DEVICES=6,7

CORPUS_PATH="/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl"
TOPIC_BASE_DIR="/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics"
TOPIC_NAME="0606-biobert-mnli-reduce-outlier"
TOPIC_DIR="${TOPIC_BASE_DIR}/${TOPIC_NAME}"

# FEW_SHOT_PATH=/home/guest/r12922050/GitHub/d2qplus/prompts/promptagator/few_shot_examples.jsonl
FEW_SHOT_PATH=/home/guest/r12922050/GitHub/d2qplus/prompts/few_shot_query_set_nfcorpus.jsonl
FEW_SHOT_NUM=2

QUERY_PER_DOC=5
NUM_RETURN_SEQUENCES=4

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.9
TEMPERATURE=0.8
MAX_TOKENS=256

python3 /home/guest/r12922050/GitHub/d2qplus/src/few_shot_generate_w_keywords.py \
    --corpus_path "$CORPUS_PATH" \
    --topic_dir "$TOPIC_DIR" \
    --few_shot_path "$FEW_SHOT_PATH" \
    --few_shot_num "$FEW_SHOT_NUM" \
    --query_per_doc "$QUERY_PER_DOC" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --model "$MODEL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS"