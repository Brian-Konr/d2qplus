export CUDA_VISIBLE_DEVICES=0,1

TOPIC_DIR=/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0606-biobert-mnli
TOPIC_INFO_PKL=$TOPIC_DIR/topic_info_dataframe.pkl
CORPUS_TOPICS_PATH=$TOPIC_DIR/doc_topics.jsonl
CORPUS_PATH=/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl
KEYWORDS_PATH=/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/keywords/candidate_keywords_scibert.pkl
OUTPUT_PATH=$TOPIC_DIR/llm_extracted_keywords.jsonl

MODEL=meta-llama/Llama-3.1-8B-Instruct
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=3000
FINAL_EXTRACT_KEYWORDS=10
MAX_TOKENS=256
GPU_MEMORY_UTILIZATION=0.8

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