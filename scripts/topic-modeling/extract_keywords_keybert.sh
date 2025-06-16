
# Set default values
CORPUS_PATH=/home/guest/r12922050/GitHub/d2qplus/data/fiqa-5000/corpus.jsonl
OUTPUT_DIR=/home/guest/r12922050/GitHub/d2qplus/augmented-data/fiqa-5000/keywords
MODEL_NAME=ProsusAI/finbert
DEVICE="cuda:1"
TOP_N_CANDIDATES=10
NGRAM_MIN=1
NGRAM_MAX=2
DIVERSITY=0.6

OUTPUT_NAME="candidate_keywords_finbert.pkl"


python3 /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/extract_keywords_keybert.py \
    --corpus_path "$CORPUS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --device "$DEVICE" \
    --top_n_candidates "$TOP_N_CANDIDATES" \
    --ngram_min "$NGRAM_MIN" \
    --ngram_max "$NGRAM_MAX" \
    --diversity "$DIVERSITY" \
    --use_mmr \
    --output_name "$OUTPUT_NAME"