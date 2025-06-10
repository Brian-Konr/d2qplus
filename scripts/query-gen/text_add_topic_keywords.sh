BASE_TOPIC_DIR="/home/guest/r12922050/GitHub/d2qplus/augmented-data/CSFCube-1.1/topics/0609-pritamdeka_scibert-biobert-pos-keybert-mmr"
BASE_DATASET_DIR="/home/guest/r12922050/GitHub/d2qplus/data/CSFCube-1.1"

for TOP_K in 1 3 5 10; do
    echo "Processing for TOP_K = ${TOP_K}"
    python3 /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/keywords_as_pq.py \
        --doc-topics "$BASE_TOPIC_DIR/doc_topics.jsonl" \
        --topic-info "$BASE_TOPIC_DIR/topic_info_dataframe.pkl" \
        --corpus "$BASE_DATASET_DIR/corpus.jsonl" \
        --output "/home/guest/r12922050/GitHub/d2qplus/gen/CSFCube-1.1/text_add_top_${TOP_K}_topic_keywords.jsonl" \
        --top-k "$TOP_K"
    echo "Generated text_add_top_${TOP_K}_topic_keywords.jsonl with top ${TOP_K} keywords per topic."
done