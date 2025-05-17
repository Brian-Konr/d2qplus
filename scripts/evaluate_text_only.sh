export CUDA_VISIBLE_DEVICES=2
# python experiments/zero_shot.py \
#     --test \
#     --scored_file "/home/guest/r12922050/GitHub/d2qplus/scored/scifact_llm_N20_no_max_input.jsonl" \
#     --index_dir "test-index" \
#     --eval_dir "eval" \
#     --runs_dir "runs" \
#     --filter_type 'top' \
#     --percentages 30 50 \
#     --dataset 'irds:beir/trec-covid' \
#     --pt_name 'irds:beir/trec-covid' \
#     --query_column 'text' # in trec-covid, the query column is 'text', it also has other columns including query, narrative

DATASET="scidocs" 

python experiments/zero_shot.py \
    --test \
    --scored_file "/home/guest/r12922050/GitHub/d2qplus/scored/scifact_llm_N20_no_max_input.jsonl" \
    --index_dir "test-index" \
    --eval_dir "eval" \
    --runs_dir "runs" \
    --filter_type 'top' \
    --percentages 30 50 \
    --dataset "irds:beir/$DATASET" \
    --qrels "/home/guest/r12922050/GitHub/d2qplus/data/$DATASET/qrels/cut.trec" \
    --pt_name "$DATASET" \
    --corpus_jsonl_path "/home/guest/r12922050/GitHub/d2qplus/data/$DATASET/corpus_cut.jsonl" \
    --queries "/home/guest/r12922050/GitHub/d2qplus/data/$DATASET/queries_cut.jsonl"
