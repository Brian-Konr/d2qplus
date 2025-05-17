python experiments/zero_shot.py \
    --scored_file "/home/guest/r12922050/GitHub/d2qplus/scored/scifact_llm_N20_no_max_input.jsonl" \
    --index_dir "/home/guest/r12922050/GitHub/d2qplus/ind" \
    --eval_dir "eval" \
    --runs_dir "runs" \
    --filter_type 'top' \
    --percentages 30 50 \
    --dataset 'irds:beir/scifact' \
    --qrels "/home/guest/r12922050/GitHub/d2qplus/data/scifact-qrels/test.trec" \
    --pt_name 'irds:beir/scifact' \