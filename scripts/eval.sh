export CUDA_VISIBLE_DEVICES=2

python3 /home/guest/r12922050/GitHub/d2qplus/src/eval.py \
    --corpus /home/guest/r12922050/GitHub/d2qplus/gen/scidocs_gen_10q_text_only.jsonl \
    --queries /home/guest/r12922050/GitHub/d2qplus/data/scidocs/queries.jsonl \
    --qrels /home/guest/r12922050/GitHub/d2qplus/data/scidocs/qrels/test.trec \
    --index-dir /home/guest/r12922050/GitHub/d2qplus/built-index/scidocs \
    --output /home/guest/r12922050/GitHub/d2qplus/eval/scidocs/llm_gen_10q_text_only.csv