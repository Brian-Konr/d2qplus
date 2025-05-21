export CUDA_VISIBLE_DEVICES=2

# python3 /home/guest/r12922050/GitHub/d2qplus/src/eval.py \
#     --corpus /home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus_dt5q_gen_10q.jsonl \
#     --queries /home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/queries.jsonl \
#     --qrels /home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/qrels/test.trec \
#     --index-dir /home/guest/r12922050/GitHub/d2qplus/built-index/nfcorpus \
#     --output /home/guest/r12922050/GitHub/d2qplus/eval/nfcorpus/llm_gen_10q_text_only.csv

python3 /home/guest/r12922050/GitHub/d2qplus/src/eval.py \
    --corpus /home/guest/r12922050/GitHub/d2qplus/gen/scidocs_gen_10q_text_only_cut.jsonl \
    --queries /home/guest/r12922050/GitHub/d2qplus/data/scidocs/queries.jsonl \
    --qrels /home/guest/r12922050/GitHub/d2qplus/data/scidocs/qrels/cut.trec \
    --index-dir /home/guest/r12922050/GitHub/d2qplus/built-index/scidocs \
    --output /home/guest/r12922050/GitHub/d2qplus/eval/scidocs/llm_gen_10q_text_only.csv