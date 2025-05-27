export CUDA_VISIBLE_DEVICES=0

# python3 /home/guest/r12922050/GitHub/d2qplus/src/eval.py \
#     --corpus /home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus_dt5q_gen_10q.jsonl \
#     --queries /home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/queries.jsonl \
#     --qrels /home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/qrels/test.trec \
#     --index-dir /home/guest/r12922050/GitHub/d2qplus/built-index/nfcorpus \
#     --output /home/guest/r12922050/GitHub/d2qplus/eval/nfcorpus/llm_gen_10q_text_only.csv

python3 /home/guest/r12922050/GitHub/d2qplus/src/eval.py \
    --corpus /home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus/wit_topic_grpo_1b.jsonl \
    --queries /home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/queries.jsonl \
    --qrels /home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/qrels/test.trec \
    --index-dir /home/guest/r12922050/GitHub/d2qplus/built-index/nfcorpus-test \
    --output /home/guest/r12922050/GitHub/d2qplus/eval/nfcorpus/grpo_10q.csv