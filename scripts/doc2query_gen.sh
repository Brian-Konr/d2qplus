# max_input_length 0 means no limit, default to 512
export CUDA_VISIBLE_DEVICES=2

# Generate queries using DocTTTTTQuery

python /home/guest/r12922050/GitHub/d2qplus/src/generate_arxiv.py \
    --engine "doc2query" \
    --model "macavaney/doc2query-t5-base-msmarco" \
    --num_examples 10 \
    --max_input_length 512 \
    --max_output_length 64 \
    --top_k 10 \
    --save_file "/home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus_dt5q_gen_10q.jsonl" \
    --corpus_path "/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl" \
    --log_file "/home/guest/r12922050/GitHub/d2qplus/gen/dt5q.log"

