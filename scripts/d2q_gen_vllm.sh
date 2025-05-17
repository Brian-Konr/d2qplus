# only use document text
python3 /home/guest/r12922050/GitHub/d2qplus/src/generate.py \
    --corpus_path /home/guest/r12922050/GitHub/d2qplus/data/scidocs/corpus.jsonl \
    --output_path "/home/guest/r12922050/GitHub/d2qplus/gen/scidocs_gen_10q_text_only.jsonl" \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --cuda_visible_devices 1,2 \
    --tensor_parallel_size 2 \
    --max_model_len 8192 \
    --temperature 0.7 \
    --max_tokens 512

# if has enhanced_rep (topic info)

# python3 /home/guest/r12922050/GitHub/d2qplus/src/generate.py \
#     --use_enhanced_rep \
#     --enhanced_rep_path /home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/topics/enhanced_rep.jsonl \
#     --corpus_path /home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl \
#     --output_path /home/guest/r12922050/GitHub/d2qplus/gen/nfcorpus_gen_10q_text_only.jsonl \
#     --model meta-llama/Llama-3.2-1B-Instruct \
#     --cuda_visible_devices 2 \
#     --tensor_parallel_size 1 \
#     --max_model_len 8192 \
#     --temperature 0.7 \
#     --max_tokens 512