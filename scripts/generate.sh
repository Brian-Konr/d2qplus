# max_input_length 0 means no limit, default to 512
export CUDA_VISIBLE_DEVICES=2
python src/generate.py \
    --use_few_shot \
    --test \
    --engine "reasoning_llm" \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --num_examples 20 \
    --max_input_length 0 \
    --max_output_length 64 \
    --top_k 10 \
    --save_file "/home/guest/r12922050/GitHub/d2qplus/output/queries_llm_no_max_input.jsonl" \
    --dataset "irds:beir/scifact" \
    --few_shot_examples "/home/guest/r12922050/GitHub/d2qplus/examples/scifact_qg_examples.jsonl" \
    --base_url "http://localhost:8000/v1" \
    --max_workers 3 \
    --log_file "/home/guest/r12922050/GitHub/d2qplus/gen/reasoning_llm_N10.log"