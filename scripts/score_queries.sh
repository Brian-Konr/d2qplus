# please run this script in the root directory of the project
export CUDA_VISIBLE_DEVICES=2
python experiments/score_generator.py \
    --input /home/guest/r12922050/GitHub/d2qplus/output/queries_llm_no_max_input.jsonl \
    --output /home/guest/r12922050/GitHub/d2qplus/scored/scifact_llm_N20_no_max_input.jsonl \
    --log /home/guest/r12922050/GitHub/d2qplus/scored/scifact_llm_N20_no_max_input.log