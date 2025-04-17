export CUDA_VISIBLE_DEVICES=2
export VLLM_USE_V1=0
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --enable-reasoning --reasoning-parser deepseek_r1