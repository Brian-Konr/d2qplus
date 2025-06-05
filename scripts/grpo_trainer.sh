export CUDA_VISIBLE_DEVICES=5,6
# export NCCL_P2P_DISABLE=1 
# export NCCL_IB_DISABLE=1
accelerate launch --num_processes 2 /home/guest/r12922050/GitHub/d2qplus/src/grpo_trainer.py
