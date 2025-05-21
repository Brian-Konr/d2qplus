# trainer.py
#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from reward import reward_fn

def main():
    model_name = "gpt2"

    # — Tokenizer & Models —
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model     = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).cuda()
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).cuda()

    # — Dataset (each example must have a “prompt” field) —
    dataset = load_dataset("json", data_files="data/prompts.jsonl", split="train")

    # — PPO Config with W&B logging —
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=16,
        forward_batch_size=4,
        log_with="wandb",
        wandb_project="doc2query_rl",
    )

    # — Data collator to batch prompts —
    def collator(batch):
        texts = [ex["prompt"] for ex in batch]
        tok   = tokenizer(texts, return_tensors="pt", padding=True)
        return {
            "input_ids":     tok.input_ids.cuda(),
            "attention_mask":tok.attention_mask.cuda()
        }

    # — PPO Trainer setup —
    trainer = PPOTrainer(
        config            = ppo_config,
        model             = model,
        ref_model         = ref_model,
        tokenizer         = tokenizer,
        train_dataset     = dataset,
        data_collator     = collator,
        reward_fn         = reward_fn,
        generation_kwargs = {
            "max_new_tokens": 120,
            "top_p":          0.9,
            "temperature":    0.8,
            "pad_token_id":   tokenizer.eos_token_id
        }
    )

    trainer.train()

if __name__ == "__main__":
    main()
