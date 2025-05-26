# trainer.py
#!/usr/bin/env python3

import wandb
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from trl.trainer import GRPOConfig, GRPOTrainer
from sentence_transformers import SentenceTransformer
from reward import CoverageRewardModel, topic_reward_func
from utils.constants import D2Q_SYS_PROMPT_NEW
from unsloth import FastModel

def main():
    # start a new wandb run to track this training
    wandb.login()
    wandb.init(project="doc2query++", name="topic-coverage-only", notes="topic coverage only with PPO")

    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    model, tokenizer = FastModel.from_pretrained(
        model_name,
        max_seq_length=1024,
        full_finetuning=True,
        token="" # need to load from .env
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!

        r = 8,           # Larger = higher accuracy, but might overfit
        lora_alpha = 8,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )
    output_dir = f"outputs/{model_name.split('/')[-1]}-GRPO"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        beta=0.04, # divergence coefficient – how much the policy is allowed to deviate from the reference model. higher value – more conservative updates. Default is 0.04
        per_device_train_batch_size=2,
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        num_generations=4, # group size
        logging_steps=1,
        bf16=True,
        gradient_accumulation_steps=2,
        max_prompt_length=1024,
        num_train_epochs=1,
        save_steps=100,
        use_vllm=True,
        temperature=0.7,
        max_completion_length=256, # maximum length of the generated completion
    )

    def preprocess_dataset(data_path, chunk_size=100) -> Dataset:
        # — Dataset (each example must have a “prompt” field) —
        dataset = load_dataset("json", data_files=data_path)
        dataset = dataset["train"].shuffle(seed=42).select(range(1000))
        total_samples = len(dataset)
        print(f"Loaded {total_samples} samples")

        def process_batch(batch):
            # batch is like {"prompt": [...], "topics":[...], ...}
            messages, topic_ids, topic_weights = [], [], []
            for raw_prompt, raw_topics in zip(batch["prompt"], batch["topics"]):
                # build the Chat‐style prompt
                messages.append([
                    {"role": "system", "content": D2Q_SYS_PROMPT_NEW},
                    {"role": "user",   "content": raw_prompt},
                ])
                # pull out topic_id and weight from each topic dict
                topic_ids.append([t["topic_id"]  for t in raw_topics])
                topic_weights.append([t["weight"]    for t in raw_topics])

            return {
                "prompt":        messages,
                "topic_ids":     topic_ids,
                "topic_weights": topic_weights,
            }

        return dataset.map(process_batch, batched=True, batch_size=chunk_size, remove_columns=["prompt", "topics"])
    
    dataset = preprocess_dataset("/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/integrated/data_with_prompt_1.jsonl")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[topic_reward_func]
    )

    trainer.train()
    


if __name__ == "__main__":
    main()
