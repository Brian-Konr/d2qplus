# trainer.py
#!/usr/bin/env python3
import wandb
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from datasets import load_dataset
from trl.trainer import GRPOConfig, GRPOTrainer
from sentence_transformers import SentenceTransformer
from reward import CoverageRewardModel
from utils.constants import D2Q_SYS_PROMPT_NEW

def main():
    # start a new wandb run to track this training
    wandb.login()
    wandb.init(project="doc2query++", name="topic-coverage-only", notes="topic coverage only with PPO")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    output_dir = f"outputs/{model_name.split('/')[-1]}-GRPO"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        beta=0.04, # divergence coefficient – how much the policy is allowed to deviate from the reference model. higher value – more conservative updates. Default is 0.04
        per_device_train_batch_size=4,
        num_generations=4, # group size
        gradient_accumulation_steps=4,
        max_prompt_length=1024,
        max_completion_length=256,
        num_train_epochs=1,
        save_steps=100,
        use_vllm=True
    )

    # — Tokenizer & Models —
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    gen_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        # return_dict_in_generate=True,
    )
    model.generation_config = gen_config

    # - Reward Model -
    topic_vecs = torch.load("/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/vector-lookup/topic_vectors.pt")      # shape [K, d]
    tau = 0.35
    embed_model = SentenceTransformer("allenai/scibert_scivocab_uncased", device="cuda")
    reward_model = CoverageRewardModel(topic_vecs, embed_model, tau).cuda()

    # — Dataset (each example must have a “prompt” field) —
    DATA_PATH = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/integrated/data_with_prompt_1.jsonl"
    dataset = load_dataset("json", data_files=DATA_PATH)

    # make dataset smaller for testing
    dataset = dataset["train"].shuffle(seed=42).select(range(100))

    # — PPO Config with W&B logging —
    ppo_config = PPOConfig(
        exp_name="doc2query-rl"
    )

    # — Data collator to batch prompts —
    def collator(batch):
        messages = []
        topic_ids = []
        topic_weights = []
        for ex in batch:
            messages.append([
                {"role": "system", "content": D2Q_SYS_PROMPT_NEW},
                {"role": "user",   "content": ex["prompt"]},
            ])
            topic_ids.append([topic["topic_id"] for topic in ex["topics"]])
            topic_weights.append([topic["weight"] for topic in ex["topics"]])

        tok = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True, truncation=True)
        return {
            "input_ids": tok,
            "topic_ids": topic_ids,
            "topic_weights": topic_weights,
        }


    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        value_model=model,
        reward_model=reward_model,
        ref_model=ref_model,
        train_dataset=dataset,
        data_collator=collator,
    )
    ppo_trainer.train()

if __name__ == "__main__":
    main()
