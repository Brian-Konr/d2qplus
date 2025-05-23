# trainer.py
#!/usr/bin/env python3
import wandb
import torch
from tqdm import tqdm
from types import MethodType
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from sentence_transformers import SentenceTransformer
from reward import CoverageRewardModel
from utils.constants import D2Q_SYS_PROMPT_NEW

def tuple_to_dataclass(self, *args, **kwargs):
    kwargs["return_dict"] = True
    out = self.__old_fwd(*args, **kwargs)
    if isinstance(out, tuple):
        logits, *rest = out
        out = CausalLMOutputWithPast(logits=logits, past_key_values=None)
    return out

def main():
    # start a new wandb run to track this training
    wandb.login()
    wandb.init(project="doc2query++", name="topic-coverage-only", notes="topic coverage only with PPO")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # — Tokenizer & Models —
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model     = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).cuda()
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).cuda()

    for m in (model, ref_model):
        m.base_model_prefix = "pretrained_model"

    gen_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        # return_dict_in_generate=True,
    )
    model.generation_config = gen_config
    ref_model.generation_config = gen_config

    backbone = getattr(ref_model, ref_model.base_model_prefix)  # "pretrained_model"
    backbone.__old_fwd = backbone.forward
    backbone.forward   = MethodType(tuple_to_dataclass, backbone)

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
