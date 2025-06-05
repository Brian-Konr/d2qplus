# trainer.py
#!/usr/bin/env python3

import wandb
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from trl.trainer import GRPOConfig, GRPOTrainer
from sentence_transformers import SentenceTransformer
from reward_v2 import create_reward_functions
from utils.constants import D2Q_SYS_PROMPT_WITH_TOPIC
from rank_bm25 import BM25Okapi


def preprocess_dataset(data_path, chunk_size=1000) -> Dataset:
    # — Dataset (each example must have a “prompt” field) —
    dataset = load_dataset("json", data_files=data_path)
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    def process_batch(batch):
        # batch is like {"prompt": [...], "topics":[...], ...}
        messages, topic_ids, topic_weights, keywords = [], [], [], []
        for raw_prompt, raw_topics, raw_keywords in zip(batch["prompt"], batch["topics"], batch["keywords"]):
            # build the Chat‐style prompt
            messages.append([
                {"role": "system", "content": D2Q_SYS_PROMPT_WITH_TOPIC},
                {"role": "user",   "content": raw_prompt},
            ])
            # pull out topic_id and weight from each topic dict
            topic_ids.append([t["topic_id"]  for t in raw_topics])
            topic_weights.append([t["weight"]    for t in raw_topics])
            keywords.append([kw["key"] for kw in raw_keywords]) # keywords is a list of dicts with {"key": "keyword", "wegiht": 0.52..}

        return {
            "prompt":        messages,
            "topic_ids":     topic_ids,
            "topic_weights": topic_weights,
            "keywords_list": keywords
        }

    return dataset.map(process_batch, batched=True, batch_size=chunk_size, remove_columns=["prompt", "topics", "keywords"])

def main():
    # start a new wandb run to track this training
    wandb.login()
    wandb.init(project="doc2query++", name="topic-coverage-only-grpo-separate-reward", notes="topic coverage only with PPO")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()

    # - Dataset -
    data_path = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/integrated/data_with_prompt_1.jsonl"
    dataset = preprocess_dataset(data_path, chunk_size=1000)
    dataset = dataset["train"]
    print(f"Dataset processed with {len(dataset)} examples.")

    # - Topic coverage -
    embed_model = SentenceTransformer("allenai/scibert_scivocab_uncased", device="cuda")
    topic_vecs_path = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/vector-lookup/topic_vectors.pt"
    topic_similarity_threshold = 0.65

    # - Keyword Coverage -
    tokenized_texts = [doc.lower().split() for doc in dataset["text"]]
    bm25_corpus = BM25Okapi(tokenized_texts, k1=1.5, b=0.75)

    # - Relevance Reward -
    doc_vectors_path = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/corpus-lookup/document_vectors.pt" # Document ID: MED-10, Vector shape: torch.Size([768])
    
    weights = {
        'format': 0.2,
        'coverage': 0.2,
        'diversity': 0.2,
        'keyword': 0.2,
        'relevance': 0.2
    }

    reward_functions = create_reward_functions(
        embed_model=embed_model, 
        topic_vecs_path=topic_vecs_path, 
        doc_vecs_path=doc_vectors_path,
        bm25=bm25_corpus, 
        expected_n=10,
        tau=topic_similarity_threshold,
        weights=weights
    )

    output_dir = f"outputs/{model_name.split('/')[-1]}-GRPO-separate-reward"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
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
        save_steps=200,
        use_vllm=True,
        temperature=0.8,
        max_completion_length=256, # maximum length of the generated completion
    )

    # — Tokenizer & Models —
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_functions,  # Pass list of separate reward functions
    )

    trainer.train()
    


if __name__ == "__main__":
    main()
