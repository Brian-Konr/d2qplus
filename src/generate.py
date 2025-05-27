import json
from typing import List
from utils.util import read_jsonl
from pydantic import BaseModel, Field
from utils.constants import D2Q_SYS_PROMPT_WITH_TOPIC, D2Q_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate queries using VLLM")
    parser.add_argument("--enhanced_rep_path", type=str, help="Path to the enhanced_rep.jsonl file")
    # parser.add_argument("--corpus_path", type=str, help="Path to the corpus.jsonl file")
    parser.add_argument("--output_path", type=str, help="Path to save the generated queries")
    parser.add_argument("--integrated_data_with_prompt_path", type=str, required=True, help="Path to the integrated data with prompt file")
    parser.add_argument("--with_topic_keywords", action="store_true", default=False, help="Use topic and keyword information in the prompts")
    # test
    parser.add_argument("--test", action="store_true", default=False, help="Run in test mode")
    parser.add_argument("--use_enhanced_rep", action="store_true", default=False, help="Use enhanced representation")

    # vllm config
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens for generation")
    return parser.parse_args()

def combine_topic_info(enhanced_rep_path: str, corpus_path: str) -> List[dict]:
    enhnaced_rep = read_jsonl(enhanced_rep_path)
    corpus = read_jsonl(corpus_path)
    corpus_dict = {doc["_id"]: doc["text"] for doc in corpus}
    for doc in enhnaced_rep:
        doc['text'] = corpus_dict[doc['doc_id']]
    return enhnaced_rep

def make_messages(data: List[dict], with_topic_keywords = False) -> List[dict]:
    """
    make conversation messages for LLM to generate queries based on the provided documents.
    
    `with_topic_keywords`: if True, the system prompt and user prompt will include topic, keyword information. But the data needs to have 'prompt' field.

    `data`: The data is expected to be a list of dictionaries, where each dictionary contains:

    - text: the document text
    - prompt (optional): the organized user prompt (can be obtained by running `prepare_training_data` in `utils/data.py`)
    """
    messages = []
    for doc in data:
        text = doc['text']
        sys_prompt = {"role": "system", "content": D2Q_SYS_PROMPT_WITH_TOPIC if with_topic_keywords else D2Q_SYSTEM_PROMPT}
        user_prompt = {"role": "user", "content": doc['prompt'] if with_topic_keywords else USER_PROMPT_TEMPLATE.replace("[DOCUMENT]", text)}
        messages.append([sys_prompt, user_prompt])
    return messages

def generate_queries_vllm(messages: List[dict], llm: LLM, sampling_params: SamplingParams) -> List[str]:
    gen_q = []
    outputs = llm.chat(messages, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        gen_q.append(generated_text)
    return gen_q


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output_path
    corpus = read_jsonl(args.integrated_data_with_prompt_path)
    
    # Create messages for vllm
    messages = make_messages(corpus, with_topic_keywords=args.with_topic_keywords)

    if args.test:
        # Test mode: only process the first 10 documents
        corpus = corpus[:10]
        messages = messages[:10]
        # save messages to jsonl to check
        with open("test_messages.jsonl", 'w') as f:
            for message in messages:
                f.write(json.dumps(message) + '\n')
        print("Test mode: only processing the first 10 documents.")
        print(f"Test messages saved to test_messages.jsonl for verification.")


    # Initialize vllm
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Generate queries
    generated_q = generate_queries_vllm(messages, llm, sampling_params)

    # Save the generated queries to jsonl (id, text, queries)
    output_data = []
    for doc, queries in zip(corpus, generated_q):
        output_data.append({
            "id": doc["doc_id"],
            "title": doc.get("title", ""),
            "text": doc["text"],
            "predicted_queries": queries
        })
    with open(output_path, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')
    print(f"Generated queries saved to {output_path}")
    