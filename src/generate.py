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
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus.jsonl file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated queries")

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

"""
Example of the enhanced_rep.jsonl file:

{"doc_id": "MED-10", "topics": [{"topic_id": 299, "words": ["statin", "statins", "users", "breast", "use", "diagnosis", "prescribing", "disaggregate", "researched", "situated"], "weight": 0.3333333333333333, "Enhanced_Topic": "Statin use and breast cancer survival"}, {"topic_id": 21, "words": ["breast", "invasive", "diagnosed", "cancer", "nh", "incidence", "lan", "mammography", "2003", "died"], "weight": 0.3333333333333333, "Enhanced_Topic": "Breast cancer incidence rates among non-Hispanic white women"}, {"topic_id": 1318, "words": ["statins", "ezetimibe", "postal", "idc", "829", "ilc", "151", "statin", "sharp", "monotherapy"], "weight": 0.3333333333333333, "Enhanced_Topic": "Statin and ezetimibe use and cancer risk"}]}


Example of the corpus.jsonl file:
{"_id": "MED-10", "title": "Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland", "text": "Recent studies have suggested that statins, an established drug group...", "metadata": {"url": "http://www.ncbi.nlm.nih.gov/pubmed/25329299"}}
"""

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
    enhanced_rep_path = args.enhanced_rep_path
    corpus_path = args.corpus_path
    output_path = args.output_path


    # Load the enhanced representation and corpus
    if args.use_enhanced_rep:
        corpus = combine_topic_info(enhanced_rep_path, corpus_path)
    else:
        corpus = read_jsonl(corpus_path)
    
    # Create messages for vllm
    messages = make_messages(corpus)

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
            "id": doc["_id"],
            "text": doc["text"],
            "predicted_queries": queries
        })
    with open(output_path, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')
    print(f"Generated queries saved to {output_path}")
    