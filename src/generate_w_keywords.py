import re, os
import argparse
import sys
from typing import Dict, List

# Third-party imports
import pandas as pd

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel

from utils.util import read_jsonl
from utils.data import combine_topic_info

from transformers import AutoTokenizer
from utils.constants import PROMPTAGATOR_SET_GEN_SYS_PROMPT, PROMPTAGATOR_SET_GEN_USER_PROMPT

import json

def truncate_document(text: str, tokenizer, max_tokens: int = 512) -> str:
    """
    Truncate document text to fit within specified token limit.
    
    Args:
        text: Document text to truncate
        tokenizer: VLLM tokenizer
        max_tokens: Maximum number of tokens allowed
    
    Returns:
        Truncated document text
    """
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)
    
    return truncated_text

def construct_prompt_messages(corpus, tokenizer, docid2keywords, few_shots: List, query_per_doc: int = 5) -> List[List]:
    # construct messages list
    
    messages = []
    for doc in corpus:
        document = truncate_document(doc['text'], tokenizer)
        keywords = docid2keywords[doc['doc_id']]
        sys_prompt = PROMPTAGATOR_SET_GEN_SYS_PROMPT.replace("<num_of_queries>", str(query_per_doc))
        for example in few_shots:
            sys_prompt += f"Article:\n{example['doc_text']}\n\n"
            sys_prompt += f"Query:\n{example['query_text']}\n\n"

        user_template = PROMPTAGATOR_SET_GEN_USER_PROMPT

        user_content = user_template.replace("<document>", document).replace("<keywords>", keywords).replace("<num_of_queries>", str(query_per_doc))
        messages.append([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}])
    return messages
def genereate_queries_using_llm(
        messages: List[List],
        llm: LLM,
        sampling_params: SamplingParams
    ) -> List[str]:
    all_gen_q = []

    keywords = []
    outputs = llm.chat(messages, sampling_params)
    for i, output in enumerate(outputs):
        all_gen_q.append([])  # Initialize list for each document
        for seq in output.outputs:
            queries = seq.text.strip().split('\n')
            # Filter out empty strings and strip whitespace
            queries = [q.strip() for q in queries if q.strip()]
            all_gen_q[i].append(queries)
        
    return all_gen_q



def main():
    parser = argparse.ArgumentParser(description="Extract keywords using VLLM")
    
    parser.add_argument("--corpus_path", type=str,
                        default="/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl",
                        help="Path to the corpus JSONL file")
    parser.add_argument("--topic_dir", type=str, required=True,
                        help="Path to the directory containing topic information files that guide the query generation")

    # few shot
    parser.add_argument("--few_shot_path", type=str, default="/home/guest/r12922050/GitHub/d2qplus/prompts/few_shot_query_set_nfcorpus.jsonl")
    parser.add_argument("--few_shot_num", type=int, default=2)

    parser.add_argument("--query_per_doc", type=int, default=5,
                        help="Number of queries to generate per document")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of sequences to return per message (that is, number of query set per document)")
        
    # Model parameters
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model name for VLLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=4096,
                       help="Maximum model length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=256,
                       help="Maximum tokens to generate")
    
    parser.add_argument("--test", action="store_true", default=False)
    
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # construct docid2keywords dict (docid -> keywords str)
    docid2keywords = {}
    llm_keywords_path = args.topic_dir + "/llm_extracted_keywords.txt"
    with open(llm_keywords_path, "r") as f:
        llm_keywords = [line.strip() for line in f.readlines()]
    for line in llm_keywords:
        doc_id, keywords = line.split(":", 1)  # only first colon is used to split
        docid2keywords[doc_id.strip()] = keywords
    # elif 自己加整入不同 keywords 的方式

    import random
    few_shots = []
    with open(args.few_shot_path, "r") as f:
        few_shot_data = [json.loads(line) for line in f.readlines()]
        # Randomly sample few_shot_num examples
        few_shots = random.sample(few_shot_data, args.few_shot_num)

    corpus = combine_topic_info(
        enhanced_topic_info_pkl=f"{args.topic_dir}/topic_info_dataframe.pkl",
        corpus_topics_path=f"{args.topic_dir}/doc_topics.jsonl",
        corpus_path=args.corpus_path
    )

    if args.test:
        corpus = corpus[:10]

    messages = construct_prompt_messages(
        corpus=corpus, 
        tokenizer=tokenizer,
        docid2keywords=docid2keywords,
        few_shots=few_shots,
        query_per_doc=args.query_per_doc
    )

    # output top 3 messages to check
    for i, message in enumerate(messages[:3]):
        print(f"Message {i+1}:")
        for part in message:
            print(f"{part['role']}: {part['content']}")
        print("\n")    
    # vllm part
    llm = LLM(model=args.model, 
              tensor_parallel_size=args.tensor_parallel_size, 
              max_model_len=args.max_model_len, 
              gpu_memory_utilization=args.gpu_memory_utilization
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_return_sequences,
    )

    generated_q = genereate_queries_using_llm(messages=messages, llm=llm, sampling_params=sampling_params)
    # if output directory doesn't exist, create it
    # output dir is topic_dir + /gen
    total_query_per_doc = args.query_per_doc * args.num_return_sequences
    output_path = os.path.join(args.topic_dir, "gen", f"few_shot_query_set_gen_{total_query_per_doc}q.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)    

    output_data = []
    for doc, queries in zip(corpus, generated_q):
        output_data.append({
            "id": doc["doc_id"],
            "title": doc.get("title", ""),
            "text": doc["text"],
            "predicted_queries": queries
        })

    with open(output_path, "w") as f:
        for doc in output_data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Generated queries saved to {output_path}")

    
if __name__ == "__main__":
    main()
