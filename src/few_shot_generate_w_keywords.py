import re, os
import argparse
import sys
from typing import List

# Third-party imports
import pandas as pd

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel

from .utils.util import read_jsonl
from .utils.data import combine_topic_info

from transformers import AutoTokenizer
from utils.constants import PROMPTAGATOR_SET_GEN_SYS_PROMPT, PROMPTAGATOR_SET_GEN_USER_PROMPT

class KeywordSet(BaseModel):
    keywords: list[str]

json_schema = KeywordSet.model_json_schema()


def truncate_document(text: str, tokenizer, max_tokens: int = 1024) -> str:
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

def construct_prompt_messages(enhanced_corpus, tokenizer, keywords: str, few_shots: List, query_per_doc: int = 5) -> List[List]:
    # construct messages list
    messages = []
    for doc in enhanced_corpus:
        document = truncate_document(doc['text'], tokenizer, max_tokens=1024)
        prompt = PROMPTAGATOR_SET_GEN_SYS_PROMPT.replace("<num_of_queries>", str(query_per_doc))
        for example in few_shots:
            prompt += f"Article: {example['doc_text']}\n"
            prompt += f"Query: {example['query_text']}\n\n"
        user_template = PROMPTAGATOR_SET_GEN_USER_PROMPT

        user_content = user_template.replace("<document>", document).replace("<keywords>", ", ".join(keywords)).replace("<num_of_queries>", str(query_per_doc))
        messages.append(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that selects the most relevant keywords for a given document. Your task is to choose keywords that best represent the core theme of the document."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]

        )
    return messages
    """Extract keywords from a JSON string."""
    try:
        data = KeywordSet.model_validate_json(json_str)
        return data.keywords
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []
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
    
    parser.add_argument("--topic_info_pkl", type=str,
                        help="Path to the topic information pickle file")
    parser.add_argument("--llm_keywords_path", type=str, default=None)
    parser.add_argument("--keywords_path", type=str,default=None)
    parser.add_argument("--corpus_topics_path", type=str,
                        default="/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus_topics.jsonl",
                        help="Path to the corpus topics JSONL file")
    parser.add_argument("--corpus_path", type=str,
                        default="/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl",
                        help="Path to the corpus JSONL file")
    parser.add_argument("--output_path", type=str,
                        default="/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/keywords/extracted_keywords.txt",
                        help="Path to save the extracted keywords")

    # few shot
    parser.add_argument("--few_shot_path", type=str, default="/home/guest/r12922050/GitHub/d2qplus/prompts/few_shot_query_set_nfcorpus.jsonl")
    parser.add_argument("--few_shot_num", type=int, default=2)


    
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
    parser.add_argument("--max_tokens", type=int, default=128,
                       help="Maximum tokens to generate")
    
    parser.add_argument("--final_extract_keywords_num", type=int, default=10,
                        help="Number of keywords to extract for each document")
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    corpus = combine_topic_info(
        enhanced_topic_info_pkl=args.topic_info_pkl,
        corpus_topics_path=args.corpus_topics_path,
        corpus_path=args.corpus_path
    )

    llm_keywords_dict = {}
    if args.llm_keywords_path:
        with open(args.llm_keywords_path, "r") as f:
            llm_keywords = [line.strip() for line in f.readlines()]
        for line in llm_keywords:
            doc_id, keywords = line.split(":", 1)  # only first colon is used to split
            llm_keywords_dict[doc_id.strip()] = keywords
    
    messages = construct_prompt_messages(
        corpus, 
        keybert_extracted_path=args.keywords_path, 
        final_extract_keyword_num=args.final_extract_keywords_num,
        tokenizer=tokenizer
    )

    # vllm part
    llm = LLM(model=args.model, 
              tensor_parallel_size=args.tensor_parallel_size, 
              max_model_len=args.max_model_len, 
              gpu_memory_utilization=args.gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        guided_decoding=GuidedDecodingParams(json=json_schema)
    )

    extracted_keywords = genereate_queries_using_llm(messages=messages, llm=llm, sampling_params=sampling_params)

    corpus = read_jsonl(args.corpus_path)
    with open(args.output_path, "w") as f:
        for doc, keywords in zip(corpus, extracted_keywords):
            doc_id = doc['_id']
            keywords_str = ", ".join(keywords)
            f.write(f"{doc_id}: {keywords_str}\n")

    
if __name__ == "__main__":
    main()
