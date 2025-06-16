import re, os
import argparse
import sys
from typing import List

# Third-party imports
import pandas as pd

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel

from .util import read_jsonl
from .data import combine_topic_info

from transformers import AutoTokenizer
from .constants import LLM_EXTRACT_KEYWORD_USER_PROMPT

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

def construct_prompt_messages(enhanced_corpus, docid2keywords: str, tokenizer, final_extract_keyword_num: int = 10) -> List[List]:
    """
    docid2keywords: dict mapping document IDs to lists of keywords (extracted by KeyBERT)
    """
    # construct messages list
    messages = []
    for doc in enhanced_corpus:
        document = truncate_document(doc['text'], tokenizer, max_tokens=1024)
        candidate_kws = [kw[0] for kw in docid2keywords[doc['doc_id']]]
        # for each document, select top 3 topics' topic-level keywords
        sorted_topics = sorted(doc['topics'], key=lambda x: x['weight'], reverse=True)[:3]
        for topic in sorted_topics:
            candidate_kws.extend(topic['Representation'])


        user_template = LLM_EXTRACT_KEYWORD_USER_PROMPT


        user_content = user_template.replace("<document>", document).replace("<keywords>", ", ".join(candidate_kws)).replace("<final_keyword_num>", str(final_extract_keyword_num))
        messages.append(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that selects the most relevant keywords from a candidate keyword set for a given document."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]

        )
    return messages

def extract_keywords_from_json(json_str: str) -> List[str]:
    """Extract keywords from a JSON string."""
    try:
        data = KeywordSet.model_validate_json(json_str)
        return data.keywords
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []

def extract_keywords_using_llm_with_retry(
        messages: List[List],
        doc_keywords: List[List],
        llm: LLM,
        sampling_params: SamplingParams,
        max_retries: int = 3
    ) -> List[str]:

    keywords = []
    outputs = llm.chat(messages, sampling_params)
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        extracted_keywords = extract_keywords_from_json(generated_text)
        
        if extracted_keywords:
            keywords.append(extracted_keywords)
        else:
            print(f"Failed to extract keywords from output {i}, retrying with different sampling...")
            
            # Retry logic with max_retries
            success = False
            for retry_count in range(max_retries):
                print(f"Retry attempt {retry_count + 1}/{max_retries} for document {i}")
                
                # Use different sampling parameters for each retry
                retry_sampling = SamplingParams(
                    temperature=max(0.01, sampling_params.temperature - 0.1 * (retry_count + 1)),  # Decrease temperature
                    max_tokens=sampling_params.max_tokens,
                    guided_decoding=sampling_params.guided_decoding
                )
                
                retry_output = llm.chat([messages[i]], retry_sampling)
                retry_text = retry_output[0].outputs[0].text
                retry_keywords = extract_keywords_from_json(retry_text)
                
                if retry_keywords:
                    keywords.append(retry_keywords)
                    success = True
                    print(f"Retry successful on attempt {retry_count + 1}")
                    break
            
            if not success:
                print(f"All {max_retries} retries failed for document {i}, using empty list")
                keywords.append(doc_keywords[i])
    
    return keywords



def main():
    parser = argparse.ArgumentParser(description="Extract keywords using VLLM")
    
    parser.add_argument("--topic_info_pkl", type=str,
                        help="Path to the topic information pickle file")
    parser.add_argument("--keywords_path", type=str,
                        help="Path to the keybert extracted file (keywords per document)")
    parser.add_argument("--corpus_topics_path", type=str,
                        default="/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus_topics.jsonl",
                        help="Path to the corpus topics JSONL file")
    parser.add_argument("--corpus_path", type=str,
                        default="/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl",
                        help="Path to the corpus JSONL file")
    parser.add_argument("--output_path", type=str,required=True,
                        help="Path to save the extracted keywords")

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
    
    docid2keywords = pd.read_pickle(args.keywords_path)
    messages = construct_prompt_messages(
        corpus, 
        docid2keywords=docid2keywords, 
        final_extract_keyword_num=args.final_extract_keywords_num,
        tokenizer=tokenizer
    )

    # print out some messages to check
    for i, message in enumerate(messages[:3]):
        print(f"Message {i}:")
        for part in message:
            print(f"  {part['role']}: {part['content']}")
    
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

    # construct KeyBERT extracted keywords list (list of keywords for each document) for fallback if needed
    doc_keywords = [[kw[0] for kw in docid2keywords[doc['doc_id']]] for doc in corpus]

    extracted_keywords = extract_keywords_using_llm_with_retry(messages=messages, doc_keywords=doc_keywords, llm=llm, sampling_params=sampling_params)

    corpus = read_jsonl(args.corpus_path)

    import json
    with open(args.output_path, "w") as f:
        for doc, keywords in zip(corpus, extracted_keywords):
            doc_id = doc['_id']
            entry = {"doc_id": doc_id, "text": doc['text'], "keywords": keywords}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    
if __name__ == "__main__":
    main()
