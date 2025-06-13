import re, os
import argparse
from typing import List

# Third-party imports
import pandas as pd
import sys

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from util import read_jsonl
from pydantic import BaseModel

class KeywordSet(BaseModel):
    keywords: list[str]

json_schema = KeywordSet.model_json_schema()

def construct_prompt_messages(candidate_keywords_pkl: str, corpus_path: str, size: int=5):
    # construct messages list
    corpus = read_jsonl(corpus_path)  # List of documents, each document is a dict with 'docid' and 'text'
    docid2text = {doc['_id']: doc['text'] for doc in corpus}
    docid2candidates = pd.read_pickle(candidate_keywords_pkl) #Topic, Count, Name, Representation (list of keywords), Representative_Docs (list of sentences)
    messages = []
    i = 0
    for docid, keywords in docid2candidates.items():
        if i >= size:
            break
        i += 1
        
        document = docid2text[docid]
        user_template = "You will receive a document along with a set of candidate keywords. Your task is to select the keywords that best align with the core theme of the document. Exclude keywords that are too broad or less relevant. You may list up to 5 keywords, using only the keywords in the candidate set.\n\nDocument: <document>\nCandidate keyword set: <keywords>\n\nWhen you reply, **only** output a JSON object of the form:\n```json\n{\"keywords\": [\"kw1\", \"kw2\", …]}\n```\nDo **not** include any additional explanations or text—just the JSON."
        user_content = user_template.replace("<document>", document).replace("<keywords>", ", ".join([kw[0] for kw in keywords]))
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

def extract_keywords_from_json(json_str: str) -> List[str]:
    """Extract keywords from a JSON string."""
    try:
        data = KeywordSet.model_validate_json(json_str)
        return data.keywords
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []
def extract_keywords_using_llm(
        messages: List[List],
        llm: LLM,
        sampling_params: SamplingParams
    ) -> List[str]:

    keywords = []
    outputs = llm.chat(messages, sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        extracted_keywords = extract_keywords_from_json(generated_text)
        if extracted_keywords:
            keywords.append(extracted_keywords)
        else:
            print(f"Failed to extract keywords from output {i}: {generated_text}")
            keywords.append([])
    return keywords



def main():
    parser = argparse.ArgumentParser(description="Extract keywords using VLLM")
    
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
    
    args = parser.parse_args()
    
    messages = construct_prompt_messages("/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/keywords/candidate_keywords_scibert.pkl", "/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl")

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

    extracted_keywords = extract_keywords_using_llm(messages=messages, llm=llm, sampling_params=sampling_params)
    with open("/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/keywords/extracted_keywords.jsonl", "w") as f:
        for keywords in extracted_keywords:
            f.write(f"{keywords}\n")

    
if __name__ == "__main__":
    main()
