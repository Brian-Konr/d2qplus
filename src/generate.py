import json
from typing import Any, List
from utils.util import read_jsonl
from utils.data import combine_topic_info
from pydantic import BaseModel, Field
from utils.constants import D2Q_SYS_PROMPT_WITH_TOPIC, D2Q_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, PROMPTAGATOR_SYS_PROMPT, PROMPTAGATOR_USER_PROMPT
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate queries using VLLM")
    parser.add_argument("--enhanced_topic_info_pkl", type=str, help="Path to the enhanced topic info pickle file")
    parser.add_argument("--corpus_path", type=str, help="Path to the corpus.jsonl file")
    parser.add_argument("--corpus_topics_path", type=str, help="Path to the corpus topics JSONL file")
    parser.add_argument("--output_path", type=str, help="Path to save the generated queries")

    # test
    parser.add_argument("--test", action="store_true", default=False, help="Run in test mode")

    # vllm config
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model length")

    # vllm sampling parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens for generation")
    parser.add_argument("--return_sequence_num", type=int, default=1, help="Number of return sequences")

    # prompt parameters
    parser.add_argument("--with_topic_keywords", action="store_true", default=False, help="Use topic and keyword information in the prompts")
    parser.add_argument("--with_topic_weights", action="store_true", default=True, help="Include topic weights in prompts (only for D2Q template)")
    parser.add_argument("--prompt_template", type=str, choices=["d2q", "promptagator", "inpars"], default="d2q", help="Prompt template to use for query generation") # d2q + with_topic_keywords = prompt template for D2Q with topic keywords information
    parser.add_argument("--few_shot_examples_path", type=str, default=None, help="Path to the few-shot examples file (if using promptagator / InPars template)")
    
    # Dynamic prompt generation parameters (for constructing 'prompt' field)
    parser.add_argument("--max_keywords", type=int, default=15, help="Maximum number of keywords for dynamic prompt generation")
    parser.add_argument("--max_topics", type=int, default=5, help="Maximum number of topics for dynamic prompt generation")
    parser.add_argument("--random_pick_keywords", action="store_true", default=False, help="Randomly pick keywords instead of taking top ones")
    parser.add_argument("--proportional_selection", action="store_true", default=True, help="Select keywords proportionally based on topic weights")

    return parser.parse_args()

def make_messages(
        data: List[dict], 
        with_topic_keywords: bool,
        with_topic_weights: bool = True,  
        prompt_template: str = "d2q", 
        few_shot_examples: List[Any] = [],
        # Parameters for dynamic prompt generation (when 'prompt' field is not available)
        max_keywords: int = 15,
        max_topics: int = 5,
        random_pick_keywords: bool = False,
        proportional_selection: bool = True
    ) -> List[dict]:
    """
    make conversation messages for LLM to generate queries based on the provided documents.
    
    `with_topic_keywords`: if True, the system prompt and user prompt will include topic, keyword information.
    `with_topic_weights`: if True, the user prompt will include topic weights information. This is only used for D2Q prompt template.
    `prompt_template`: the template to use for generating queries ("d2q" or "promptagator")
    `max_keywords`, `max_topics`, etc.: Parameters for dynamic prompt generation when 'prompt' field is not available
    
    `data`: The data is expected to be a list of dictionaries, where each dictionary contains:
    - text: the document text
    - prompt (optional): the organized user prompt (can be obtained by running `prepare_prompts` in `utils/data.py`)
    - topics (optional): topic information for dynamic prompt generation
    - title (optional): document title for dynamic prompt generation
    """
    from utils.data import prepare_prompts  # Import here to avoid circular import
    
    messages = []
    
    if with_topic_keywords:
        print("Generating prompts dynamically using prepare_prompts...")
        data = prepare_prompts(
            data.copy(),  # Don't modify original data
            max_keywords=max_keywords,
            max_topics=max_topics,
            random_pick_keywords=random_pick_keywords,
            proportional_selection=proportional_selection,
            with_topic_weights=with_topic_weights,
        )
    for doc in data:
        text = doc['text']
        sys_prompt = {}
        user_prompt = {}
        if prompt_template == "d2q":
            sys_prompt = {"role": "system", "content": D2Q_SYS_PROMPT_WITH_TOPIC if with_topic_keywords else D2Q_SYSTEM_PROMPT}
            if with_topic_keywords:
                user_prompt = {"role": "user", "content": doc['prompt']}
            else:
                # Basic template without topic information
                user_prompt = {"role": "user", "content": USER_PROMPT_TEMPLATE.replace("[DOCUMENT]", text)}
        elif prompt_template == "promptagator":
            prompt = PROMPTAGATOR_SYS_PROMPT
            for example in few_shot_examples:
                prompt += f"Article: {example['doc_text']}\n"
                prompt += f"Query: {example['query_text']}\n\n"
            sys_prompt = {"role": "system", "content": prompt}
            user_prompt = {"role": "user", "content": PROMPTAGATOR_USER_PROMPT.replace("[DOCUMENT]", text)}
            
        messages.append([sys_prompt, user_prompt])
    return messages

def generate_queries_vllm(messages: List[dict], llm: LLM, sampling_params: SamplingParams) -> List[str]:
    all_gen_q = []
    outputs = llm.chat(messages, sampling_params)
    for output in outputs:
        gen_q = []
        for seq in output.outputs:
            gen_q.append(seq.text.strip())
        all_gen_q.append(gen_q)
    return all_gen_q


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output_path
    
    print(f"🔄 Loading corpus data...")
    print(f"  - Enhanced topic info: {args.enhanced_topic_info_pkl}")
    print(f"  - Corpus topics: {args.corpus_topics_path}")
    print(f"  - Corpus: {args.corpus_path}")
    
    corpus = combine_topic_info(
        enhanced_topic_info_pkl=args.enhanced_topic_info_pkl, 
        corpus_topics_path=args.corpus_topics_path, 
        corpus_path=args.corpus_path
    )
    
    print(f"✅ Loaded {len(corpus)} documents")
    print(f"📋 Generation settings:")
    print(f"  - Model: {args.model}")
    print(f"  - With topic keywords: {args.with_topic_keywords}")
    print(f"  - With topic weights: {args.with_topic_weights}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Max tokens: {args.max_tokens}")

    few_shot_examples = []
    if args.few_shot_examples_path:
        few_shot_examples = read_jsonl(args.few_shot_examples_path)
        print(f"📝 Loaded {len(few_shot_examples)} few-shot examples")
    
    # Create messages for vllm
    print(f"🛠️ Creating messages for generation...")
    messages = make_messages(
        corpus, 
        with_topic_keywords=args.with_topic_keywords,
        with_topic_weights=args.with_topic_weights,
        prompt_template=args.prompt_template, 
        few_shot_examples=few_shot_examples,
        max_keywords=args.max_keywords,
        max_topics=args.max_topics,
        random_pick_keywords=args.random_pick_keywords,
        proportional_selection=args.proportional_selection
    )
    
    print(f"✅ Created {len(messages)} messages")

    if args.test:
        # Test mode: only process the first 10 documents
        corpus = corpus[:10]
        messages = messages[:10]
        # save messages to jsonl to check
        with open("test_messages.jsonl", 'w') as f:
            for message in messages:
                f.write(json.dumps(message) + '\n')
        print("🧪 Test mode: only processing the first 10 documents.")
        print(f"💾 Test messages saved to test_messages.jsonl for verification.")

    # Initialize vllm
    print(f"🚀 Initializing VLLM with model: {args.model}")
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.return_sequence_num,
    )
    
    print(f"⚡ Starting query generation...")
    # Generate queries
    generated_q = generate_queries_vllm(messages, llm, sampling_params)
    
    print(f"✅ Generated queries for {len(generated_q)} documents")

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
    print(f"💾 Generated queries saved to {output_path}")
