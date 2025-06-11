#!/usr/bin/env python3
"""
Minimal Promptagator implementation for query generation.
Uses token analysis to dynamically select appropriate number of few-shot examples.
"""
import json
import argparse
from typing import List, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.util import read_jsonl
from utils.data import combine_topic_info
from utils.constants import PROMPTAGATOR_SYS_PROMPT, PROMPTAGATOR_USER_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Generate queries using Promptagator method")
    
    # Data paths
    parser.add_argument("--enhanced_topic_info_pkl", type=str, help="Path to enhanced topic info pickle")
    parser.add_argument("--corpus_path", type=str, help="Path to corpus.jsonl file")
    parser.add_argument("--corpus_topics_path", type=str, help="Path to corpus topics JSONL file")
    parser.add_argument("--few_shot_examples_path", type=str, 
                        default="/home/guest/r12922050/GitHub/d2qplus/prompts/promptagator/few_shot_examples.jsonl",
                        help="Path to few-shot examples file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for generated queries")
    
    # Model config
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model length")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    
    # Generation config
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens for generation")
    parser.add_argument("--target_queries_per_doc", type=int, default=5, help="Target queries per document")
    
    # Token management
    parser.add_argument("--safety_buffer", type=int, default=100, help="Safety buffer for token calculation")
    parser.add_argument("--auto_select_examples", action="store_true", default=True, 
                        help="Automatically select number of few-shot examples based on token limits")
    parser.add_argument("--max_examples", type=int, default=4, help="Maximum few-shot examples to use")
    
    # Test mode
    parser.add_argument("--test", action="store_true", help="Test mode with first 10 docs")
    
    return parser.parse_args()


def calculate_tokens(text: str, tokenizer) -> int:
    """Calculate token count for given text."""
    return len(tokenizer.encode(text))


def select_few_shot_examples(
    few_shot_examples: List[Dict],
    document_text: str,
    tokenizer,
    max_model_len: int,
    max_output_tokens: int,
    safety_buffer: int = 100,
    max_examples: int = 4
) -> List[Dict]:
    """
    Select optimal number of few-shot examples that fit within token limits.
    """
    # Calculate base prompt tokens
    base_system_prompt = PROMPTAGATOR_SYS_PROMPT
    user_prompt = PROMPTAGATOR_USER_PROMPT.replace("[DOCUMENT]", document_text)
    
    base_tokens = calculate_tokens(base_system_prompt, tokenizer) + calculate_tokens(user_prompt, tokenizer)
    available_tokens = max_model_len - base_tokens - max_output_tokens - safety_buffer
    
    if available_tokens <= 0:
        print(f"⚠️  Warning: No tokens available for examples. Base: {base_tokens}, Need: {max_output_tokens}")
        return []
    
    # Add examples until we hit the limit
    selected_examples = []
    current_example_tokens = 0
    
    for i, example in enumerate(few_shot_examples[:max_examples]):
        example_text = f"Article: {example['doc_text']}\nQuery: {example['query_text']}\n\n"
        example_tokens = calculate_tokens(example_text, tokenizer)
        
        if current_example_tokens + example_tokens > available_tokens:
            break
            
        selected_examples.append(example)
        current_example_tokens += example_tokens
        
    print(f"📝 Selected {len(selected_examples)}/{min(len(few_shot_examples), max_examples)} examples ({current_example_tokens} tokens)")
    return selected_examples


def create_promptagator_messages(
    documents: List[Dict],
    few_shot_examples: List[Dict],
    tokenizer,
    max_model_len: int,
    max_output_tokens: int,
    safety_buffer: int = 100,
    auto_select_examples: bool = True,
    max_examples: int = 4
) -> List[List[Dict]]:
    """Create conversation messages for Promptagator method."""
    messages = []
    
    for doc in documents:
        text = doc['text']
        
        # Select appropriate examples for this document
        if auto_select_examples:
            selected_examples = select_few_shot_examples(
                few_shot_examples, text, tokenizer, max_model_len, 
                max_output_tokens, safety_buffer, max_examples
            )
        else:
            selected_examples = few_shot_examples[:max_examples]
        
        # Build system prompt with examples
        system_content = PROMPTAGATOR_SYS_PROMPT
        for example in selected_examples:
            system_content += f"Article: {example['doc_text']}\nQuery: {example['query_text']}\n\n"
        
        sys_prompt = {"role": "system", "content": system_content}
        user_prompt = {"role": "user", "content": PROMPTAGATOR_USER_PROMPT.replace("[DOCUMENT]", text)}
        
        messages.append([sys_prompt, user_prompt])
    
    return messages


def generate_queries(
    messages: List[List[Dict]], 
    llm: LLM, 
    target_queries_per_doc: int,
    temperature: float = 0.7,
    max_tokens: int = 128
) -> List[List[str]]:
    """Generate queries using VLLM with multiple runs to reach target count."""
    all_queries = []
    
    # Promptagator generates one query per run, so we need multiple runs
    for run in range(target_queries_per_doc):
        print(f"🔄 Generation run {run + 1}/{target_queries_per_doc}")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=1  # One sequence per run
        )
        
        outputs = llm.chat(messages, sampling_params)
        
        for i, output in enumerate(outputs):
            if run == 0:
                all_queries.append([])
            
            # Extract query from output
            query = output.outputs[0].text.strip()
            all_queries[i].append(query)
    
    return all_queries


def main():
    args = parse_args()
    
    print("🚀 Starting Promptagator query generation")
    print(f"Model: {args.model}")
    print(f"Target queries per doc: {args.target_queries_per_doc}")
    
    # Load tokenizer for token analysis
    print("🤖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load data
    print("📂 Loading corpus data...")
    corpus = combine_topic_info(
        enhanced_topic_info_pkl=args.enhanced_topic_info_pkl,
        corpus_topics_path=args.corpus_topics_path,
        corpus_path=args.corpus_path
    )
    
    print("📝 Loading few-shot examples...")
    few_shot_examples = read_jsonl(args.few_shot_examples_path)
    
    print(f"✅ Loaded {len(corpus)} documents and {len(few_shot_examples)} examples")
    
    # Test mode
    if args.test:
        corpus = corpus[:10]
        print("🧪 Test mode: Using first 10 documents")
    
    # Create messages
    print("🛠️  Creating messages with token-aware example selection...")
    messages = create_promptagator_messages(
        documents=corpus,
        few_shot_examples=few_shot_examples,
        tokenizer=tokenizer,
        max_model_len=args.max_model_len,
        max_output_tokens=args.max_tokens,
        safety_buffer=args.safety_buffer,
        auto_select_examples=args.auto_select_examples,
        max_examples=args.max_examples
    )
    
    # Initialize model
    print("🚀 Initializing VLLM...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    
    # Generate queries
    print("⚡ Generating queries...")
    generated_queries = generate_queries(
        messages=messages,
        llm=llm,
        target_queries_per_doc=args.target_queries_per_doc,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Save results
    print("💾 Saving results...")
    output_data = []
    for doc, queries in zip(corpus, generated_queries):
        output_data.append({
            "id": doc["doc_id"],
            "title": doc.get("title", ""),
            "text": doc["text"],
            "predicted_queries": queries
        })
    
    with open(args.output_path, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Generated {len(generated_queries)} documents with queries")
    print(f"📊 Average queries per doc: {sum(len(q) for q in generated_queries) / len(generated_queries):.1f}")
    print(f"💾 Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
