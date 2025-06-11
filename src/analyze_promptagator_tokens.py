#!/usr/bin/env python3
"""
Quick token analysis for Promptagator method.
Analyzes token usage and provides recommendations for few-shot example selection.
"""
import argparse
import statistics
from typing import List, Dict
from transformers import AutoTokenizer

from utils.util import read_jsonl
from utils.data import combine_topic_info
from utils.constants import PROMPTAGATOR_SYS_PROMPT, PROMPTAGATOR_USER_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Quick token analysis for Promptagator")
    parser.add_argument("--enhanced_topic_info_pkl", type=str, help="Enhanced topic info pickle")
    parser.add_argument("--corpus_path", type=str, help="Corpus JSONL file")
    parser.add_argument("--corpus_topics_path", type=str, help="Corpus topics JSONL file")
    parser.add_argument("--few_shot_examples_path", type=str,
                        default="/home/guest/r12922050/GitHub/d2qplus/prompts/promptagator/few_shot_examples.jsonl",
                        help="Few-shot examples file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Max model length")
    parser.add_argument("--max_output_tokens", type=int, default=128, help="Max output tokens")
    parser.add_argument("--sample_size", type=int, default=100, help="Sample size for analysis")
    
    return parser.parse_args()


def analyze_promptagator_tokens(
    documents: List[Dict],
    few_shot_examples: List[Dict],
    tokenizer,
    max_model_len: int,
    max_output_tokens: int
):
    """Analyze token usage for Promptagator method."""
    
    print("🔍 Analyzing token usage for Promptagator method")
    print("=" * 60)
    
    # Analyze few-shot examples
    example_tokens = []
    for example in few_shot_examples:
        example_text = f"Article: {example['doc_text']}\nQuery: {example['query_text']}\n\n"
        tokens = len(tokenizer.encode(example_text))
        example_tokens.append(tokens)
    
    print("📝 Few-shot examples analysis:")
    print(f"  Count: {len(example_tokens)}")
    print(f"  Mean: {statistics.mean(example_tokens):.1f} tokens")
    print(f"  Median: {statistics.median(example_tokens):.1f} tokens")
    print(f"  Min: {min(example_tokens)} tokens")
    print(f"  Max: {max(example_tokens)} tokens")
    
    # Cumulative tokens for multiple examples
    cumulative_tokens = []
    for i in range(1, len(example_tokens) + 1):
        cumulative = sum(example_tokens[:i])
        cumulative_tokens.append(cumulative)
        print(f"  {i} examples: {cumulative} tokens")
    
    # Base system prompt
    base_system_tokens = len(tokenizer.encode(PROMPTAGATOR_SYS_PROMPT))
    print(f"\n🔧 Base system prompt: {base_system_tokens} tokens")
    
    # Sample document analysis
    sample_docs = documents[:min(len(documents), 20)]  # Analyze first 20 docs
    doc_tokens = []
    user_prompt_tokens = []
    total_base_tokens = []
    
    for doc in sample_docs:
        text = doc['text']
        doc_token_count = len(tokenizer.encode(text))
        doc_tokens.append(doc_token_count)
        
        user_prompt = PROMPTAGATOR_USER_PROMPT.replace("[DOCUMENT]", text)
        user_token_count = len(tokenizer.encode(user_prompt))
        user_prompt_tokens.append(user_token_count)
        
        base_total = base_system_tokens + user_token_count
        total_base_tokens.append(base_total)
    
    print(f"\n📄 Document analysis (sample of {len(sample_docs)}):")
    print(f"  Document tokens - Mean: {statistics.mean(doc_tokens):.1f}, Median: {statistics.median(doc_tokens):.1f}")
    print(f"  User prompt tokens - Mean: {statistics.mean(user_prompt_tokens):.1f}, Median: {statistics.median(user_prompt_tokens):.1f}")
    print(f"  Base total tokens - Mean: {statistics.mean(total_base_tokens):.1f}, Median: {statistics.median(total_base_tokens):.1f}")
    
    # Calculate available tokens for examples
    median_base = statistics.median(total_base_tokens)
    safety_buffer = 100
    available_for_examples = max_model_len - median_base - max_output_tokens - safety_buffer
    
    print(f"\n💡 Token budget analysis:")
    print(f"  Max model length: {max_model_len:,}")
    print(f"  Median base tokens: {median_base:.0f}")
    print(f"  Output tokens: {max_output_tokens}")
    print(f"  Safety buffer: {safety_buffer}")
    print(f"  Available for examples: {available_for_examples:.0f}")
    
    if available_for_examples > 0:
        # Recommend number of examples
        mean_example_tokens = statistics.mean(example_tokens)
        recommended_examples = int(available_for_examples / mean_example_tokens)
        recommended_examples = min(recommended_examples, len(few_shot_examples))
        
        print(f"\n📊 Recommendations:")
        print(f"  Mean example size: {mean_example_tokens:.1f} tokens")
        print(f"  Recommended max examples: {recommended_examples}")
        
        # Show token usage for recommended number
        if recommended_examples > 0:
            actual_example_tokens = sum(example_tokens[:recommended_examples])
            total_usage = median_base + actual_example_tokens + max_output_tokens
            utilization = total_usage / max_model_len
            
            print(f"  With {recommended_examples} examples:")
            print(f"    Example tokens: {actual_example_tokens}")
            print(f"    Total usage: {total_usage:.0f} / {max_model_len} ({utilization:.1%})")
    else:
        print(f"⚠️  WARNING: No tokens available for examples!")
        print(f"  Consider reducing max_output_tokens or using shorter documents")


def main():
    args = parse_args()
    
    print(f"🤖 Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print("📂 Loading data...")
    corpus = combine_topic_info(
        enhanced_topic_info_pkl=args.enhanced_topic_info_pkl,
        corpus_topics_path=args.corpus_topics_path,
        corpus_path=args.corpus_path
    )
    
    few_shot_examples = read_jsonl(args.few_shot_examples_path)
    
    # Sample data if needed
    if args.sample_size > 0 and len(corpus) > args.sample_size:
        import random
        corpus = random.sample(corpus, args.sample_size)
    
    print(f"✅ Loaded {len(corpus)} documents and {len(few_shot_examples)} examples")
    
    analyze_promptagator_tokens(
        documents=corpus,
        few_shot_examples=few_shot_examples,
        tokenizer=tokenizer,
        max_model_len=args.max_model_len,
        max_output_tokens=args.max_output_tokens
    )


if __name__ == "__main__":
    main()
