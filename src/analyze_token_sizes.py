import json
import argparse
import statistics
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from utils.util import read_jsonl
from utils.data import combine_topic_info
from utils.constants import PROMPTAGATOR_SYS_PROMPT, PROMPTAGATOR_USER_PROMPT, D2Q_SYS_PROMPT_WITH_TOPIC, D2Q_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze token sizes for corpus and few-shot examples")
    parser.add_argument("--enhanced_topic_info_pkl", type=str, help="Path to the enhanced topic info pickle file")
    parser.add_argument("--corpus_path", type=str, help="Path to the corpus.jsonl file")
    parser.add_argument("--corpus_topics_path", type=str, help="Path to the corpus topics JSONL file")
    parser.add_argument("--few_shot_examples_path", type=str, default="/home/guest/r12922050/GitHub/d2qplus/prompts/promptagator/few_shot_examples.jsonl", help="Path to the few-shot examples file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name for tokenizer")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model length")
    parser.add_argument("--max_tokens_output", type=int, default=512, help="Maximum tokens for generation output")
    parser.add_argument("--output_dir", type=str, default="./token_analysis", help="Output directory for analysis results")
    parser.add_argument("--sample_size", type=int, default=1000, help="Sample size for analysis (0 = all data)")
    
    return parser.parse_args()

def analyze_document_tokens(documents: List[Dict], tokenizer, prompt_template: str = "promptagator") -> Dict:
    """Analyze token sizes for documents with different prompt templates"""
    doc_tokens = []
    user_prompt_tokens = []
    
    for doc in documents:
        text = doc['text']
        
        # Document text tokens
        doc_token_count = len(tokenizer.encode(text))
        doc_tokens.append(doc_token_count)
    
        # User prompt tokens (document + template)
        if prompt_template == "promptagator":
            user_prompt = PROMPTAGATOR_USER_PROMPT.replace("[DOCUMENT]", text)
        elif prompt_template == "d2q":
            user_prompt = USER_PROMPT_TEMPLATE.replace("[DOCUMENT]", text)
        else:
            user_prompt = text
            
        user_prompt_token_count = len(tokenizer.encode(user_prompt))
        user_prompt_tokens.append(user_prompt_token_count)
    
    return {
        "doc_tokens": doc_tokens,
        "user_prompt_tokens": user_prompt_tokens,
        "stats": {
            "doc_tokens": calculate_stats(doc_tokens),
            "user_prompt_tokens": calculate_stats(user_prompt_tokens)
        }
    }

def analyze_few_shot_examples(examples: List[Dict], tokenizer) -> Dict:
    """Analyze token sizes for few-shot examples"""
    example_tokens = []
    
    for example in examples:
        # Format as it would appear in the prompt
        example_text = f"Article: {example['doc_text']}\nQuery: {example['query_text']}\n\n"
        token_count = len(tokenizer.encode(example_text))
        example_tokens.append(token_count)
    
    return {
        "example_tokens": example_tokens,
        "stats": calculate_stats(example_tokens)
    }

def analyze_system_prompts(tokenizer, few_shot_examples: List[Dict]) -> Dict:
    """Analyze token sizes for different system prompts"""
    results = {}
    
    # Base system prompts
    prompts = {
        "d2q_basic": D2Q_SYSTEM_PROMPT,
        "d2q_with_topic": D2Q_SYS_PROMPT_WITH_TOPIC,
        "promptagator_base": PROMPTAGATOR_SYS_PROMPT
    }
    
    for name, prompt in prompts.items():
        token_count = len(tokenizer.encode(prompt))
        results[name] = token_count
    
    # Promptagator with different numbers of examples
    for num_examples in [1, 2, 3, 4, 5]:
        prompt = PROMPTAGATOR_SYS_PROMPT
        for example in few_shot_examples[:num_examples]:
            prompt += f"Article: {example['doc_text']}\nQuery: {example['query_text']}\n\n"
        
        token_count = len(tokenizer.encode(prompt))
        results[f"promptagator_with_{num_examples}_examples"] = token_count
    
    return results

def calculate_stats(values: List[int]) -> Dict:
    """Calculate statistical metrics for a list of values"""
    if not values:
        return {}
    
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
        "percentiles": {
            "25th": statistics.quantiles(values, n=4)[0] if len(values) >= 4 else min(values),
            "75th": statistics.quantiles(values, n=4)[2] if len(values) >= 4 else max(values),
            "90th": statistics.quantiles(values, n=10)[8] if len(values) >= 10 else max(values),
            "95th": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            "99th": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }
    }

def calculate_token_budget(system_tokens: int, user_tokens: int, max_model_len: int, max_output_tokens: int, safety_buffer: int = 50) -> Dict:
    """Calculate available tokens for few-shot examples"""
    base_tokens = system_tokens + user_tokens
    available_for_examples = max_model_len - base_tokens - max_output_tokens - safety_buffer
    
    return {
        "base_tokens": base_tokens,
        "available_for_examples": available_for_examples,
        "utilization_rate": base_tokens / max_model_len if max_model_len > 0 else 0
    }

def print_analysis_results(doc_analysis: Dict, example_analysis: Dict, system_analysis: Dict, max_model_len: int, max_output_tokens: int):
    """Print comprehensive analysis results"""
    print("=" * 80)
    print("TOKEN SIZE ANALYSIS REPORT")
    print("=" * 80)
    
    # Document analysis
    print("\n📄 DOCUMENT ANALYSIS")
    print("-" * 40)
    doc_stats = doc_analysis["stats"]["doc_tokens"]
    user_stats = doc_analysis["stats"]["user_prompt_tokens"]
    
    print(f"Document Text Tokens:")
    print(f"  Count: {doc_stats['count']:,}")
    print(f"  Mean: {doc_stats['mean']:.1f}")
    print(f"  Median: {doc_stats['median']:.1f}")
    print(f"  95th percentile: {doc_stats['percentiles']['95th']:.1f}")
    print(f"  Max: {doc_stats['max']:,}")
    
    print(f"\nUser Prompt Tokens (with template):")
    print(f"  Mean: {user_stats['mean']:.1f}")
    print(f"  Median: {user_stats['median']:.1f}")
    print(f"  95th percentile: {user_stats['percentiles']['95th']:.1f}")
    print(f"  Max: {user_stats['max']:,}")
    
    # Few-shot examples analysis
    print("\n📝 FEW-SHOT EXAMPLES ANALYSIS")
    print("-" * 40)
    ex_stats = example_analysis["stats"]
    print(f"Individual Example Tokens:")
    print(f"  Count: {ex_stats['count']}")
    print(f"  Mean: {ex_stats['mean']:.1f}")
    print(f"  Median: {ex_stats['median']:.1f}")
    print(f"  Max: {ex_stats['max']:,}")
    
    # Cumulative tokens for multiple examples
    cumulative = []
    for i in range(1, min(6, len(example_analysis["example_tokens"]) + 1)):
        cumulative_tokens = sum(example_analysis["example_tokens"][:i])
        cumulative.append(cumulative_tokens)
        print(f"  {i} examples: {cumulative_tokens:,} tokens")
    
    # System prompt analysis
    print("\n🔧 SYSTEM PROMPT ANALYSIS")
    print("-" * 40)
    for name, tokens in system_analysis.items():
        print(f"  {name}: {tokens:,} tokens")
    
    # Token budget recommendations
    print("\n💡 TOKEN BUDGET RECOMMENDATIONS")
    print("-" * 40)
    print(f"Model max length: {max_model_len:,} tokens")
    print(f"Reserved for output: {max_output_tokens:,} tokens")
    
    # Calculate budgets for different scenarios
    scenarios = [
        ("D2Q Basic", system_analysis["d2q_basic"], user_stats["median"]),
        ("D2Q with Topics", system_analysis["d2q_with_topic"], user_stats["median"]),
        ("Promptagator (no examples)", system_analysis["promptagator_base"], user_stats["median"]),
        ("Promptagator (2 examples)", system_analysis["promptagator_with_2_examples"], user_stats["median"]),
        ("Promptagator (4 examples)", system_analysis["promptagator_with_4_examples"], user_stats["median"])
    ]
    
    print("\nBudget analysis (using median document size):")
    for scenario_name, sys_tokens, user_tokens in scenarios:
        budget = calculate_token_budget(sys_tokens, user_tokens, max_model_len, max_output_tokens)
        print(f"\n  {scenario_name}:")
        print(f"    Base tokens: {budget['base_tokens']:,}")
        print(f"    Available for examples: {budget['available_for_examples']:,}")
        print(f"    Model utilization: {budget['utilization_rate']:.1%}")
        
        if budget['available_for_examples'] > 0 and example_analysis["stats"]["mean"] > 0:
            max_examples = int(budget['available_for_examples'] / example_analysis["stats"]["mean"])
            print(f"    Max examples (avg size): {max_examples}")

def save_analysis_to_json(results: Dict, output_path: str):
    """Save analysis results to JSON file"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Analysis results saved to: {output_path}")

def create_visualizations(doc_analysis: Dict, example_analysis: Dict, output_dir: str):
    """Create token distribution visualizations"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Document token distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(doc_analysis["doc_tokens"], bins=50, alpha=0.7, edgecolor='black')
    plt.title("Document Text Token Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.axvline(statistics.median(doc_analysis["doc_tokens"]), color='red', linestyle='--', label='Median')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.hist(doc_analysis["user_prompt_tokens"], bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.title("User Prompt Token Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.axvline(statistics.median(doc_analysis["user_prompt_tokens"]), color='red', linestyle='--', label='Median')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.hist(example_analysis["example_tokens"], bins=20, alpha=0.7, edgecolor='black', color='green')
    plt.title("Few-shot Example Token Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.axvline(statistics.median(example_analysis["example_tokens"]), color='red', linestyle='--', label='Median')
    plt.legend()
    
    # Box plot comparison
    plt.subplot(2, 2, 4)
    data = [doc_analysis["doc_tokens"], doc_analysis["user_prompt_tokens"], example_analysis["example_tokens"]]
    labels = ["Document Text", "User Prompt", "Few-shot Examples"]
    plt.boxplot(data, labels=labels)
    plt.title("Token Count Comparison")
    plt.ylabel("Token Count")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to: {output_dir}/token_distributions.png")

def main():
    args = parse_args()
    
    print("🔍 Starting token size analysis...")
    print(f"Model: {args.model_name}")
    print(f"Max model length: {args.max_model_len:,} tokens")
    
    # Initialize tokenizer
    print("🤖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    print("📂 Loading corpus data...")
    corpus = combine_topic_info(
        enhanced_topic_info_pkl=args.enhanced_topic_info_pkl,
        corpus_topics_path=args.corpus_topics_path,
        corpus_path=args.corpus_path
    )
    
    # Sample data if requested
    if args.sample_size > 0 and len(corpus) > args.sample_size:
        import random
        corpus = random.sample(corpus, args.sample_size)
        print(f"📊 Using random sample of {args.sample_size} documents")
    
    print("📝 Loading few-shot examples...")
    few_shot_examples = read_jsonl(args.few_shot_examples_path)
    
    # Perform analysis
    print("🔬 Analyzing document tokens...")
    doc_analysis = analyze_document_tokens(corpus, tokenizer, "promptagator")
    
    print("🔬 Analyzing few-shot examples...")
    example_analysis = analyze_few_shot_examples(few_shot_examples, tokenizer)
    
    print("🔬 Analyzing system prompts...")
    system_analysis = analyze_system_prompts(tokenizer, few_shot_examples)
    
    # Print results
    print_analysis_results(doc_analysis, example_analysis, system_analysis, args.max_model_len, args.max_tokens_output)
    
    # Save results
    results = {
        "model_name": args.model_name,
        "max_model_len": args.max_model_len,
        "max_tokens_output": args.max_tokens_output,
        "sample_size": len(corpus),
        "document_analysis": doc_analysis["stats"],
        "example_analysis": example_analysis["stats"],
        "system_prompt_analysis": system_analysis
    }
    
    save_analysis_to_json(results, f"{args.output_dir}/token_analysis.json")
    
    # Create visualizations
    try:
        create_visualizations(doc_analysis, example_analysis, args.output_dir)
    except ImportError:
        print("⚠️  Matplotlib/Seaborn not available. Skipping visualizations.")
    
    print("\n✅ Token analysis complete!")

if __name__ == "__main__":
    main()
