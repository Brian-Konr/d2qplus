import json
from typing import Any, List
from utils.util import read_jsonl, read_txt
from utils.data import combine_topic_info
from pydantic import BaseModel, Field
from utils.constants import D2Q_SYS_PROMPT_WITH_TOPIC, D2Q_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, PROMPTAGATOR_SYS_PROMPT, PROMPTAGATOR_USER_PROMPT, D2Q_FEW_SHOT_SYS_PROMPT_WITH_TOPIC
from vllm import LLM, SamplingParams
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate queries using VLLM")
    parser.add_argument("--enhanced_topic_info_pkl", type=str, help="Path to the enhanced topic info pickle file")
    parser.add_argument("--corpus_path", type=str, help="Path to the corpus.jsonl file")
    parser.add_argument("--corpus_topics_path", type=str, help="Path to the corpus topics JSONL file")
    
    parser.add_argument("--core_phrase_path", type=str, default=None, help="Path to the core phrases JSONL file (optional)")
    parser.add_argument("--use_core_phrases", type=int, default=0, help="Use extracted core phrases for dynamic prompt generation")
    
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
    parser.add_argument("--num_of_queries", type=int, default=5, help="Number of queries to generate per output")
    parser.add_argument("--total_target_queries", type=int, default=10, help="Target number of queries per document")
    parser.add_argument("--with_topic_keywords", action="store_true", default=False, help="Use topic and keyword information in the prompts")
    parser.add_argument("--with_topic_weights", action="store_true", default=True, help="Include topic weights in prompts (only for D2Q template)")
    parser.add_argument("--prompt_template", type=str, default="d2q", help="Prompt template to use for query generation") # d2q + with_topic_keywords = prompt template for D2Q with topic keywords information
    parser.add_argument("--few_shot_examples_path", type=str, default="/home/guest/r12922050/GitHub/d2qplus/prompts/promptagator/few_shot_examples.jsonl", help="Path to the few-shot examples file (if using promptagator / InPars template)")
    
    # Dynamic prompt generation parameters (for constructing 'prompt' field)
    parser.add_argument("--max_keywords", type=int, default=15, help="Maximum number of keywords for dynamic prompt generation")
    parser.add_argument("--max_topics", type=int, default=5, help="Maximum number of topics for dynamic prompt generation")
    parser.add_argument("--random_pick_keywords", action="store_true", default=False, help="Randomly pick keywords instead of taking top ones")
    parser.add_argument("--proportional_selection", action="store_true", default=True, help="Select keywords proportionally based on topic weights")
    
    # Document truncation parameters
    parser.add_argument("--max_doc_tokens", type=int, default=1024, help="Maximum number of tokens for document truncation")

    return parser.parse_args()


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
        proportional_selection: bool = True,
        num_of_queries: int = 5,
        # Document truncation parameters
        tokenizer = None,
        max_doc_tokens: int = 1024,
        use_extracted_core_phrases: bool = False,
    ) -> List[dict]:
    """
    make conversation messages for LLM to generate queries based on the provided documents.
    
    `with_topic_keywords`: if True, the system prompt and user prompt will include topic, keyword information.
    `with_topic_weights`: if True, the user prompt will include topic weights information. This is only used for D2Q prompt template.
    `prompt_template`: the template to use for generating queries ("d2q" or "promptagator")
    `max_keywords`, `max_topics`, etc.: Parameters for dynamic prompt generation when 'prompt' field is not available

    `use_extracted_core_phrases`: if True, use extracted core phrases for dynamic prompt generation (data should contain 'core_phrases' field)
    
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
            use_extracted_core_phrases=use_extracted_core_phrases,
        )
    for doc in data:
        text = doc['text']
        
        # Truncate document if tokenizer is provided
        if tokenizer is not None:
            text = truncate_document(text, tokenizer, max_doc_tokens)
        
        sys_prompt = {}
        user_prompt = {}

        if prompt_template == "d2q":
            sys_prompt = {"role": "system", "content": D2Q_SYS_PROMPT_WITH_TOPIC if with_topic_keywords else D2Q_SYSTEM_PROMPT}
            if with_topic_keywords:
                user_prompt = {"role": "user", "content": doc['prompt']}
            else:
                # Basic template without topic information
                user_prompt = {"role": "user", "content": USER_PROMPT_TEMPLATE}
        elif prompt_template == "d2q-fewshot-topics":
            
            prompt = D2Q_FEW_SHOT_SYS_PROMPT_WITH_TOPIC
            for example in few_shot_examples:
                prompt += f"Article: {example['doc_text']}\n\n"
                prompt += f"Query Set: {example['query_text']}\n\n"
            sys_prompt = {"role": "system", "content": prompt}
            if with_topic_keywords:
                user_prompt = {"role": "user", "content": doc['prompt']}
            else:
                # Basic template without topic information
                user_prompt = {"role": "user", "content": USER_PROMPT_TEMPLATE}
        elif prompt_template == "promptagator":
            prompt = PROMPTAGATOR_SYS_PROMPT
            for example in few_shot_examples:
                prompt += f"Article: {example['doc_text']}\n"
                prompt += f"Query: {example['query_text']}\n\n"
            sys_prompt = {"role": "system", "content": prompt}
            user_prompt = {"role": "user", "content": PROMPTAGATOR_USER_PROMPT}
        elif prompt_template == "plan-then-write-given-topics-plan":
            sys_template = read_txt("/home/guest/r12922050/GitHub/d2qplus/prompts/plan-then-write/given-toipcs-plan/system.txt")
            user_template = read_txt("/home/guest/r12922050/GitHub/d2qplus/prompts/plan-then-write/given-toipcs-plan/user.txt")
            sys_prompt = {"role": "system", "content": sys_template.replace("<num_of_queries>", str(num_of_queries))}
            user_prompt = {"role": "user", "content": user_template.replace("<doc_snippet>", text).replace("<topics>", doc.get("formatted_topics", "")).replace("<keywords>", doc.get("formatted_keywords", ""))}
        elif prompt_template == "plan-then-write-identify-then-plan":
            sys_template = read_txt("/home/guest/r12922050/GitHub/d2qplus/prompts/plan-then-write/identify-then-plan/system.txt")
            user_template = read_txt("/home/guest/r12922050/GitHub/d2qplus/prompts/plan-then-write/identify-then-plan/user.txt")
            sys_prompt = {"role": "system", "content": sys_template}
            user_prompt = {"role": "user", "content": user_template.replace("<doc_snippet>", text)}
        
        # replace num_of_queries placeholder in user prompt
        sys_prompt['content'] = sys_prompt['content'].replace("<num_of_queries>", str(num_of_queries))
        user_prompt['content'] = user_prompt['content'].replace("<num_of_queries>", str(num_of_queries)).replace("[DOCUMENT]", text)

        messages.append([sys_prompt, user_prompt])
    return messages


def generate_queries_vllm(messages: List[dict], llm: LLM, sampling_params: SamplingParams, prompt_template: str, target_queries: int = 50) -> List[str]:
    all_gen_q = []
    
    if prompt_template == "promptagator":
        # For single-query methods, run multiple times to reach target
        queries_per_run = sampling_params.n  # num_return_sequences
        num_runs = target_queries // queries_per_run
        
        for run in range(num_runs):
            outputs = llm.chat(messages, sampling_params)
            for i, output in enumerate(outputs):
                if run == 0:
                    all_gen_q.append([])
                for seq in output.outputs:
                    all_gen_q[i].append(seq.text.strip())
    else:
        # For multi-query methods (like D2Q), use existing logic
        outputs = llm.chat(messages, sampling_params)
        for i, output in enumerate(outputs):
            all_gen_q.append([])  # Initialize list for each document
            for seq in output.outputs:
                queries = seq.text.strip().split('\n')
                # Filter out empty strings and strip whitespace
                queries = [q.strip() for q in queries if q.strip()]
                all_gen_q[i].append(queries)
    
    return all_gen_q


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output_path

    #construct output directory if it does not exist
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"🔄 Loading corpus data...")
    print(f"  - Enhanced topic info: {args.enhanced_topic_info_pkl}")
    print(f"  - Corpus topics: {args.corpus_topics_path}")
    print(f"  - Corpus: {args.corpus_path}")
    
    corpus = read_jsonl(args.corpus_path)
    # replace original _id in corpus with doc_id
    for doc in corpus:
        doc["doc_id"] = doc.pop("_id", None)

    if args.with_topic_keywords:
        corpus = combine_topic_info(
            enhanced_topic_info_pkl=args.enhanced_topic_info_pkl, 
            corpus_topics_path=args.corpus_topics_path, 
            corpus_path=args.corpus_path,
            core_phrase_path=args.core_phrase_path
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
    
    # Initialize vllm
    print(f"🚀 Initializing VLLM with model: {args.model}")
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    
    # Get tokenizer for document truncation
    tokenizer = llm.get_tokenizer()
    
    # Create messages for vllm
    print(f"🛠️ Creating messages for generation...")
    print(f"📏 Document truncation: max {args.max_doc_tokens} tokens")
    messages = make_messages(
        corpus, 
        with_topic_keywords=args.with_topic_keywords,
        with_topic_weights=args.with_topic_weights,
        prompt_template=args.prompt_template, 
        few_shot_examples=few_shot_examples,
        max_keywords=args.max_keywords,
        max_topics=args.max_topics,
        random_pick_keywords=args.random_pick_keywords,
        proportional_selection=args.proportional_selection,
        num_of_queries=args.num_of_queries,
        tokenizer=tokenizer,
        max_doc_tokens=args.max_doc_tokens,
        use_extracted_core_phrases=True if args.use_core_phrases == 1 else False
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

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.return_sequence_num,
    )
    
    print(f"⚡ Starting query generation...")
    # Generate queries
    generated_q = generate_queries_vllm(messages, llm, sampling_params, args.prompt_template, args.total_target_queries)
    
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
