import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from typing import List, Tuple
from vllm import LLM, SamplingParams
import re
import json
import os
from openai import OpenAI
from pydantic import BaseModel
import concurrent.futures

# use pyterrier to get dataset
import pyterrier as pt
if not pt.started():
    pt.init(version='5.7', helper_version='0.0.7')
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate queries from documents using Doc2Query or Doc2QueryLLM')
    
    # General arguments
    parser.add_argument('--engine', type=str, choices=['doc2query', 'llm', 'reasoning_llm'], default='llm',
                        help='Query generation engine to use (doc2query or llm)')
    parser.add_argument('--num_examples', type=int, default=20,
                        help='Number of queries to generate per document')
    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum input length in tokens')
    parser.add_argument('--max_output_length', type=int, default=64,
                        help='Maximum output length in tokens')
    parser.add_argument('--save_file', type=str, required=True,
                        help='Path to save generated queries')
    parser.add_argument('--dataset', type=str, default='irds:beir/scifact',
                        help='Dataset to use for document corpus in pt format')
    
    # Doc2Query specific arguments
    parser.add_argument('--model', type=str, choices=['macavaney/doc2query-t5-base-msmarco', 'meta-llama/Llama-3.2-1B-Instruct'], default='meta-llama/Llama-3.2-1B-Instruct',
                        help='Model for Query Generation')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top k parameter for Doc2Query generation')

    parser.add_argument('--use_few_shot', action='store_true',
                        help='Whether to use few-shot learning')
    parser.add_argument('--few_shot_examples', type=str, default=None,
                        help='Path to few-shot examples file (jsonl)')
    
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Number of workers for concurrent processing')
    
    parser.add_argument('--base_url', type=str, default="http://localhost:8000/v1",
                        help='Base URL for the reasoning LLM API')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to save log file')
    parser.add_argument('--test', action='store_true',
                        help='Whether to run in test mode')

    return parser.parse_args()



class Doc2Query:
    """Generates pseudo queries from document text."""
    def __init__(self, 
        model,
        num_examples,
        max_input_length,
        max_output_length,
        top_k):
        """
        Args:
            model: The model checkpoint to use (default: 'macavaney/doc2query-t5-base-msmarco').
            num_examples: Number of queries to generate per document.
            device: Device for inference (e.g., 'cuda', 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        """
        self.num_examples = num_examples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5TokenizerFast.from_pretrained(model)
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        self.pattern = re.compile(r"^\s*http\S+")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.top_k = top_k

    def generate_queries(self, document: str) -> List[str]:
        """
        Generates pseudo queries for a given document.
        
        Args:
            document: The input document text.
            
        Returns:
            List of generated queries as strings.
        """
        # Clean document by removing URLs
        document = re.sub(self.pattern, "", document)
        
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                document,
                max_length=self.max_input_length,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).input_ids.to(self.device)
            
            # Generate queries
            outputs = self.model.generate(
                input_ids=inputs,
                max_length=self.max_output_length,
                do_sample=True,
                top_k=self.top_k,
                num_return_sequences=self.num_examples
            )
            
            # Decode outputs to strings
            queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return queries
    



class Doc2QueryLLM:
    """Generates pseudo queries from document text using a large language model (LLM)."""
    def __init__(self, 
        model,
        num_examples,
        max_input_length,
        max_output_length,
        use_few_shot,
        top_k,
        few_shot_examples):
        """
        Args:
            model: The model checkpoint to use (default: 'meta-llama/Llama-3.2-1B-Instruct').
            num_examples: Number of queries to generate per document.
            device: Device for inference (e.g., 'cuda', 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
            use_few_shot: Whether to use few-shot learning (default: False).
            few_shot_examples: Path to few-shot examples file (jsonl), Each line should be a json object with keys "doc_text" and "query".
        """
        self.llm = LLM(model=model, dtype="bfloat16")
        self.tokenizer = self.llm.get_tokenizer()

        self.num_examples = num_examples
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.use_few_shot = use_few_shot

        self.sample_params = SamplingParams(
            n=num_examples,
            max_tokens=max_output_length,
            top_k=top_k
        )
        if use_few_shot:
            if few_shot_examples is None:
                raise ValueError("Few-shot examples file must be provided if use_few_shot is True.")
            self.few_shot_examples = []
            with open(few_shot_examples, 'r') as f:
                for line in f:
                    example = json.loads(line)
                    self.few_shot_examples.append(example)
    
    def _prepare_chat_messages_for_all_documents(self, documents: List[str]):
        messages = []
        for document in documents:
            # truncate document to max_input_length, unit is tokens
            if self.max_input_length != 0:
                document = self._truncate_document(document)

            prompt = f"Generate a relevant claim for the following document without any additional context: \nDocument: {document}\nClaim: "
    
            if self.use_few_shot:
                few_shot_prompt = "Your task is to generate a relevant query based on the given document. The following are examples:\n"
                for example in self.few_shot_examples:
                    few_shot_prompt += f"Document: {example['doc_text']}\Claim: {example['query']}\n\n"
                prompt = few_shot_prompt + prompt
            
            # apply chat template
            messages.append([
                {"role": "user", "content": prompt}
            ])
        return messages
    def generate_queries(self, documents: List[str]) -> List[str]:
        """ˆ
        use vllm to generate queries
        """
        messages = self._prepare_chat_messages_for_all_documents(documents)
        
        outputs = self.llm.chat(messages, sampling_params=self.sample_params)

        return outputs
    
    def _truncate_document(self, document: str) -> str:
        """
        Truncate the document to the maximum input length.
        """
        tokenized_content = self.tokenizer(document, truncation=True, max_length=self.max_input_length, return_tensors="pt")
        truncated_content = self.tokenizer.decode(tokenized_content['input_ids'][0], skip_special_tokens=True)
        return truncated_content

class Doc2QueryReasoningLLM:
    def __init__(self, base_url: str = "http://localhost:8000/v1", logger = None):
        self.client = OpenAI(
            api_key="EMPTY", # self hosted does not need this
            base_url=base_url,
        )
        models = self.client.models.list()
        self.model = models.data[0].id
        self.logger = logger
    def generate_queries(self, document: str) -> Tuple[List[str], str]:
        class QuerySet(BaseModel):
            queries: list[str]

        json_schema = QuerySet.model_json_schema()
        prompt = """Your task is to generate 10 relevant queries that can be answered by a document. The generated queries need to be practical (a user might ask) and relevant to the document. The 10 queries should cover all the important topics the document contains. For coverage, think about key topics, concepts, and phrases in the document, extract them and cover them in the query set. For relevance, consider term-based overlap, semantic overlap, and contextual dependency of a query to the document. 
        Now, generate exactly 10 relevant queries for the following document:
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"{prompt}\n\n{document}",
            }],
            extra_body={"guided_json": json_schema},
        )
        queries = []
        reasoning_content = ""
        try:
            queries = json.loads(completion.choices[0].message.content)["queries"]
            reasoning_content = completion.choices[0].message.reasoning_content
        except:
            if self.logger:
                self.logger.error(f"Error in generating queries: {completion.choices[0].message.content}")
        return queries, reasoning_content

def initialize_query_generator(args):
    """
    Initialize the appropriate query generator based on the engine type.

    Args:
        args: Parsed command-line arguments.

    Returns:
        An instance of Doc2Query or Doc2QueryLLM.
    """
    if args.engine == 'doc2query':
        # Initialize Doc2Query
        return Doc2Query(
            model=args.model,
            num_examples=args.num_examples,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            top_k=args.top_k,
        )
    elif args.engine == 'llm':
        return Doc2QueryLLM(
            model=args.model,
            num_examples=args.num_examples,
            use_few_shot=args.use_few_shot,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            few_shot_examples=args.few_shot_examples,
            top_k=args.top_k,
        )
    elif args.engine == 'reasoning_llm':
        return Doc2QueryReasoningLLM(base_url=args.base_url)
if __name__ == "__main__":
    args = parse_arguments()

    save_file = args.save_file
    engine = args.engine
    # Initialize the query generator
    doc2query = initialize_query_generator(args)

    # load data
    data = pt.get_dataset(args.dataset)
    corpus_iter = data.get_corpus_iter()

    # prepare jsonl file for scoring (3 columns, id, text, predicted_queries)

    generated = []

    all_docs = []

    for doc in corpus_iter:
        all_docs.append({
            "docno": doc['docno'],
            "title": doc['title'],
            "text": doc['text']
        })
    
    if args.test:
        all_docs = all_docs[:10]

    all_docs_text = [doc['text'] for doc in all_docs]

    if engine == 'doc2query':
        #=== generate queries per document, use for Doc2Query ===#
        for doc in corpus_iter:
            docno = doc['docno']
            title = doc['title']
            text = doc['text']

            queries = doc2query.generate_queries(text)
            
            generated.append({
                "id": docno,
                "text": text,
                "predicted_queries": queries
            })
    elif engine == 'llm':
        # make sure connection is successful

        #=== generate queries with many documents at a time to accelerate the process, use for vLLM===#
        queries = doc2query.generate_queries(all_docs_text)
        for i, doc in enumerate(tqdm(all_docs, desc="Generating queries")):
            docno = doc['docno']
            title = doc['title']
            text = doc['text']

            generated.append({
                "id": docno,
                "text": text,
                "predicted_queries": [output.text.split(":")[-1].strip() for output in queries[i].outputs] # because many responses start with: Here is a relevant claim: ..., so I use split to get the claim
            })
    elif engine == 'reasoning_llm':
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
           results = list(tqdm(executor.map(doc2query.generate_queries, all_docs_text), total=len(all_docs_text), desc="Generating queries"))
        for i, doc in enumerate(all_docs):
            docno = doc['docno']
            title = doc['title']
            text = doc['text']

            queries, reasoning_content = results[i]
            generated.append({
                "id": docno,
                "text": text,
                "predicted_queries": queries,
                "reasoning_content": reasoning_content
            })

    # save to jsonl file
    with open(save_file, "w") as f:
        for item in generated:
            f.write(json.dumps(item) + "\n")

    print(f"Generated queries saved to {save_file}")