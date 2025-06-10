import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from typing import List
import re
import json
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate queries from documents using Doc2Query or Doc2QueryLLM')
    
    # General arguments
    parser.add_argument('--num_examples', type=int, default=20,
                        help='Number of queries to generate per document')
    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum input length in tokens')
    parser.add_argument('--max_output_length', type=int, default=64,
                        help='Maximum output length in tokens')
    parser.add_argument('--save_file', type=str, required=True,
                        help='Path to save generated queries')
    parser.add_argument('--corpus_path', type=str, required=True, help='Path to the corpus file (jsonl). Use cut version for small dataset testing')
    
    # Doc2Query specific arguments
    parser.add_argument('--model', type=str, default='macavaney/doc2query-t5-base-msmarco',
                        help='Model for Query Generation')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top k parameter for Doc2Query generation')
    parser.add_argument('--test', action='store_true',
                        help='Whether to run in test mode')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing documents')

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

    def generate_queries(self, documents: List[str]) -> List[List[str]]:
        """
        Generates pseudo queries for a given document.
        
        Args:
            document: The input document text.
            
        Returns:
            List of generated queries as strings.
        """
        # Clean document by removing URLs
        docs = [re.sub(self.pattern, "", d) for d in documents]
        
        # Tokenize input
        inputs = self.tokenizer(
            docs,
            max_length=self.max_input_length,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            # Generate queries
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                do_sample=True,
                top_k=self.top_k,
                num_return_sequences=self.num_examples
            )
            
            all_queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            batch_size = len(docs)
            grouped = [
                all_queries[i * self.num_examples : (i + 1) * self.num_examples]
                for i in range(batch_size)
            ]
        
        return grouped


def initialize_query_generator(args):
    return Doc2Query(
        model=args.model,
        num_examples=args.num_examples,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        top_k=args.top_k,
    )
if __name__ == "__main__":
    args = parse_arguments()

    save_file = args.save_file
    # Initialize the query generator
    doc2query = initialize_query_generator(args)


    # load data from jsonl file
    with open(args.corpus_path, 'r') as f:
        all_docs = [json.loads(line) for line in f]
    if args.test:
        all_docs = all_docs[:10]

    all_docs_text = [doc['text'] for doc in all_docs]


    batch_size = args.batch_size
    generated = []
    for i in tqdm(range(0, len(all_docs_text), batch_size), desc="Generating queries"):
        batch_docs = all_docs_text[i:i + batch_size]
        batch_query_lists = doc2query.generate_queries(batch_docs)

        for doc, queries in zip(all_docs[i:i + batch_size], batch_query_lists):
            generated.append({
                "id": doc["_id"],
                "text": doc["text"],
                "predicted_queries": queries,
            })
    # save to jsonl file
    with open(save_file, "w") as f:
        for item in generated:
            f.write(json.dumps(item) + "\n")

    print(f"Generated queries saved to {save_file}")