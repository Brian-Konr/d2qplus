import json
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predicted queries with topic keywords')
    parser.add_argument('--llm-extracted-keywords-path', type=str, required=True,
                        help='Path to LLM extracted keywords txt file')
    parser.add_argument('--corpus', type=str, required=True,
                        help='Path to input corpus JSONL file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSONL file')
    return parser.parse_args()

def main():
    args = parse_args()

    
    with open(args.corpus, "r") as f:
        corpus = [json.loads(line) for line in f]

    with open(args.llm_extracted_keywords_path, "r") as f:
        llm_keywords = [line.strip() for line in f.readlines()]
        # each line looks like "doc_id: keyword1, keyword2, ...", so need to parse it
    llm_keywords_dict = {}
    for line in llm_keywords:
        doc_id, keywords = line.split(":", 1)  # only first colon is used to split
        llm_keywords_dict[doc_id.strip()] = keywords
    for doc in corpus:
        doc_id = doc.pop('_id', None)
        doc['id'] = doc_id
        # get keywords for this document from llm_keywords_dict
        if doc_id in llm_keywords_dict:
            keywords = llm_keywords_dict[doc_id]
        else:
            keywords = []
        doc['predicted_queries'] = keywords

    # save updated corpus_topic with predicted queries
    # create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()