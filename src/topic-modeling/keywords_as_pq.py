import json
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predicted queries with topic keywords')
    parser.add_argument('--doc-topics', type=str, required=True,
                        help='Path to document topics JSONL file')
    parser.add_argument('--topic-info', type=str, required=True,
                        help='Path to topic info DataFrame pickle file')
    parser.add_argument('--corpus', type=str, required=True,
                        help='Path to input corpus JSONL file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSONL file')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top keywords to use per topic (default: 10)')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.doc_topics, "r") as f:
        corpus_topic = [json.loads(line) for line in f]
    topic_info_df = pd.read_pickle(args.topic_info)

    with open(args.corpus, "r") as f:
        corpus = [json.loads(line) for line in f]

    corpus_topic = {doc['doc_id']: doc['topics'] for doc in corpus_topic}

    topic_keywords = {} # {topic_id: [keyword1, keyword2, ...]}
    for tid, topic in topic_info_df.iterrows():
        if tid == -1:
            continue
        keywords = topic['Representation']
        topic_keywords[tid] = keywords

    # for every document, add their topic keywords
    for doc in corpus:
        doc_id = doc.pop('_id', None)
        doc['id'] = doc_id
        keywords = []
        for topic in corpus_topic[doc_id]:
            topic_id = topic['topic_id']
            if topic_id in topic_keywords:
                keywords.extend(topic_keywords[topic_id][:args.top_k])
        doc['predicted_queries'] = keywords

    # save updated corpus_topic with predicted queries
    with open(args.output, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()