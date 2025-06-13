import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter T5 generated queries based on scores")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file with scored queries")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output filtered JSONL file")
    parser.add_argument("--percentage", type=float, default=30, help="Percentage of top queries to keep (default: 30)")
    
    args = parser.parse_args()
    
    with open(args.input_file, "r") as f:
        scored_t5 = [json.loads(line) for line in f]
    print(f"Length of scored_t5: {len(scored_t5)}. Keys: {scored_t5[0].keys()}")
    print(scored_t5[0]['querygen_score'])

    # sort based on querygen_score and keep top k% queries
    k = args.percentage / 100.0
    for doc in scored_t5:
        query_score = zip(doc['predicted_queries'], doc['querygen_score'])
        sorted_queries = sorted(query_score, key=lambda x: x[1], reverse=True)
        top_k_queries = sorted_queries[:int(len(sorted_queries) * k)]
        doc['predicted_queries'] = [q[0] for q in top_k_queries]
    
    print(f"Len of predicted queries after filtering: {len(scored_t5[0]['predicted_queries'])}")
    
    # Save filtered results
    with open(args.output_file, "w") as f:
        for doc in scored_t5:
            f.write(json.dumps(doc) + "\n")
    
    print(f"Filtered results saved to: {args.output_file}")

if __name__ == "__main__":
    main()