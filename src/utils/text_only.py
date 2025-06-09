import json
import argparse

def main(input_path, output_path):
    # Read corpus
    with open(input_path, "r") as f:
        corpus = [json.loads(line) for line in f]

    # Process documents
    for doc in corpus:
        doc['id'] = doc.pop('_id')  # rename '_id' to 'id'
        doc['predicted_queries'] = ""

    # Save corpus to file
    with open(output_path, "w") as f:
        for doc in corpus:
            json.dump(doc, f)
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process corpus and add empty predicted_queries field')
    parser.add_argument('--input', type=str, default="/home/guest/r12922050/GitHub/d2qplus/data/trec-car-10000/corpus.jsonl",
                        help='Input corpus file path')
    parser.add_argument('--output', type=str, default="/home/guest/r12922050/GitHub/d2qplus/gen/trec-car/text-only.jsonl",
                        help='Output file path')
    
    args = parser.parse_args()
    main(args.input, args.output)
