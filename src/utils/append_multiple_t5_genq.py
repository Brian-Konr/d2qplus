import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

def read_jsonl(filepath: str) -> List[Dict[Any, Any]]:
    """Read JSONL file and return list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def write_jsonl(data: List[Dict[Any, Any]], filepath: str) -> None:
    """Write list of dictionaries to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def merge_generated_queries(input_files: List[str], output_file: str) -> None:
    """
    Merge multiple gen_queries.jsonl files by extending their predicted_queries fields.
    
    Args:
        input_files: List of paths to input gen_queries.jsonl files
        output_file: Path to output unified file
    """
    print(f"📥 Loading {len(input_files)} input files...")
    
    # Dictionary to store merged data by document ID
    merged_data = {}
    
    for i, input_file in enumerate(input_files):
        print(f"  📄 Processing file {i+1}/{len(input_files)}: {input_file}")
        
        if not Path(input_file).exists():
            print(f"    ⚠️  Warning: File not found, skipping: {input_file}")
            continue
            
        data = read_jsonl(input_file)
        print(f"    ✅ Loaded {len(data)} documents")
        
        for doc in data:
            doc_id = doc.get("id")
            if doc_id is None:
                print(f"    ⚠️  Warning: Document without ID found, skipping")
                continue
                
            predicted_queries = doc.get("predicted_queries", [])
            
            if doc_id not in merged_data:
                # First time seeing this document
                merged_data[doc_id] = {
                    "id": doc_id,
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                    "predicted_queries": predicted_queries.copy()
                }
            else:
                # Extend existing predicted_queries
                merged_data[doc_id]["predicted_queries"].extend(predicted_queries)
    
    # Convert to list and sort by ID for consistent output
    unified_data = list(merged_data.values())
    unified_data.sort(key=lambda x: x["id"])
    
    total_docs = len(unified_data)
    total_queries = sum(len(doc["predicted_queries"]) for doc in unified_data)
    
    print(f"\n📊 Merge Statistics:")
    print(f"  - Total documents: {total_docs}")
    print(f"  - Total queries: {total_queries}")
    
    # Write unified file
    print(f"\n💾 Saving unified file to: {output_file}")
    write_jsonl(unified_data, output_file)
    print(f"✅ Successfully created unified file with {total_docs} documents")

def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple gen_queries.jsonl files by extending predicted_queries")
    parser.add_argument("--input_files", type=str, nargs='+', required=True, 
                       help="List of input gen_queries.jsonl file paths")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output unified gen_queries.jsonl file")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Print detailed information during processing")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print(f"🔄 Merging {len(args.input_files)} gen_queries.jsonl files...")
    print(f"📤 Output file: {args.output_file}")
    
    if args.verbose:
        print(f"📋 Input files:")
        for i, file in enumerate(args.input_files, 1):
            print(f"  {i}. {file}")
    
    merge_generated_queries(args.input_files, args.output_file)
    print(f"🎉 Merge completed!")
