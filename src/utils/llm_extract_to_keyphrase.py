llm_extract_keywords = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0606-biobert-mnli-reduce-outlier/llm_extracted_keywords.txt"

with open(llm_extract_keywords, "r") as f:
    llm_keywords = [line.strip() for line in f.readlines()]
docid2keywords = {}
for line in llm_keywords:
    doc_id, keywords = line.split(":", 1)  # only first colon is used to split
    doc_id = doc_id.strip()
    keywords = keywords.strip()
    for keyword in keywords.split(","):
        keyword = keyword.strip()
        if keyword:
            if doc_id in docid2keywords:
                docid2keywords[doc_id].append(keyword)
            else:
                docid2keywords[doc_id] = [keyword]
# Save the docid2keywords dictionary to a JSON file
import json
output_path = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/0606-biobert-mnli-reduce-outlier/llm_extracted_keywords.jsonl"
with open(output_path, "w") as f:
    for doc_id, keywords in docid2keywords.items():
        entry = {"doc_id": doc_id, "keywords": keywords}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")