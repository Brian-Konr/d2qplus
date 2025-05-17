import pyterrier as pt
import pandas as pd
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

if not pt.started():
    pt.init(version='5.7', helper_version='0.0.7')
    
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Initialize Pyterrier
if not pt.started():
    pt.init()

# Step 1: Prepare a sample document collection
docs = [
    {"docno": "doc1", "text": "The quick brown fox jumps over the lazy dog."},
    {"docno": "doc2", "text": "A fox fled from danger in the forest."},
    {"docno": "doc3", "text": "Dogs are loyal companions to humans."},
    {"docno": "doc4", "text": "The forest is home to many wild animals."}
]

jsonl_file = "sample_collection.jsonl"
with open(jsonl_file, "w") as f:
    for doc in docs:
        f.write(json.dumps(doc) + "\n")

# Step 2: Index the collection
index_dir = "./sample_index"
indexer = pt.IterDictIndexer(index_dir, meta={"docno": 20, "text": 4096})
index_ref = indexer.index(docs)
index = pt.IndexFactory.of(index_ref)

# Step 3: Define sample queries and qrels
# Queries (topics)
queries = pd.DataFrame([
    {"qid": "q1", "query": "fox in forest"},
    {"qid": "q2", "query": "loyal dogs"}
])

# Qrels (relevance judgments: 1 = relevant, 0 = non-relevant)
qrels = pd.DataFrame([
    {"qid": "q1", "docno": "doc2", "label": 1},  # Relevant for "fox in forest"
    {"qid": "q1", "docno": "doc4", "label": 1},
    {"qid": "q1", "docno": "doc1", "label": 0},
    {"qid": "q1", "docno": "doc3", "label": 0},
    {"qid": "q2", "docno": "doc3", "label": 1},  # Relevant for "loyal dogs"
    {"qid": "q2", "docno": "doc1", "label": 0},
    {"qid": "q2", "docno": "doc2", "label": 0},
    {"qid": "q2", "docno": "doc4", "label": 0}
])

# Step 4: First-stage retrieval with BM25
bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=10, metadata=["docno", "text"])

# Step 5: Define a neural reranker using a cross-encoder
class CrossEncoderReranker(pt.Transformer):
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32, device="cpu"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

    def transform(self, run):
        queries = run["query"].tolist()
        docs = run["text"].tolist()
        docnos = run["docno"].tolist()
        qids = run["qid"].tolist()

        pairs = [[q, d] for q, d in zip(queries, docs)]
        scores = []

        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            inputs = self.tokenizer(batch_pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
            scores.extend(batch_scores)

        result = pd.DataFrame({
            "qid": qids,
            "docno": docnos,
            "score": scores,
            "query": queries,
            "text": docs
        })
        result["rank"] = result.groupby("qid")["score"].rank(ascending=False, method="first").astype(int) - 1
        return result

# Step 6: Create the two-stage pipeline
neural_reranker = CrossEncoderReranker(device="cuda" if torch.cuda.is_available() else "cpu")
pipeline = bm25 >> neural_reranker

# Step 7: Run pt.Experiment to evaluate BM25 and the two-stage pipeline
experiment = pt.Experiment(
    retr_systems=[bm25, pipeline],  # Compare BM25 alone vs. BM25 + reranker
    topics=queries,
    qrels=qrels,
    eval_metrics=["ndcg_cut_10", "map", "P_5"],
    names=["BM25", "BM25 + Neural Reranker"]
)

# Step 8: Print experiment results
print("Experiment Results:")
print(experiment)

# Clean up
os.remove(jsonl_file)