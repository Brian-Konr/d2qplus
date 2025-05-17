import pyterrier as pt
import pandas as pd
from pyterrier_pisa import PisaIndex
from pyterrier.measures import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
from tqdm import tqdm
import os

# Initialize PyTerrier
if not pt.started():
    pt.init(version='5.7', helper_version='0.0.7')

# Your build_index function
def build_index(df, index_path):
    pisa_index = PisaIndex(index_path, stops='none', overwrite=True)
    index_ref = pisa_index.index(df.to_dict(orient="records"))
    return pisa_index

# Load the nfcorpus dataset
dataset = pt.get_dataset("irds:beir/scifact")
topics = dataset.get_topics()  # Queries
qrels = pd.read_csv("/home/guest/r12922050/GitHub/d2qplus/data/scifact-qrels/test.trec", names=['qid', 'Q0', 'docno', 'label'], sep='\s+', dtype={"qid": "str", "docno": "str"})


# Create a DataFrame from the corpus for indexing and mapping
corpus_iter = dataset.get_corpus_iter()
# corpus_df = pd.DataFrame(corpus_iter)
corpus_df = pd.read_json("/home/guest/r12922050/GitHub/d2qplus/gen/scifact_llm_gen20.jsonl", lines=True)
corpus_df = corpus_df.rename(columns={"id": "docno"})

# Build the index using your function
index_path = "./scifact_index"
pisa_index = build_index(corpus_df[["docno", "text"]], index_path)

# Create a mapping from docno to text
docno_to_text = dict(zip(corpus_df["docno"], corpus_df["text"]))
docno_to_queries = dict(zip(corpus_df["docno"], corpus_df["predicted_queries"]))

# Create BM25 retriever with your custom b and k1 parameters
b = 0.4  # Example value
k1 = 0.9  # Example value
bm25 = pisa_index.bm25(b=b, k1=k1, num_results=100)
# bm25 = pt.pisa.PisaRetrieve(pisa_index, wmodel=bm25_scorer, num_results=1000)

# Cross-encoder reranker class
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_seq_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.max_seq_length = max_seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.docno_to_text = None

    def set_mapping(self, docno_to_text, docno_to_queries):
        self.docno_to_text = docno_to_text
        self.docno_to_queries = docno_to_queries

    def transform(self, df):
        if self.docno_to_text is None:
            raise ValueError("docno_to_text mapping not set")
        if self.docno_to_queries is None:
            raise ValueError("docno_to_queries mapping not set")
        
        # Group by qid to process each query
        grouped = df.groupby("qid")
        results = []
        
        for qid, group in grouped:
            query = group["query"].iloc[0]
            scores = []
            # Progress bar for documents in this query
            for _, row in group.iterrows():
                docno = row["docno"]
                doc_text = self.docno_to_text.get(docno, "")
                query_text = " ".join(self.docno_to_queries.get(docno, []))
                rerank_text = f"{doc_text}\n\n{query_text}"
                if not doc_text:
                    scores.append(0.0)
                    continue
                inputs = self.tokenizer(
                    query,
                    rerank_text,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    score = torch.sigmoid(logits).item()
                scores.append(score)
            
            group = group.copy()
            # Update group with new scores
            group["score"] = scores

            # Sort by score in descending order and assign ranks
            group = group.sort_values(by="score", ascending=False).reset_index(drop=True)
            group["rank"] = group.index  # Rank starts from 0
            
            results.append(group[["qid", "docno", "score", "rank"]])
            
        # Combine results
        result_df = pd.concat(results, ignore_index=True)
        
        return result_df
# Initialize cross-encoder reranker and set the mapping
cross_encoder = CrossEncoderReranker()
cross_encoder.set_mapping(docno_to_text, docno_to_queries)

# Define the pipeline: BM25 retrieval followed by cross-encoder reranking
# Define the pipeline
pipeline = bm25 >> pt.apply.generic(cross_encoder.transform)

# Custom transformer to track query progress
class ProgressTracker(pt.Transformer):
    def __init__(self, transformer):
        self.transformer = transformer
    
    def transform(self, queries):
        # Initialize tqdm for query progress
        results = []
        total_start_time = time.time()
        
        print(f"Starting experiment with {len(queries)} queries...")
        for _, query in tqdm(queries.iterrows(), total=len(queries), desc="Processing queries", unit="query"):
            # Process one query at a time
            query_df = query.to_frame().T
            result = self.transformer.transform(query_df)
            results.append(result)
        
        # Combine results
        result_df = pd.concat(results, ignore_index=True)
        
        # Log total time for experiment
        total_time = time.time() - total_start_time
        print(f"Experiment completed for {len(queries)} queries in {total_time:.2f} seconds "
              f"({len(queries)/total_time:.2f} queries/second)")
        
        return result_df

# Wrap the pipeline with progress tracker
progress_pipeline = ProgressTracker(pipeline)

# Run the experiment
df_curr = pt.Experiment(
    [bm25, progress_pipeline],
    topics,
    qrels,
    eval_metrics=[RR@10, nDCG@10, R@10, R@100, P@10, P@100],
    names=["BM25", "BM25 + Reranker"],
    verbose=True,
    save_dir="./experiment_results",
    save_mode="overwrite"
)

print(df_curr)