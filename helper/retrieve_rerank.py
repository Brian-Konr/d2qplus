import pyterrier as pt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

# Initialize PyTerrier
if not pt.started():
    pt.init()

# Load the nfcorpus dataset
dataset = pt.get_dataset("irds:beir/nfcorpus")
topics = dataset.get_topics("text")  # Queries
qrels = pd.read_csv("/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/qrels/test.trec", names=['qid', 'Q0', 'docno', 'label'], sep='\s+', dtype={"qid": "str", "docno": "str"})


# BM25 retrieval pipeline
bm25 = pt.BatchRetrieve.from_dataset("irds:beir/nfcorpus", "terrier_stemmed", wmodel="BM25", num_results=100)

# Cross-encoder reranker class
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_seq_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.max_seq_length = max_seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def transform(self, df):
        scores = []
        for _, row in df.iterrows():
            query = row["query"]
            doc_text = row["text"]
            # Tokenize query and document
            inputs = self.tokenizer(
                query,
                doc_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Get score from cross-encoder
            with torch.no_grad():
                logits = self.model(**inputs).logits
                score = torch.sigmoid(logits).item()
            scores.append(score)
        df["score"] = scores
        return df[["qid", "docno", "score"]]

# Initialize cross-encoder reranker
cross_encoder = CrossEncoderReranker()

# Define the pipeline: BM25 retrieval followed by cross-encoder reranking
pipeline = bm25 >> pt.text.get_text(dataset, "text") >> pt.apply.generic(cross_encoder.transform)

# Run the experiment
pt.Experiment(
    [pipeline],
    topics,
    qrels,
    eval_metrics=["map", "ndcg_cut_10", "P_10"],
    names=["BM25 + Cross-Encoder"],
    verbose=True
)