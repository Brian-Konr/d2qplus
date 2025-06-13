# parser.add_argument("--corpus_path", required=True, help="Path to input JSONL (with '_id' and 'text').")
#     parser.add_argument("--base_output_dir", required=True, help="Base directory for output files.")

#     parser.add_argument("--chunk_mode", choices=["sentence", "window"], default="sentence", help="Chunking mode.")
#     parser.add_argument("--win_size", type=int, default=4, help="Window size if chunk_mode='window'.")
#     parser.add_argument("--win_step", type=int, default=2, help="Window step if chunk_mode='window'.")
    
#     parser.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
#     parser.add_argument("--embed_device", default="cpu", help="Device for embeddings (e.g., 'cpu' or 'cuda:0').")

#     # Grid search parameters
#     parser.add_argument("--n_neighbors", nargs="+", type=int, default=[10, 15, 30], help="List of n_neighbors values for UMAP.")
#     parser.add_argument("--n_components", nargs="+", type=int, default=[5, 10, 15], help="List of n_components values for UMAP.")
#     parser.add_argument("--min_topic_size", nargs="+", type=int, default=[5, 10, 15], help="List of min_topic_size values for BERTopic.")

#     # BERTopic parameters
#     parser.add_argument("--top_n_words", type=int, default=10, help="Number of top words per topic to extract.")
#     parser.add_argument("--save_topic_model", action="store_true", default=False, help="Save the BERTopic model to disk.")
export CUDA_VISIBLE_DEVICES=7

CORPUS_PATH=/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl
BASE_OUTPUT_DIR=/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics
EMBED_MODEL=pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
EMBED_DEVICE=cuda

python3 /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/grid_search_topic_modeling.py \
    --corpus_path "$CORPUS_PATH" \
    --base_output_dir "$BASE_OUTPUT_DIR" \
    --embed_model "$EMBED_MODEL" \
    --embed_device "$EMBED_DEVICE"