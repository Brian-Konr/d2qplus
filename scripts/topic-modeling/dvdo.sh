export CUDA_VISIBLE_DEVICES=7

python3 /home/guest/r12922050/GitHub/d2qplus/src/topic-modeling/dvdo.py \
    --corpus /home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl \
    --out_dir /home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/topics/dvdo-0605 \
    --embed_model dmis-lab/biobert-v1.1 \
    --device cuda