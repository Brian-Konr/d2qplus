from pyterrier_pisa import PisaIndex
import pyterrier as pt
import json
import os



def is_index_built(index_path):
    pisa_index = PisaIndex(index_path, stops='none')
    is_built  = pisa_index.built()
    return is_built

def read_txt(file_path: str):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def read_jsonl(file_path: str):
    jsonl = []
    with open(file_path, 'r') as f:
        for line in f:
            jsonl.append(json.loads(line))
    return jsonl

def build_index(corpus_jsonl_path, index_path):
    corpus = read_jsonl(corpus_jsonl_path) # dict_keys(['_id', 'title', 'text', 'metadata'])
    if not corpus:
        print("No documents to index.")
        return None

    if not os.path.exists(index_path):
        os.makedirs(index_path)
    
    for doc in corpus:
        doc['docno'] = doc['_id']

    pisa_index = PisaIndex(index_path, overwrite=True, text_field=['text'])
    # pisaindex takes a list of {docno, text}
    index_ref = pisa_index.index(corpus)
    return pisa_index