import pandas as pd
import json
import random
import numpy as np
import os
from collections import defaultdict
from typing import Set, List, Dict, Tuple
import pyterrier as pt
from pyterrier_pisa import PisaIndex
from tqdm import tqdm

def create_small_corpus_with_hard_negatives(
    corpus_path: str,
    qrels_path: str,
    queries_path: str = None,
    target_corpus_size: int = 10000,
    num_queries: int = 100,
    hard_negative_ratio: float = 0.3,
    random_seed: int = 42
) -> Dict:
    """
    創建固定大小的小corpus，使用hard negatives增加檢索難度
    
    Args:
        corpus_path: corpus.jsonl文件路徑
        qrels_path: qrels文件路徑 (tsv格式)
        queries_path: queries.jsonl文件路徑（可選）
        target_corpus_size: 目標corpus大小
        num_queries: 選取的query數量
        hard_negative_ratio: hard negative documents的比例
        random_seed: 隨機種子
    
    Returns:
        dict: 包含corpus_small, selected_queries, qrels_small等的字典
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"Loading corpus from {corpus_path}...")
    # 載入corpus
    corpus_docs = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc.get('_id') or doc.get('id')
            corpus_docs[doc_id] = doc
    
    print(f"Loaded {len(corpus_docs)} documents")
    
    # 載入qrels
    print(f"Loading qrels from {qrels_path}...")
    qrels_df = pd.read_csv(qrels_path, sep='\t')
    if 'qid' not in qrels_df.columns:
        qrels_df.columns = ['qid', 'docno', 'label'] if len(qrels_df.columns) == 3 else ['qid', 'Q0', 'docno', 'label']
    
    # 只保留正相關的qrels
    positive_qrels = qrels_df[qrels_df['label'] > 0]
    
    # 構建query -> relevant_docs mapping
    query_to_docs = defaultdict(set)
    for _, row in positive_qrels.iterrows():
        query_to_docs[str(row['qid'])].add(str(row['docno']))
    
    # 選擇有足夠相關文檔的queries
    valid_queries = [qid for qid, docs in query_to_docs.items() if len(docs) >= 2]
    
    if len(valid_queries) < num_queries:
        print(f"Warning: Only {len(valid_queries)} valid queries found, using all of them")
        num_queries = len(valid_queries)
    
    # 隨機選擇queries
    selected_query_ids = random.sample(valid_queries, num_queries)
    print(f"Selected {len(selected_query_ids)} queries")
    
    # 收集所有相關文檔
    relevant_docs = set()
    for qid in selected_query_ids:
        relevant_docs.update(query_to_docs[qid])
    
    relevant_docs = {doc_id for doc_id in relevant_docs if doc_id in corpus_docs}
    print(f"Found {len(relevant_docs)} relevant documents")
    
    # 計算需要的hard negative和random文檔數量
    remaining_slots = target_corpus_size - len(relevant_docs)
    if remaining_slots <= 0:
        print(f"Relevant documents ({len(relevant_docs)}) already exceed target size ({target_corpus_size})")
        corpus_small = {doc_id: corpus_docs[doc_id] for doc_id in list(relevant_docs)[:target_corpus_size]}
        hard_negatives = set()
        random_docs = set()
    else:
        # 添加hard negatives (BM25高分但不相關的文檔)
        hard_negative_count = int(remaining_slots * hard_negative_ratio)
        random_count = remaining_slots - hard_negative_count
        
        print(f"Adding {hard_negative_count} hard negatives and {random_count} random documents")
        
        # 獲取hard negatives (這裡用簡化版本，可以改用真正的BM25檢索)
        hard_negatives = get_hard_negatives(
            corpus_docs, 
            relevant_docs, 
            selected_query_ids, 
            queries_path,
            hard_negative_count
        )
        
        # 獲取random documents
        available_docs = set(corpus_docs.keys()) - relevant_docs - hard_negatives
        random_docs = set(random.sample(list(available_docs), min(random_count, len(available_docs))))
        
        # 組合最終的corpus
        final_doc_ids = relevant_docs | hard_negatives | random_docs
        corpus_small = {doc_id: corpus_docs[doc_id] for doc_id in final_doc_ids}
    
    # 創建filtered qrels
    qrels_small = qrels_df[
        (qrels_df['qid'].astype(str).isin(selected_query_ids)) & 
        (qrels_df['docno'].astype(str).isin(corpus_small.keys()))
    ]
    
    # 載入selected queries (如果有queries文件)
    selected_queries = None
    if queries_path and os.path.exists(queries_path):
        with open(queries_path, 'r', encoding='utf-8') as f:
            all_queries = [json.loads(line) for line in f]
        selected_queries = [q for q in all_queries if str(q.get('_id', q.get('qid'))) in selected_query_ids]
    
    result = {
        'corpus_small': corpus_small,
        'qrels_small': qrels_small,
        'selected_queries': selected_queries,
        'selected_query_ids': selected_query_ids,
        'stats': {
            'total_docs': len(corpus_small),
            'relevant_docs': len(relevant_docs),
            'hard_negatives': len(hard_negatives),
            'random_docs': len(random_docs),
            'num_queries': len(selected_query_ids),
            'num_qrels': len(qrels_small)
        }
    }
    
    print("\n=== Corpus Splitting Statistics ===")
    for key, value in result['stats'].items():
        print(f"{key}: {value}")
    
    return result

def get_hard_negatives(
    corpus_docs: Dict, 
    relevant_docs: Set[str], 
    query_ids: List[str],
    queries_path: str,
    count: int
) -> Set[str]:
    """
    獲取hard negative documents (簡化版本)
    在實際應用中，這裡應該用BM25檢索來找高分但不相關的文檔
    """
    # 這裡用簡化版本：隨機選擇非相關文檔作為hard negatives
    # 在實際應用中，應該:
    # 1. 建立BM25 index
    # 2. 用queries檢索top-K文檔
    # 3. 從中選擇不在relevant_docs中的文檔作為hard negatives
    
    available_docs = set(corpus_docs.keys()) - relevant_docs
    if count >= len(available_docs):
        return available_docs
    
    return set(random.sample(list(available_docs), count))

def get_hard_negatives_with_bm25(
    corpus_docs: Dict, 
    relevant_docs: Set[str], 
    query_ids: List[str],
    queries_path: str,
    count: int,
    temp_index_dir: str = "./temp_bm25_index"
) -> Set[str]:
    """
    使用BM25檢索獲取真正的hard negative documents
    
    Args:
        corpus_docs: 所有文檔的字典
        relevant_docs: 已知相關文檔的集合
        query_ids: 選中的query IDs
        queries_path: queries文件路徑
        count: 需要的hard negative數量
        temp_index_dir: 臨時index目錄
    
    Returns:
        Set[str]: hard negative文檔ID集合
    """
    try:
        # 載入queries
        if not queries_path or not os.path.exists(queries_path):
            print("No queries file available, using random selection for hard negatives")
            return get_hard_negatives(corpus_docs, relevant_docs, query_ids, queries_path, count)
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            all_queries = [json.loads(line) for line in f]
        
        query_dict = {str(q.get('_id', q.get('qid'))): q.get('text', q.get('query', '')) for q in all_queries}
        selected_queries = [(qid, query_dict.get(qid, '')) for qid in query_ids if qid in query_dict]
        
        if not selected_queries:
            print("No matching queries found, using random selection for hard negatives")
            return get_hard_negatives(corpus_docs, relevant_docs, query_ids, queries_path, count)
        
        print(f"Building temporary BM25 index for hard negative mining...")
        
        # 準備文檔數據供索引
        docs_for_index = []
        for doc_id, doc in corpus_docs.items():
            if doc_id not in relevant_docs:  # 只索引非相關文檔
                text = doc.get('text', '') + ' ' + doc.get('title', '')
                docs_for_index.append({
                    'docno': str(doc_id),
                    'text': text.strip()
                })
        
        if len(docs_for_index) < count:
            print(f"Not enough non-relevant documents ({len(docs_for_index)}) for hard negatives")
            return set([doc['docno'] for doc in docs_for_index])
        
        # 使用PISA建立臨時索引
        os.makedirs(temp_index_dir, exist_ok=True)
        pisa_index = PisaIndex(temp_index_dir, overwrite=True)
        pisa_index.index(docs_for_index)
        
        # 用BM25檢索器
        bm25 = pisa_index.bm25(b=0.4, k1=0.9, num_results=min(1000, len(docs_for_index)))
        
        # 收集hard negatives
        hard_negatives = set()
        candidates_per_query = max(1, count // len(selected_queries))
        
        for qid, query_text in selected_queries:
            if not query_text.strip():
                continue
                
            # 檢索
            query_df = pd.DataFrame([{'qid': qid, 'query': query_text}])
            try:
                results = bm25.transform(query_df)
                
                # 取top-K作為hard negatives (這些是BM25認為相關但實際不相關的)
                top_docs = results.head(candidates_per_query)['docno'].tolist()
                hard_negatives.update(top_docs)
                
                if len(hard_negatives) >= count:
                    break
                    
            except Exception as e:
                print(f"Error retrieving for query {qid}: {e}")
                continue
        
        # 如果還不夠，補充隨機文檔
        if len(hard_negatives) < count:
            remaining = count - len(hard_negatives)
            available_docs = set([doc['docno'] for doc in docs_for_index]) - hard_negatives
            additional = set(random.sample(list(available_docs), min(remaining, len(available_docs))))
            hard_negatives.update(additional)
        
        # 清理臨時索引
        try:
            import shutil
            shutil.rmtree(temp_index_dir)
        except:
            pass
        
        print(f"Found {len(hard_negatives)} hard negatives using BM25")
        return hard_negatives
        
    except Exception as e:
        print(f"Error in BM25 hard negative mining: {e}")
        print("Falling back to random selection")
        return get_hard_negatives(corpus_docs, relevant_docs, query_ids, queries_path, count)

def create_small_corpus_random(
    corpus_path: str,
    qrels_path: str,
    queries_path: str = None,
    target_corpus_size: int = 10000,
    num_queries: int = 100,
    random_seed: int = 42
) -> Dict:
    """
    創建固定大小的小corpus，純隨機選擇documents
    
    Args:
        corpus_path: corpus.jsonl文件路徑
        qrels_path: qrels文件路徑
        queries_path: queries.jsonl文件路徑（可選）
        target_corpus_size: 目標corpus大小
        num_queries: 選取的query數量
        random_seed: 隨機種子
    
    Returns:
        dict: 包含corpus_small, selected_queries, qrels_small等的字典
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"Loading corpus from {corpus_path}...")
    # 載入corpus
    corpus_docs = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc.get('_id') or doc.get('id')
            corpus_docs[doc_id] = doc
    
    print(f"Loaded {len(corpus_docs)} documents")
    
    # 載入qrels
    print(f"Loading qrels from {qrels_path}...")
    qrels_df = pd.read_csv(qrels_path, sep='\t')
    if 'qid' not in qrels_df.columns:
        qrels_df.columns = ['qid', 'docno', 'label'] if len(qrels_df.columns) == 3 else ['qid', 'Q0', 'docno', 'label']
    
    # 只保留正相關的qrels
    positive_qrels = qrels_df[qrels_df['label'] > 0]
    
    # 構建query -> relevant_docs mapping
    query_to_docs = defaultdict(set)
    for _, row in positive_qrels.iterrows():
        query_to_docs[str(row['qid'])].add(str(row['docno']))
    
    # 選擇有相關文檔的queries
    valid_queries = [qid for qid, docs in query_to_docs.items() if len(docs) >= 1]
    
    if len(valid_queries) < num_queries:
        print(f"Warning: Only {len(valid_queries)} valid queries found, using all of them")
        num_queries = len(valid_queries)
    
    # 隨機選擇queries
    selected_query_ids = random.sample(valid_queries, num_queries)
    print(f"Selected {len(selected_query_ids)} queries")
    
    # 收集所有相關文檔
    relevant_docs = set()
    for qid in selected_query_ids:
        relevant_docs.update(query_to_docs[qid])
    
    relevant_docs = {doc_id for doc_id in relevant_docs if doc_id in corpus_docs}
    print(f"Found {len(relevant_docs)} relevant documents")
    
    # 隨機選擇documents直到達到目標大小
    if len(relevant_docs) >= target_corpus_size:
        # 如果相關文檔就已經足夠，隨機選擇其中一部分
        selected_doc_ids = set(random.sample(list(relevant_docs), target_corpus_size))
    else:
        # 先包含所有相關文檔，然後隨機添加其他文檔
        remaining_slots = target_corpus_size - len(relevant_docs)
        available_docs = set(corpus_docs.keys()) - relevant_docs
        additional_docs = set(random.sample(list(available_docs), min(remaining_slots, len(available_docs))))
        selected_doc_ids = relevant_docs | additional_docs
    
    corpus_small = {doc_id: corpus_docs[doc_id] for doc_id in selected_doc_ids}
    
    # 創建filtered qrels
    qrels_small = qrels_df[
        (qrels_df['qid'].astype(str).isin(selected_query_ids)) & 
        (qrels_df['docno'].astype(str).isin(corpus_small.keys()))
    ]
    
    # 載入selected queries (如果有queries文件)
    selected_queries = None
    if queries_path and os.path.exists(queries_path):
        with open(queries_path, 'r', encoding='utf-8') as f:
            all_queries = [json.loads(line) for line in f]
        selected_queries = [q for q in all_queries if str(q.get('_id', q.get('qid'))) in selected_query_ids]
    
    result = {
        'corpus_small': corpus_small,
        'qrels_small': qrels_small,
        'selected_queries': selected_queries,
        'selected_query_ids': selected_query_ids,
        'stats': {
            'total_docs': len(corpus_small),
            'relevant_docs': len([doc_id for doc_id in selected_doc_ids if doc_id in relevant_docs]),
            'random_docs': len(selected_doc_ids) - len([doc_id for doc_id in selected_doc_ids if doc_id in relevant_docs]),
            'num_queries': len(selected_query_ids),
            'num_qrels': len(qrels_small)
        }
    }
    
    print("\n=== Corpus Splitting Statistics ===")
    for key, value in result['stats'].items():
        print(f"{key}: {value}")
    
    return result

def create_small_corpus_with_bm25_hard_negatives(
    corpus_path: str,
    qrels_path: str,
    queries_path: str = None,
    target_corpus_size: int = 10000,
    num_queries: int = 100,
    hard_negative_ratio: float = 0.3,
    random_seed: int = 42
) -> Dict:
    """
    創建固定大小的小corpus，使用真正的BM25檢索來獲取hard negatives
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"Loading corpus from {corpus_path}...")
    # 載入corpus
    corpus_docs = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc.get('_id') or doc.get('id')
            corpus_docs[doc_id] = doc
    
    print(f"Loaded {len(corpus_docs)} documents")
    
    # 載入qrels
    print(f"Loading qrels from {qrels_path}...")
    qrels_df = pd.read_csv(qrels_path, sep='\t')
    if 'qid' not in qrels_df.columns:
        qrels_df.columns = ['qid', 'docno', 'label'] if len(qrels_df.columns) == 3 else ['qid', 'Q0', 'docno', 'label']
    
    # 只保留正相關的qrels
    positive_qrels = qrels_df[qrels_df['label'] > 0]
    
    # 構建query -> relevant_docs mapping
    query_to_docs = defaultdict(set)
    for _, row in positive_qrels.iterrows():
        query_to_docs[str(row['qid'])].add(str(row['docno']))
    
    # 選擇有足夠相關文檔的queries
    valid_queries = [qid for qid, docs in query_to_docs.items() if len(docs) >= 2]
    
    if len(valid_queries) < num_queries:
        print(f"Warning: Only {len(valid_queries)} valid queries found, using all of them")
        num_queries = len(valid_queries)
    
    # 隨機選擇queries
    selected_query_ids = random.sample(valid_queries, num_queries)
    print(f"Selected {len(selected_query_ids)} queries")
    
    # 收集所有相關文檔
    relevant_docs = set()
    for qid in selected_query_ids:
        relevant_docs.update(query_to_docs[qid])
    
    relevant_docs = {doc_id for doc_id in relevant_docs if doc_id in corpus_docs}
    print(f"Found {len(relevant_docs)} relevant documents")
    
    # 計算需要的hard negative和random文檔數量
    remaining_slots = target_corpus_size - len(relevant_docs)
    if remaining_slots <= 0:
        print(f"Relevant documents ({len(relevant_docs)}) already exceed target size ({target_corpus_size})")
        corpus_small = {doc_id: corpus_docs[doc_id] for doc_id in list(relevant_docs)[:target_corpus_size]}
        hard_negatives = set()
        random_docs = set()
    else:
        # 添加hard negatives (使用真正的BM25檢索)
        hard_negative_count = int(remaining_slots * hard_negative_ratio)
        random_count = remaining_slots - hard_negative_count
        
        print(f"Adding {hard_negative_count} BM25 hard negatives and {random_count} random documents")
        
        # 獲取hard negatives (使用BM25)
        hard_negatives = get_hard_negatives_with_bm25(
            corpus_docs, 
            relevant_docs, 
            selected_query_ids, 
            queries_path,
            hard_negative_count
        )
        
        # 獲取random documents
        available_docs = set(corpus_docs.keys()) - relevant_docs - hard_negatives
        random_docs = set(random.sample(list(available_docs), min(random_count, len(available_docs))))
        
        # 組合最終的corpus
        final_doc_ids = relevant_docs | hard_negatives | random_docs
        corpus_small = {doc_id: corpus_docs[doc_id] for doc_id in final_doc_ids}
    
    # 創建filtered qrels
    qrels_small = qrels_df[
        (qrels_df['qid'].astype(str).isin(selected_query_ids)) & 
        (qrels_df['docno'].astype(str).isin(corpus_small.keys()))
    ]
    
    # 載入selected queries (如果有queries文件)
    selected_queries = None
    if queries_path and os.path.exists(queries_path):
        with open(queries_path, 'r', encoding='utf-8') as f:
            all_queries = [json.loads(line) for line in f]
        selected_queries = [q for q in all_queries if str(q.get('_id', q.get('qid'))) in selected_query_ids]
    
    result = {
        'corpus_small': corpus_small,
        'qrels_small': qrels_small,
        'selected_queries': selected_queries,
        'selected_query_ids': selected_query_ids,
        'stats': {
            'total_docs': len(corpus_small),
            'relevant_docs': len(relevant_docs),
            'hard_negatives': len(hard_negatives),
            'random_docs': len(random_docs),
            'num_queries': len(selected_query_ids),
            'num_qrels': len(qrels_small)
        }
    }
    
    print("\n=== Corpus Splitting Statistics (with BM25 Hard Negatives) ===")
    for key, value in result['stats'].items():
        print(f"{key}: {value}")
    
    return result

def save_small_corpus(result: Dict, output_dir: str, dataset_name: str):
    """
    保存切分後的小corpus到文件
    
    Args:
        result: create_small_corpus_*函數的返回結果
        output_dir: 輸出目錄
        dataset_name: 數據集名稱
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存corpus
    corpus_path = os.path.join(output_dir, f"{dataset_name}_corpus_small.jsonl")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for doc_id, doc in result['corpus_small'].items():
            # 確保doc有正確的id字段
            if '_id' not in doc and 'id' not in doc:
                doc['_id'] = doc_id
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # 保存qrels
    qrels_path = os.path.join(output_dir, f"{dataset_name}_qrels_small.tsv")
    result['qrels_small'].to_csv(qrels_path, sep='\t', index=False)
    
    # 保存queries (如果有)
    if result['selected_queries']:
        queries_path = os.path.join(output_dir, f"{dataset_name}_queries_small.jsonl")
        with open(queries_path, 'w', encoding='utf-8') as f:
            for query in result['selected_queries']:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    # 保存統計信息
    stats_path = os.path.join(output_dir, f"{dataset_name}_split_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(result['stats'], f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved small corpus to {output_dir}:")
    print(f"- Corpus: {corpus_path}")
    print(f"- Qrels: {qrels_path}")
    if result['selected_queries']:
        print(f"- Queries: {queries_path}")
    print(f"- Stats: {stats_path}")

print("Dataset splitting functions loaded successfully!")
