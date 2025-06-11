from .util import read_jsonl, read_txt
import pandas as pd
from typing import List, Any
from .constants import USER_PROMPT_TOPIC_WITH_WEIGHT_TEMPLATE, USER_PROMPT_TEMPLATE, USER_PROMPT_TOPIC_WITHOUT_WEIGHT_TEMPLATE

def get_topic_info_dict(enhanced_topic_info_pkl: str) -> dict:
    """
    get topic information dictionary where key is topic ID and value is a dictionary with 'keywords' and 'Enhanced_Topic'.
    """
    topic_dict = {}
    enhanced_df = pd.read_pickle(enhanced_topic_info_pkl)
    for _, row in enhanced_df.iterrows():
        topic_id = row['Topic']
        topic_dict[topic_id] = {
            'keywords': row['Representation'],
            'Enhanced_Topic': row['Enhanced_Topic']
        }
    return topic_dict

def combine_topic_info(enhanced_topic_info_pkl: str, corpus_topics_path: str, corpus_path: str) -> List[dict]:
    """
    Enhance corpus topics with corpus text & title, topic representation (keywords) and topic NL description 及 weights 

    - enhanced_topic_info_pkl: Path to the topic information pickle with enhanced llm representation (e.g., topic_info_dataframe_enhanced.pkl)
    - corpus_topics_path: Path to the corpus topics JSONL file (e.g., augmented-data/CSFCube-1.1/doc_topics.jsonl)
    - corpus_path: Path to the original corpus JSONL file (e.g., augmented-data/CSFCube-1.1/corpus.jsonl)

    Returns:
        List of dictionaries with enhanced corpus topics, each containing:
        - doc_id: Document ID
        - text: Document text
        - title: Document title (if available)
        - topics: List of topics with their IDs, weights, keywords, and enhanced topic descriptions.
    """
    topic_info_dict = get_topic_info_dict(enhanced_topic_info_pkl)
    corpus_topics = read_jsonl(corpus_topics_path) # doc_id, "topics": [{"topic_id": 1, "weight": 0.5}, ...]
    corpus = read_jsonl(corpus_path)
    doc_id2doc = {doc['_id']: doc for doc in corpus}  # Map doc_id to document content (text, title)
    
    enhanced_corpus_topics = []
    for doc in corpus_topics:
        doc_id = doc['doc_id']
        topics = doc.get('topics', [])
        
        enhanced_topics = []
        for topic in topics:
            topic_id = topic['topic_id']
            if topic_id in topic_info_dict:
                enhanced_topic = {
                    'topic_id': topic_id,
                    'weight': topic['weight'],
                    'Representation': topic_info_dict[topic_id]['keywords'],
                    'Enhanced_Topic': topic_info_dict[topic_id]['Enhanced_Topic']
                }
                enhanced_topics.append(enhanced_topic)
        
        enhanced_corpus_topics.append({
            'doc_id': doc_id,
            'text': doc_id2doc[doc_id]['text'],
            'title': doc_id2doc[doc_id]['title'] if 'title' in doc_id2doc[doc_id] else '',
            'topics': enhanced_topics
        })
    
    return enhanced_corpus_topics

def prepare_prompts(
        data, 
        max_keywords=15, 
        max_topics=5, 
        random_pick_keywords=False, 
        proportional_selection=True, 
        with_topic_weights=True,
    ):
    """
    Prepare prompts for documents with keyword and topic guidance.
    
    Args:
        data: List of documents with 'title', 'text', and 'topics' fields
        max_keywords: Maximum number of keywords to include
        max_topics: Maximum number of topics to include
        random_pick_keywords: If True, randomly pick keywords from topics; otherwise pick top keywords
        proportional_selection: If True, select keywords proportionally based on topic weights
        with_topic_weights: If True, include topic weights in prompt; otherwise only include topic names
        with_topic_keywords: If True, include topic keywords in the prompt
    
    Returns:
        List of documents with added 'prompt', 'keywords', 'formatted_topics', and 'formatted_keywords' fields.
    """
    import random
    
    print(f"📝 Preparing prompts for {len(data)} documents...")
    print(f"  - Max keywords: {max_keywords}, Max topics: {max_topics}")
    print(f"  - Random keywords: {random_pick_keywords}, Proportional: {proportional_selection}")
    print(f"  - With topic weights: {with_topic_weights}")
    
    sample_shown = False
    
    for i, d in enumerate(data):
        doc_content = d['title'] + "\n" + d['text']
        topics = d['topics']

        # sort topics based on weight and limit to max_topics
        topics = sorted(topics, key=lambda x: x['weight'], reverse=True)[:max_topics]
        
        if proportional_selection:
            # Calculate total weight for normalization
            total_weight = sum(t['weight'] for t in topics)
            if total_weight == 0:
                # If no topics have weight, distribute equally
                normalized_weights = [1.0/len(topics) for _ in topics]
            else:
                normalized_weights = [t['weight'] / total_weight for t in topics]
            
            # Collect keywords from all topics proportionally
            all_keywords = []
            for topic, norm_weight in zip(topics, normalized_weights):
                topic_keywords = topic.get('Representation', [])
                # Calculate how many keywords to pick from this topic
                num_keywords_from_topic = max(1, int(max_keywords * norm_weight))
                
                if random_pick_keywords:
                    # Randomly sample keywords
                    selected_keywords = random.sample(
                        topic_keywords, 
                        min(num_keywords_from_topic, len(topic_keywords))
                    )
                else:
                    # Take top keywords
                    selected_keywords = topic_keywords[:num_keywords_from_topic]
                
                all_keywords.extend(selected_keywords)
        else:
            # Collect all keywords from all topics first
            all_keywords = []
            for topic in topics:
                topic_keywords = topic.get('Representation', [])
                all_keywords.extend(topic_keywords)
            
            # Remove duplicates
            all_keywords = list(dict.fromkeys(all_keywords))  # preserves order
            
            # Randomly pick up to max_keywords
            if len(all_keywords) > max_keywords:
                all_keywords = random.sample(all_keywords, max_keywords)
        
        # Remove duplicates while preserving order and limit to max_keywords
        seen = set()
        unique_keywords = []
        for kw in all_keywords:
            if kw not in seen and len(unique_keywords) < max_keywords:
                seen.add(kw)
                unique_keywords.append(kw)
        
        keywords_str = ', '.join(unique_keywords)
        
        # format topics for prompt
        if with_topic_weights:
            topics_str = ', '.join([f"{t['Enhanced_Topic']} ({t['weight']})" for t in topics])
        else:
            topics_str = ', '.join([t['Enhanced_Topic'] for t in topics])

        # Select prompt template based on parameters
        if with_topic_weights:
            template = USER_PROMPT_TOPIC_WITH_WEIGHT_TEMPLATE
        else:
            template = USER_PROMPT_TOPIC_WITHOUT_WEIGHT_TEMPLATE
        prompt = template.replace("[DOCUMENT]", doc_content).replace("[KEYWORDS]", keywords_str).replace("[TOPICS]", topics_str)
        
        d['prompt'] = prompt
        d['keywords'] = unique_keywords  # Store selected keywords
        d['formatted_topics'] = topics_str  # Store formatted topics for reuse
        d['formatted_keywords'] = keywords_str  # Store formatted keywords for reuse
        
        # Show sample for first document
        if i == 0 and not sample_shown:
            print(f"📄 Sample document preparation:")
            print(f"  - Selected {len(unique_keywords)} keywords: {keywords_str[:100]}...")
            print(f"  - Selected {len(topics)} topics: {topics_str[:100]}...")
            sample_shown = True
    
    print(f"✅ Prompt preparation completed")
    return data

def prepare_training_data(
        integrated_data_path, 
        drop_no_topics=False, 
        max_keywords=10, 
        max_topics=5,
        with_topic_weights=True,
    ):
    """
    Take integrated data and perform following steps:

    1. Read the integrated data from a JSONL file.
    2. sort keywords based on their score and limit to `max_keywords`.
    3. sort topics based on their weight and limit to `max_topics`.
    4. Prepare a prompt for each document using the user prompt template. (with / without topics)
    5. Return the modified data with prompts.

    The modified data is used for generating final training data's prompt column with topic / keyword guidance, but it can also be used for directly prompting LLM to generate queries (baseline)
    """
    data = read_jsonl(integrated_data_path)
    if drop_no_topics:
        data = [d for d in data if d['topics']]
    
    # prepare prompts using the extracted function
    data = prepare_prompts(
        data, 
        max_keywords=max_keywords, 
        max_topics=max_topics,
        with_topic_weights=with_topic_weights,
    )
    return data

def save_document_vectors(model_name, data_path, out_path):
    """
    Embed documents and save their vectors to a .pt file.

    model_name: name of the embedding model to use (e.g., "all-MiniLM-L6-v2"). Will be loaded using SentenceTransformer.
    data_path: path to the input jsonl data file (need to make sure it has 'text' and '_id' field)
    out_path: path to the output .pt file where vectors will be saved. It will be a dictionary with document IDs (string) as keys and normalized vectors as values.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(model_name, device=device)  # use CPU for embedding

    data = read_jsonl(data_path)
    texts = [d['text'] for d in data]
    ids = [d['_id'] for d in data]
    embs = embed_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    vectors = {str(id_) : emb.cpu() for id_, emb in zip(ids, embs)}
    torch.save(vectors, out_path)
    print(f"Document vectors saved to {out_path} with {len(vectors)} entries.")


if __name__ == "__main__":
    # - Save document vectors -
    
    # embed_model_name = "allenai/scibert_scivocab_uncased"
    # data_path = "/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl"
    # out_path = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/corpus-lookup/document_vectors.pt"
    # save_document_vectors(embed_model_name, data_path, out_path)



    import json
    INTEGRATED_DATA_PATH = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/integrated/data.jsonl"
    DATA_WITH_PROMPT_OUT_PATH = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/integrated/data_with_prompt_3.jsonl"
    data = prepare_training_data(integrated_data_path=INTEGRATED_DATA_PATH)

    with open(DATA_WITH_PROMPT_OUT_PATH, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')









"""
Example of integrated data:
{'title': 'Neurobehavioral function and low-level exposure to brominated flame retardants in adolescents: a cross-sectional study', 'text': 'Background Animal and in vitro studies demonstrated a neurotoxic potential of brominated flame retardants, a group of chemicals used in many household and commercial products to prevent fire. Although the first reports of detrimental neurobehavioral effects in rodents appeared more than ten years ago, human data are sparse. Methods As a part of a biomonitoring program for environmental health surveillance in Flanders, Belgium, we assessed the neurobehavioral function with the Neurobehavioral Evaluation System (NES-3), and collected blood samples in a group of high school students. Cross-sectional data on 515 adolescents (13.6-17 years of age) was available for the analysis. Multiple regression models accounting for potential confounders were used to investigate the associations between biomarkers of internal exposure to brominated flame retardants [serum levels of polybrominated diphenyl ether (PBDE) congeners 47, 99, 100, 153, 209, hexabromocyclododecane (HBCD), and tetrabromobisphenol A (TBBPA)] and cognitive performance. In addition, we investigated the association between brominated flame retardants and serum levels of FT3, FT4, and TSH. Results A two-fold increase of the sum of serum PBDE’s was associated with a decrease of the number of taps with the preferred-hand in the Finger Tapping test by 5.31 (95% CI: 0.56 to 10.05, p\u2009=\u20090.029). The effects of the individual PBDE congeners on the motor speed were consistent. Serum levels above the level of quantification were associated with an average decrease of FT3 level by 0.18 pg/mL (95% CI: 0.03 to 0.34, p\u2009=\u20090.020) for PBDE-99 and by 0.15 pg/mL (95% CI: 0.004 to 0.29, p\u2009=\u20090.045) for PBDE-100, compared with concentrations below the level of quantification. PBDE-47 level above the level of quantification was associated with an average increase of TSH levels by 10.1% (95% CI: 0.8% to 20.2%, p\u2009=\u20090.033), compared with concentrations below the level of quantification. We did not observe effects of PBDE’s on neurobehavioral domains other than the motor function. HBCD and TBBPA did not show consistent associations with performance in the neurobehavioral tests. Conclusions This study is one of few studies and so far the largest one investigating the neurobehavioral effects of brominated flame retardants in humans. Consistently with experimental animal data, PBDE exposure was associated with changes in the motor function and the serum levels of the thyroid hormones.', 'keywords': [['neurobehavioral effects', 0.5726], ['levels polybrominated diphenyl', 0.5509], ['multiple regression models', 0.5153], ['blood samples', 0.5007], ['studies demonstrated', 0.4931], ['used investigate associations', 0.4667], ['004', 0.4577], ['18 pg ml', 0.4455], ['95 ci', 0.4427], ['47 99 100', 0.4089], ['methods', 0.4058], ['years age', 0.4051], ['flanders', 0.405], ['ci', 0.3993], ['13 17', 0.3437]], 'topics': [{'topic_id': 25, 'weight': 0.444444, 'Representation': ['pbdes', 'bde', 'pbde', '209', 'brominated', 'congeners', 'congener', 'flame', 'diphenyl', 'polybrominated'], 'Enhanced_Topic': 'Exposure to polybrominated diphenyl ethers in pregnant women'}, {'topic_id': 9, 'weight': 0.222222, 'Representation': ['exposures', 'exposure', 'chemicals', 'cb', 'ths', 'aerosol', 'indoor', 'particles', 'diacetyl', 'dose'], 'Enhanced_Topic': 'Indoor air pollution and particle exposure assessment'}, {'topic_id': 1040, 'weight': 0.111111, 'Representation': ['menarche', 'pbde', 'pbdes', 'taps', 'tapping', 'bdes', 'finger', '029', 'reproduction', '154'], 'Enhanced_Topic': 'Association of PBDE exposure with menarche and cognitive function'}, {'topic_id': 704, 'weight': 0.222222, 'Representation': ['pbde', 'neurobehavioral', 'chamacos', 'motor', 'colostrum', 'coordination', 'decrements', 'poorer', 'attention', '209'], 'Enhanced_Topic': 'In utero and child PBDE exposure and neurobehavioral development'}]}
"""