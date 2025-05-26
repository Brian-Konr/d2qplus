from util import read_jsonl, read_txt

def prepare_training_data(integrated_data_path, 
                          rl_user_prompt_template_path="/home/guest/r12922050/GitHub/d2qplus/prompts/rl_user_prompt_template.txt", 
                          drop_no_topics=True, 
                          max_keywords=10, 
                          max_topics=5):
    # see 
    data = read_jsonl(integrated_data_path)

    if drop_no_topics:
        data = [d for d in data if d['topics']]
    # prepare prompt
    for d in data:
        doc_content = d['title'] + "\n" + d['text']
        keywords = d['keywords']
        topics = d['topics']

        # sort keywords based on score
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:max_keywords]
        keywords_str = ', '.join([f"{k[0]}" for k in keywords])

        # format topics
        topics = sorted(topics, key=lambda x: x['weight'], reverse=True)[:max_topics]
        topics_str = ', '.join([f"{t['Enhanced_Topic']} ({t['weight']})" for t in topics])

        # create prompt
        prompt_template = read_txt(rl_user_prompt_template_path)
        
        prompt = prompt_template.replace("[DOCUMENT]", doc_content).replace("[KEYWORDS]", keywords_str).replace("[TOPICS]", topics_str)
        
        d['prompt'] = prompt
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
    pass
    # - Save document vectors -
    
    # embed_model_name = "allenai/scibert_scivocab_uncased"
    # data_path = "/home/guest/r12922050/GitHub/d2qplus/data/nfcorpus/corpus.jsonl"
    # out_path = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/corpus-lookup/document_vectors.pt"
    # save_document_vectors(embed_model_name, data_path, out_path)



    # import json
    # INTEGRATED_DATA_PATH = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/integrated/data.jsonl"
    # DATA_WITH_PROMPT_OUT_PATH = "/home/guest/r12922050/GitHub/d2qplus/augmented-data/nfcorpus/integrated/data_with_prompt.jsonl"
    # data = prepare_training_data(integrated_data_path=INTEGRATED_DATA_PATH)

    # with open(DATA_WITH_PROMPT_OUT_PATH, 'w') as f:
    #     for d in data:
    #         f.write(json.dumps(d) + '\n')









"""
Example of integrated data:
{'title': 'Neurobehavioral function and low-level exposure to brominated flame retardants in adolescents: a cross-sectional study', 'text': 'Background Animal and in vitro studies demonstrated a neurotoxic potential of brominated flame retardants, a group of chemicals used in many household and commercial products to prevent fire. Although the first reports of detrimental neurobehavioral effects in rodents appeared more than ten years ago, human data are sparse. Methods As a part of a biomonitoring program for environmental health surveillance in Flanders, Belgium, we assessed the neurobehavioral function with the Neurobehavioral Evaluation System (NES-3), and collected blood samples in a group of high school students. Cross-sectional data on 515 adolescents (13.6-17 years of age) was available for the analysis. Multiple regression models accounting for potential confounders were used to investigate the associations between biomarkers of internal exposure to brominated flame retardants [serum levels of polybrominated diphenyl ether (PBDE) congeners 47, 99, 100, 153, 209, hexabromocyclododecane (HBCD), and tetrabromobisphenol A (TBBPA)] and cognitive performance. In addition, we investigated the association between brominated flame retardants and serum levels of FT3, FT4, and TSH. Results A two-fold increase of the sum of serum PBDE’s was associated with a decrease of the number of taps with the preferred-hand in the Finger Tapping test by 5.31 (95% CI: 0.56 to 10.05, p\u2009=\u20090.029). The effects of the individual PBDE congeners on the motor speed were consistent. Serum levels above the level of quantification were associated with an average decrease of FT3 level by 0.18 pg/mL (95% CI: 0.03 to 0.34, p\u2009=\u20090.020) for PBDE-99 and by 0.15 pg/mL (95% CI: 0.004 to 0.29, p\u2009=\u20090.045) for PBDE-100, compared with concentrations below the level of quantification. PBDE-47 level above the level of quantification was associated with an average increase of TSH levels by 10.1% (95% CI: 0.8% to 20.2%, p\u2009=\u20090.033), compared with concentrations below the level of quantification. We did not observe effects of PBDE’s on neurobehavioral domains other than the motor function. HBCD and TBBPA did not show consistent associations with performance in the neurobehavioral tests. Conclusions This study is one of few studies and so far the largest one investigating the neurobehavioral effects of brominated flame retardants in humans. Consistently with experimental animal data, PBDE exposure was associated with changes in the motor function and the serum levels of the thyroid hormones.', 'keywords': [['neurobehavioral effects', 0.5726], ['levels polybrominated diphenyl', 0.5509], ['multiple regression models', 0.5153], ['blood samples', 0.5007], ['studies demonstrated', 0.4931], ['used investigate associations', 0.4667], ['004', 0.4577], ['18 pg ml', 0.4455], ['95 ci', 0.4427], ['47 99 100', 0.4089], ['methods', 0.4058], ['years age', 0.4051], ['flanders', 0.405], ['ci', 0.3993], ['13 17', 0.3437]], 'topics': [{'topic_id': 25, 'weight': 0.444444, 'Representation': ['pbdes', 'bde', 'pbde', '209', 'brominated', 'congeners', 'congener', 'flame', 'diphenyl', 'polybrominated'], 'Enhanced_Topic': 'Exposure to polybrominated diphenyl ethers in pregnant women'}, {'topic_id': 9, 'weight': 0.222222, 'Representation': ['exposures', 'exposure', 'chemicals', 'cb', 'ths', 'aerosol', 'indoor', 'particles', 'diacetyl', 'dose'], 'Enhanced_Topic': 'Indoor air pollution and particle exposure assessment'}, {'topic_id': 1040, 'weight': 0.111111, 'Representation': ['menarche', 'pbde', 'pbdes', 'taps', 'tapping', 'bdes', 'finger', '029', 'reproduction', '154'], 'Enhanced_Topic': 'Association of PBDE exposure with menarche and cognitive function'}, {'topic_id': 704, 'weight': 0.222222, 'Representation': ['pbde', 'neurobehavioral', 'chamacos', 'motor', 'colostrum', 'coordination', 'decrements', 'poorer', 'attention', '209'], 'Enhanced_Topic': 'In utero and child PBDE exposure and neurobehavioral development'}]}
"""