# OUTPUT FORMAT for different stages

## Topic Modeling
這個步驟會將 input document 進行 topic modeling，並且將每個 topic 的代表詞彙、代表句子、以及 topic info 存成一個 dataframe，然後存成 pickle & csv 檔案。同時也會存 topic model 成 pickle，到時候 training 的時候可以 `BERTopic.load` 來用於 `transform` generated query 的 topic。

請參考 `scripts/topic-modeling/topic_modeling.sh` 這個腳本，裡面有詳細的參數設定說明。

### Enhance Topic Representation using LLM
接下來我們可以更進一步把 topic 的 representation 從 bag of words 轉換成 LLM representation，這樣可以讓 topic 的表達更有語意。
請參考 `scripts/topic-modeling/get_llm_representation.sh` 這個腳本，裡面有詳細的參數設定說明。最後會在第一階段的 topic model info dataframe 新增一個欄位 `Enhanced_Topic`，可用於後續 guide LLM 生成 pseudo queries 及 RL training 的 prompts。

## RL Training



## Generating Pseudo Queries
這個步驟是 input document 及 prompt (whether given topic/keywords or not)，然後 prompt LLM or T5 to generate pseudo queries based on the given document

預期的 output 格式為 jsonl, 每一行為一個 json object，包含以下欄位：

- `id`: document id
- `title`: document title，如果不存在則為 empty string
- `text`: document text (就是內文)
- `predictied_queries`: 一個 string 將 generated pseudo queries concatenate 起來

## Evaluate Doc2Query Result
在有了上一步的 pseudo queries 之後，這個步驟會將 pseudo queries 與原始的 document text concat 起來，然後用 `pyterrier` build sparse (BM25) and dense (BGE-M3) index，並且 run `pyterrier.Experiment` 算 retrieval performance。 (refer to `src/eval.py`)
