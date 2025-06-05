# OUTPUT FORMAT for different stages

## Generate Pseudo Queries
這個步驟是 input document 及 prompt (whether given topic/keywords or not)，然後 prompt LLM or T5 to generate pseudo queries based on the given document

預期的 output 格式為 jsonl, 每一行為一個 json object，包含以下欄位：

- `id`: document id
- `title`: document title，如果不存在則為 empty string
- `text`: document text (就是內文)
- `predictied_queries`: 一個 string 將 generated pseudo queries concatenate 起來

## Evaluate Doc2Query Result
在有了上一步的 pseudo queries 之後，這個步驟會將 pseudo queries 與原始的 document text concat 起來，然後用 `pyterrier` build sparse (BM25) and dense (BGE-M3) index，並且 run `pyterrier.Experiment` 算 retrieval performance。 (refer to `src/eval.py`)
