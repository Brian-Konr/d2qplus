# D2Q+ 專案用戶指南

## 📖 專案概述

D2Q+ (Document-to-Query Plus) 是一個結合主題建模 (Topic Modeling) 和強化學習 (Reinforcement Learning) 的文檔查詢生成系統。該系統能夠分析文檔內容，提取主題信息，並生成高品質的偽查詢 (Pseudo Queries) 來增強檢索效果。

## 🏗️ 專案架構

```
d2qplus/
├── 📁 data/                    # 數據集和原始資料
├── 📁 src/                     # 核心程式碼
├── 📁 scripts/                 # 執行腳本
├── 📁 experiments/             # 實驗相關程式碼
├── 📁 prompts/                 # 提示模板
├── 📁 outputs/                 # 模型輸出
├── 📁 built-index/             # 建立的索引
├── 📁 augmented-data/          # 增強後的數據
├── 📁 gen/                     # 生成的查詢
└── 📁 eval/                    # 評估結果
```

## 📂 目錄結構詳解

### 🗄️ `data/` - 數據集目錄
**用途**: 存放各種數據集和預處理工具

**重要檔案**:
- `nfcorpus/`, `scidocs/`, `trec-car-10000/` - 標準檢索數據集
- `CSFCube-1.1/` - 科學文獻語料庫
- `fiqa/`, `MSMarco_qrels/` - 問答和檢索評估數據
- `process.ipynb` - 數據預處理筆記本
- `utils.py` - 數據處理工具函數

**使用場景**: 
- 新增數據集時放在此目錄
- 數據預處理和清理

### 💻 `src/` - 核心程式碼
**用途**: 系統的主要功能實現

**重要檔案**:
- `eval.py` - 評估框架，建立索引和計算檢索性能
- `generate.py` - 查詢生成主程式
- `grpo_trainer.py` - GRPO 強化學習訓練器
- `reward.py` - 獎勵函數實現
- `topic-modeling/` - 主題建模子模組
  - `run_topic_modeling.py` - 主題建模主程式
  - `dvdo.py` - DVDO 演算法實現

**子目錄**:
- `utils/` - 工具函數和常數定義

### 🔧 `scripts/` - 執行腳本
**用途**: 各種功能的執行腳本

**重要檔案**:
- `topic-modeling/` - 主題建模相關腳本
  - `topic_modeling.sh` - 基礎主題建模
  - `dvdo.sh` - DVDO 演算法執行
  - `get_llm_representation.sh` - LLM 增強主題表示
- `grpo_trainer.sh` - 強化學習訓練
- `eval.sh` - 評估腳本
- `zero_shot.sh` - 零樣本查詢生成

### 🧪 `experiments/` - 實驗程式碼
**用途**: 各種實驗和分析工具

**重要檔案**:
- `zero_shot.py` - 零樣本實驗
- `filter.py` - 查詢過濾器
- `score_generator.py` - 分數生成器
- `analysis/` - 分析工具
- `rl/` - 強化學習實驗
- `reranking/` - 重排序實驗

### 📝 `prompts/` - 提示模板
**用途**: 儲存各種 LLM 提示模板

**重要檔案**:
- `enhance_NL_topic.txt` - 主題增強提示
- `user_prompt_template.txt` - 用戶提示模板
- `promptagator/` - Promptagator 相關提示

### 📊 輸出目錄
- `outputs/` - 模型訓練輸出
- `built-index/` - PyTerrier 建立的檢索索引
- `augmented-data/` - 增強後的數據集
- `gen/` - 生成的偽查詢
- `eval/` - 評估結果

## 🚀 快速開始

### 1. 環境設置
```bash
# 安裝依賴
pip install -r requirements.txt

# 設定 CUDA 設備（可選）
export CUDA_VISIBLE_DEVICES=0
```

### 2. 完整工作流程

#### 步驟 1: 主題建模
```bash
# 編輯參數設定
vim scripts/topic-modeling/topic_modeling.sh

# 執行主題建模
bash scripts/topic-modeling/topic_modeling.sh
```

#### 步驟 2: LLM 增強主題表示
```bash
# 使用 LLM 增強主題表示
bash scripts/topic-modeling/get_llm_representation.sh
```

#### 步驟 3: 生成偽查詢
```bash
# 零樣本生成
bash scripts/zero_shot.sh

# 或使用強化學習訓練
bash scripts/grpo_trainer.sh
```

#### 步驟 4: 評估效果
```bash
# 評估檢索性能
bash scripts/eval.sh
```

## 🔧 核心功能詳解

### 主題建模 (Topic Modeling)
- **檔案**: `src/topic-modeling/run_topic_modeling.py`
- **腳本**: `scripts/topic-modeling/topic_modeling.sh`
- **功能**: 使用 BERTopic 對文檔進行主題建模
- **輸出**: 主題資訊 DataFrame、模型檔案

### DVDO 演算法
- **檔案**: `src/topic-modeling/dvdo.py`
- **腳本**: `scripts/topic-modeling/dvdo.sh`
- **功能**: 動態向量維度優化和語句級分塊
- **特色**: 結合 silhouette 和 coherence 評分

### 查詢生成
- **檔案**: `src/generate.py`
- **功能**: 使用 VLLM 和各種提示模板生成偽查詢
- **支援模板**: D2Q, Promptagator, InPars

### 強化學習訓練
- **檔案**: `src/grpo_trainer.py`
- **功能**: 使用 GRPO 演算法優化查詢生成
- **獎勵函數**: 主題覆蓋率、關鍵詞覆蓋率等

### 評估框架
- **檔案**: `src/eval.py`
- **功能**: 建立 BM25 和密集檢索索引，計算檢索性能
- **支援**: PyTerrier 整合

## 📋 配置檔案說明

### 主題建模配置
在 `scripts/topic-modeling/topic_modeling.sh` 中設定:
```bash
CORPUS_PATH="數據路徑"
EMBED_MODEL="嵌入模型"
MIN_TOPIC_SIZE=5
CHUNK_MODE="sentence"  # 或 "window"
```

### 查詢生成配置
在 `src/generate.py` 中可調整:
- 模型選擇 (`--model`)
- 採樣參數 (`--temperature`, `--max_tokens`)
- 提示模板 (`--prompt_template`)

## 🔍 常見使用場景

### 場景 1: 新數據集處理
1. 將數據集放入 `data/` 目錄
2. 使用 `data/process.ipynb` 進行預處理
3. 執行主題建模流程

### 場景 2: 模型實驗
1. 修改 `experiments/` 中的實驗腳本
2. 調整參數設定
3. 運行實驗並記錄結果

### 場景 3: 評估新方法
1. 在 `src/` 中實現新方法
2. 創建對應的執行腳本
3. 使用 `src/eval.py` 進行評估

## 🛠️ 開發指南

### 新增功能
1. 在 `src/` 中實現核心邏輯
2. 在 `scripts/` 中創建執行腳本
3. 在 `experiments/` 中添加實驗代碼
4. 更新此文檔

### 程式碼規範
- 使用有意義的變數名稱
- 添加適當的註釋和文檔字串
- 遵循 Python PEP 8 規範

### 錯誤處理
- 檢查檔案路徑是否存在
- 驗證輸入參數格式
- 適當的異常處理

## 📈 性能優化建議

1. **GPU 使用**: 設定 `CUDA_VISIBLE_DEVICES` 環境變數
2. **批次處理**: 調整批次大小以最大化 GPU 利用率
3. **記憶體管理**: 大數據集時使用分塊處理
4. **並行處理**: 利用多核心 CPU 進行預處理

## ❓ 常見問題

### Q: 如何添加新的數據集？
A: 將數據集放入 `data/` 目錄，確保格式為 JSONL，包含 `_id` 和 `text` 欄位。

### Q: 如何自定義獎勵函數？
A: 修改 `src/reward.py`，實現新的獎勵計算邏輯。

### Q: 如何調整主題數量？
A: 修改 `scripts/topic-modeling/topic_modeling.sh` 中的 `MIN_TOPIC_SIZE` 參數。

### Q: 評估結果在哪裡查看？
A: 評估結果會保存在 `eval/` 目錄中，包含各種檢索指標。

## 🔗 相關資源

- **BERTopic 文檔**: https://maartengr.github.io/BERTopic/
- **PyTerrier 文檔**: https://pyterrier.readthedocs.io/
- **VLLM 文檔**: https://docs.vllm.ai/

## 📞 聯絡資訊

如有任何問題或建議，請透過以下方式聯絡：
- 創建 GitHub Issue
- 聯絡專案維護者

---

**最後更新**: 2025年6月9日
**版本**: 1.0
