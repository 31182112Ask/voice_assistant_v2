# voice_assistant_v2

本項目是一個本地運行的語音聊天 AI：瀏覽器負責麥克風採集和音頻播放，FastAPI 服務端串接 VAD、ASR、本地 LLM 和 TTS，目標是在普通消費級 GPU 上做可打斷、低延遲、接近自然對話節奏的語音交互。

當前版本對應提交：`6375118 fix: animate captions character by character`。

## 核心能力

- 全本地管線：Silero VAD、faster-whisper、Ollama LLM、Kyutai TTS / Sesame CSM 都在本機運行。
- 低延遲首音：LLM token 流切成可朗讀片段，TTS 盡早開始產生音頻。
- 真流式 TTS 路徑：`tts.backend: kyutai` 時支援 `synthesize_text_stream()`，可把 LLM 片段持續送入同一個 TTS 狀態。
- CSM 備用路徑：`tts.backend: csm` 時使用 Sesame CSM-1B；此路徑是短塊 pseudo-streaming，不是幀級 PCM 流式輸出。
- 可打斷：AI 說話時仍持續監聽麥克風，偵測到用戶說話會停止 LLM/TTS 任務並清空前端播放隊列。
- 主動發話：對話空窗時可由模型決定是否排程一段主動追問，並可提前合成等待播放。
- 字幕流式顯示：前端接收 `assistant_delta` 後逐字播放字幕，而不是整段一次性顯示。

## 推薦環境

- Windows 10/11
- Python 3.10 到 3.12
- NVIDIA GPU，建議 RTX 4060 8GB VRAM 或更高
- CUDA 12.x 驅動
- 64GB RAM 會更穩定
- Ollama

## 安裝

```powershell
git clone https://github.com/31182112Ask/voice_assistant_v2.git
cd voice_assistant_v2

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
```

安裝 Ollama 後拉取配置中的 LLM：

```powershell
ollama pull qwen3.5:2b
```

如果 `config.yaml` 中的 `llm.model` 改成其他模型，這裡也要同步拉取對應模型。

## 模型準備

首次在線準備模型：

```powershell
huggingface-cli login
python scripts/download_models.py
python scripts/check_deployment.py
```

完成下載後，服務啟動時會根據 `tts.local_files_only: true` 設置：

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

也就是部署運行階段會優先使用本地 Hugging Face 快取，不依賴外網。

## 啟動

推薦使用腳本啟動，因為它會先清理同目錄下殘留的 `server.main` 進程，並在 Ctrl+C / 退出時再次清理：

```powershell
.\scripts\start_server.ps1
```

如果需要讓 Ollama 盡量少佔 GPU：

```powershell
.\scripts\start_server.ps1 -CpuOllama
```

也可以直接啟動：

```powershell
python -m server.main
```

打開瀏覽器訪問：

```text
http://localhost:8000
```

## 配置重點

主要配置都在 `config.yaml`。

- `asr.model`：faster-whisper 模型大小。
- `asr.device` / `asr.compute_type`：ASR 使用 CPU 或 CUDA。
- `llm.model`：Ollama 模型名。
- `llm.think`：是否開啟模型思考輸出；語音對話通常應關閉，避免拖慢首 token。
- `llm.keep_alive`：建議設成 `-1`，避免 Ollama 閒置卸載模型。
- `tts.backend`：`kyutai` 是低延遲推薦路徑，`csm` 是 Sesame CSM 備用路徑。
- `tts.local_files_only`：部署時建議保持 `true`。
- `pipeline.tts_stream_input`：Kyutai 後端可啟用流式文字輸入。
- `pipeline.first_chunk_max_words`：控制首個可朗讀片段多短就送入 TTS。
- `proactive.enabled`：是否允許 AI 在沉默時自主發話。

## 延遲觀察

服務端會輸出 `[lat]` 日誌，重點看：

- `asr=`：語音識別耗時。
- `llm_first_chunk=`：LLM 首個可朗讀片段耗時。
- `FIRST AUDIO=`：用戶停頓後到第一段音頻下發的總體感知延遲。
- `tts rtf=` 或 `kyutai ... rtf=`：TTS 實時率，理想值小於或等於 1.0。

可以用基準腳本測量：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_latency.py --profile local
```

結果會輸出到 `latency_benchmark.json`，該文件被 `.gitignore` 忽略。

## 目錄和單文件功能

### 根目錄

| 文件 | 功能 |
|---|---|
| `.gitignore` | 忽略虛擬環境、快取、日誌、延遲測試輸出、語音參考文件和本地 patch/zip 產物。 |
| `README.md` | 項目部署、運行、配置和文件功能說明。 |
| `config.yaml` | 全局配置文件，定義服務端口、音頻參數、VAD、ASR、LLM、TTS、管線和主動發話策略。 |
| `requirements.txt` | Python 依賴列表，包含 FastAPI、PyTorch CUDA、Transformers、faster-whisper、Silero VAD、Moshi/Kyutai 等。 |

### 文檔

| 文件 | 功能 |
|---|---|
| `docs/ARCHITECTURE.md` | 架構說明，記錄前端、WebSocket、VAD、ASR、LLM、TTS、barge-in、主動發話和延遲路徑。 |
| `docs/LATENCY_BASELINE.md` | 延遲目標和基線，包含自然對話延遲要求、RTX 4060 本地目標和基準測試結果格式。 |

### 啟動與工具腳本

| 文件 | 功能 |
|---|---|
| `scripts/check_deployment.py` | 檢查依賴是否可 import，輸出 Python、Torch、CUDA、GPU、LLM、TTS 後端和本地模型配置。 |
| `scripts/download_models.py` | 預下載 Hugging Face TTS、faster-whisper 和 Ollama LLM，讓後續部署可離線啟動。 |
| `scripts/benchmark_latency.py` | 測量 ASR、LLM 首片段、TTS 首塊、RTF、模型首音和總體延遲，並按 `natural` / `local` 目標評估。 |
| `scripts/start_server.ps1` | Windows 推薦啟動腳本；建立虛擬環境、設置離線環境變量、可切換 CPU Ollama、啟動前後清理殘留服務進程。 |
| `scripts/stop_server.ps1` | 根據工作目錄和端口查找並停止殘留的 `server.main` Python 服務進程。 |

### 服務端

| 文件 | 功能 |
|---|---|
| `server/__init__.py` | Python 包標記文件。 |
| `server/config.py` | 讀取 `config.yaml`，並把 dict/list 遞歸轉成可用屬性訪問的 `SimpleNamespace`。 |
| `server/main.py` | FastAPI 入口；啟動時載入 ASR、LLM、TTS；提供首頁和 `/ws` WebSocket；負責會話生命周期。 |

### 語音管線

| 文件 | 功能 |
|---|---|
| `server/pipeline/__init__.py` | 管線子包標記文件。 |
| `server/pipeline/vad.py` | Silero VAD 流式封裝；處理端點判定、pre-roll、barge-in 語音偵測和投機 ASR 觸發片段。 |
| `server/pipeline/asr.py` | faster-whisper 封裝；在線程池中執行轉寫，避免阻塞 asyncio 事件循環。 |
| `server/pipeline/llm.py` | Ollama 聊天接口；支援 token streaming、首個可朗讀 chunk 切分、句子切分和主動發話排程請求。 |
| `server/pipeline/tts.py` | Sesame CSM-1B 後端；支援 voice prompt、對話音頻上下文、torch.compile、interrupt stopping criteria 和塊級音頻輸出。 |
| `server/pipeline/tts_kyutai.py` | Kyutai/Moshi TTS 後端；支援幀級音頻輸出和 `synthesize_text_stream()` 流式文字輸入，是當前低延遲推薦 TTS 路徑。 |
| `server/pipeline/orchestrator.py` | 核心會話調度器；串接 VAD、ASR、LLM、TTS，處理用戶語音、文字輸入、字幕事件、音頻下發、打斷、主動發話和歷史裁剪。 |

### 工具

| 文件 | 功能 |
|---|---|
| `server/utils/__init__.py` | 工具子包標記文件。 |
| `server/utils/audio.py` | 音頻工具函數；負責 PCM16 與 float32 轉換，以及簡單線性重採樣。 |

### 前端

| 文件 | 功能 |
|---|---|
| `web/index.html` | 單頁前端；使用 AudioWorklet 採集 16kHz PCM16，WebSocket 收發事件和音頻，播放 24kHz PCM，支援打斷清空、文字輸入、逐字字幕、狀態波形和對話導出。 |

## 被忽略但常用的本地文件

| 路徑 | 用途 |
|---|---|
| `voices/ref.wav` | CSM voice prompt 參考音頻；建議 3 到 5 秒，越短越省延遲。 |
| `voices/ref.txt` | `voices/ref.wav` 對應逐字文本。 |
| `latency_benchmark.json` | `scripts/benchmark_latency.py` 的本地測試輸出。 |
| `*.log` | 本地服務日誌。 |
| `*.patch` / `*.zip` | 本地優化補丁或打包產物，默認不作為正式版本提交。 |

## 常見問題

### 離線模式找不到模型

如果看到 `HF_HUB_OFFLINE` 相關錯誤，說明本地 Hugging Face 快取沒有對應模型。先在有網路時執行：

```powershell
huggingface-cli login
python scripts/download_models.py
```

### 顯存不足

優先嘗試：

- 使用 `tts.backend: kyutai`，降低 CSM 壓力。
- 把 ASR 改到 CPU 或更小模型。
- 用 `.\scripts\start_server.ps1 -CpuOllama` 讓 Ollama 不佔 GPU。
- 降低 `pipeline.first_chunk_max_words` 和 TTS 上下文長度。

### 字幕不是逐字出現

前端只有在收到 `assistant_delta` 後才會逐字排隊顯示。若後端一次性只發 `assistant_sentence`，視覺上會接近整段顯示；低延遲路徑應使用 LLM chunk 流和 Kyutai stream-in。

## GitHub

遠端倉庫：

```text
https://github.com/31182112Ask/voice_assistant_v2.git
```
