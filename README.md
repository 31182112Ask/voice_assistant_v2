# voice_assistant_v2

本地全雙工語音對話助手 —— 基於 **Sesame 官方開源的 CSM-1B** 語音生成模型，
對標 Sesame 官方 demo 的對話體驗：**可隨時打斷、韻律連貫、低延遲流水線**。
全部組件本地運行，無任何雲端依賴。

> 目標硬件：RTX 4060 (8GB VRAM) + 64GB RAM ✅

```
你說話 ──▶ Silero VAD ──▶ faster-whisper ──▶ 本地 LLM (Ollama)
                │                                  │ 逐句流式
                │ barge-in 隨時打斷                 ▼
你聽到 ◀── 瀏覽器播放 ◀──────────── Sesame CSM-1B (GPU)
```

| 能力 | 實現 |
|---|---|
| 語音生成 | **CSM-1B**（Sesame 官方開源，Llama backbone + Mimi codec，官方 demo 同源組件） |
| 韻律自然 | 帶對話上下文生成：把最近幾輪的文本+音頻餵給 CSM，語氣節奏自然銜接 |
| 全雙工 | 麥克風永不關閉，AI 說話時持續監聽 |
| 可打斷 | 開口 ~0.3 秒即打斷：LLM 停、CSM token 級停、播放隊列清空 |
| 低延遲 | LLM→TTS 句級流水線並行；首音 ~2.5–3.5s，後續句無縫 |
| 語音識別 | faster-whisper（默認 CPU，零顯存佔用） |
| 對話模型 | Ollama + qwen2.5:3b（可換任意模型） |

---

## 一、環境準備

### 0. 前置
- Python 3.10–3.12、[Ollama](https://ollama.com)、NVIDIA 驅動 + CUDA 12.x
- Hugging Face 帳號，並在 [sesame/csm-1b](https://huggingface.co/sesame/csm-1b) 頁面**接受模型授權**（gated model）

### 1. 安裝依賴

```bash
git clone https://github.com/31182112Ask/voice_assistant_v2.git
cd voice_assistant_v2

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate

# 先裝匹配 CUDA 的 PyTorch (CUDA 12.x):
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

### 2. 下載模型

```bash
huggingface-cli login                # 粘貼你的 HF token
python scripts/download_models.py    # CSM-1B + whisper + ollama pull
```

### 3. 啟動

```bash
# 終端 1: 確保 Ollama 在運行 (安裝後通常已是服務)
ollama serve

# 終端 2:
python -m server.main
```

打開 **http://localhost:8000** → 點「開始對話」→ 授權麥克風 → 直接說話。
AI 說話時你隨時開口即可打斷。**強烈建議佩戴耳機**（最佳打斷體驗）。

> 首次啟動較慢：CSM 加載 + torch.compile 預熱約 1–3 分鐘，之後常駐顯存。

---

## 二、8GB 顯存方案

默認配置（`config.yaml`）的顯存分配：

| 組件 | 設備 | 佔用 |
|---|---|---|
| CSM-1B (bf16) + Mimi | GPU | ~4.3 GB |
| qwen2.5:3b (Ollama q4) | GPU | ~2.2 GB |
| whisper small (int8) | CPU | 0 GB |
| 合計 + 緩衝 | | ~7.5 GB |

**顯存吃緊時**（桌面環境本身佔 0.5–1GB 很常見）：

```bash
# 把 LLM 完全放 CPU —— 64GB RAM 跑 3B 毫無壓力, GPU 全留給 CSM
# Windows:  set OLLAMA_NUM_GPU=0 && ollama serve
# Linux:    OLLAMA_NUM_GPU=0 ollama serve
```

---

## 三、調優手冊（config.yaml）

| 想要 | 改什麼 |
|---|---|
| 回應更快/更搶話 | `vad.min_silence_ms: 650 → 500` |
| 打斷更靈敏 | `vad.barge_in_speech_ms: 320 → 250`（外放容易誤觸，耳機可放心調低） |
| 合成更快 | `tts.context_turns: 2 → 1`；保持 `compile_decoder: true` |
| 識別更準 | `asr.model: small → medium`（CPU 稍慢）或 `device: cuda`（佔 ~1GB 顯存） |
| 固定音色 | 放一段 5–15 秒乾淨人聲 `voices/ref.wav` + 其逐字稿 `voices/ref.txt`，開 `tts.voice_prompt.enabled: true` |
| 換 LLM | `llm.model: llama3.2:3b` / `qwen2.5:7b`（7B q4 約 4.7GB，需 LLM 放 CPU） |

## 四、重要說明

- **語言**：CSM-1B 官方說明僅可靠支持英文。本項目中你可以用中文說話
  （whisper 能識別、LLM 能理解），但 AI 的語音回覆為英文。
  中文語音輸出需更換 TTS 或自行微調 CSM。
- **倫理**：請遵守 Sesame 的使用條款 —— 禁止未經同意克隆真人聲音、
  禁止用於欺詐或冒充。
- 架構細節（時序圖、打斷剎車路徑、延遲預算）見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 五、上傳到你的 GitHub

```bash
cd voice_assistant_v2
git init
git add .
git commit -m "feat: full-duplex local voice assistant with Sesame CSM-1B"
git branch -M main
git remote add origin https://github.com/31182112Ask/voice_assistant_v2.git
git push -u origin main
```

## 目錄結構

```
voice_assistant_v2/
├── config.yaml                 # 全部可調參數（含 8GB 顯存預算注釋）
├── server/
│   ├── main.py                 # FastAPI + WebSocket 入口
│   ├── config.py
│   ├── pipeline/
│   │   ├── orchestrator.py     # 全雙工狀態機 + barge-in（核心）
│   │   ├── vad.py              # Silero VAD 端點/打斷檢測
│   │   ├── asr.py              # faster-whisper
│   │   ├── llm.py              # Ollama 流式 + 逐句切分
│   │   └── tts.py              # CSM-1B：上下文生成 + 可中斷解碼
│   └── utils/audio.py
├── web/index.html              # 前端：AudioWorklet 採集 + 可清空播放隊列
├── scripts/download_models.py
├── voices/                     # （可選）參考音色
└── docs/ARCHITECTURE.md
```

## 致謝

- [Sesame CSM](https://github.com/SesameAILabs/csm) — 官方開源語音生成模型
- faster-whisper / Silero VAD / Ollama / Qwen
