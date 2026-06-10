# 架構說明

## 1. 整體數據流

```
 瀏覽器                                服務端 (RTX 4060 8GB / 64GB RAM)
┌─────────────────────┐              ┌──────────────────────────────────────┐
│ getUserMedia        │   PCM16      │  StreamingVAD (Silero, CPU)          │
│  + AEC/NS/AGC       │   16kHz      │   ├─ 端點檢測 → 完整語音段            │
│ AudioWorklet 採集 ───┼──binary────▶ │   └─ barge-in 檢測 (持續監聽)        │
│                     │   WebSocket  │            │SPEECH_END               │
│ 播放隊列 (可 flush)  │ ◀──binary────┤  WhisperASR (faster-whisper, CPU)   │
│ 狀態環 / 字幕        │ ◀──JSON──────┤            │text                     │
└─────────────────────┘              │  OllamaLLM ──逐句──▶ asyncio.Queue   │
                                     │                        │             │
                                     │  CSMSynthesizer (CSM-1B, GPU bf16)   │
                                     │   逐句合成 → 24kHz PCM 下發           │
                                     └──────────────────────────────────────┘
```

## 2. 全雙工與打斷 (barge-in)

麥克風在所有狀態下持續上行，VAD 持續運行。打斷的觸發與剎車路徑：

```
用戶開口 (AI 正在說話)
  └─ VAD 連續語音 ≥ barge_in_speech_ms (320ms, 高於普通閾值以抗回聲)
       ├─ ① respond_task.cancel()          → LLM 串流立即終止
       ├─ ② tts_interrupt.set()            → CSM StoppingCriteria 在
       │                                      下一個 token 停止解碼 (~80ms 粒度)
       ├─ ③ 下發 {"type":"interrupt"}      → 前端 flush 播放隊列, 立即靜音
       └─ ④ 用戶語音段繼續收集, SPEECH_END 後正常轉寫 → 新一輪回應
```

被打斷的回覆中「已說出的句子」會寫入對話歷史並標記，保證 LLM 的上下文
與用戶實際聽到的內容一致（不會出現 AI 以為自己說完了的幻覺）。

回聲防護三層：瀏覽器 AEC (第一層) → barge-in 閾值高於普通語音閾值
(第二層) → 建議耳機 (第三層，徹底消除)。

## 3. 延遲預算 (用戶說完 → 聽到第一個字)

v2 優化後的首音路徑: **首塊不等整句** —— LLM 流一湊出最早的可朗讀斷點
(句號 / 逗號 / ~9 詞) 就立刻送 CSM, CSM 只需生成 ~1 秒音頻即可開播。

| 階段 | 優化前 | 優化後 (4060 量級) |
|---|---|---|
| 端點判定 | 650 ms | 480 ms |
| ASR (small, CUDA fp16, beam 1, 已預熱) | ~400 ms+ (CPU) | ~200–300 ms |
| LLM 首塊 (2-9 詞, GPU + keep_alive 常駐) | ~700 ms+ (CPU/整句) | ~250–500 ms |
| CSM 首塊合成 (~1s 音頻, compile) | ~3–5 s (整句) | **~0.8–1.4 s** |
| **合計首音延遲** | **~5–7 s** | **~1.8–2.6 s** |

後續塊在首塊播放期間並行生成。服務端每輪打印 `[lat]` 日誌
(`asr= / llm_first_chunk= / FIRST AUDIO= / tts rtf=`), 直接看瓶頸在哪:
- `tts rtf` 持續 > 1.0 → torch.compile 沒生效或顯存溢出到共享內存,
  檢查啟動日誌是否有 "depth decoder compiled", `nvidia-smi` 看佔用
- `llm_first_chunk` > 1s → Ollama 模型被卸載過 (確認 keep_alive)、
  thinking 沒關掉, 或 `num_gpu` 沒把模型放進 VRAM
- `asr` > 500ms → 確認 `device: cuda` / `compute_type: float16`; 若 VRAM
  不足再退回 CPU 或換 `base`

## 4. CSM 的韻律上下文

CSM 與普通 TTS 的本質區別：它以對話為條件生成。我們把最近
`context_turns` 輪的 (文本, 音頻) —— 包括用戶的原始語音 —— 一併餵入，
模型會延續對話的語速、情緒與節奏，這正是 Sesame demo「像真人」的來源。
代價是上下文越長生成越慢，8GB 顯存建議 1–2 輪。

`voice_prompt` 開啟後，一段參考音頻會固定作為第一條上下文，
從而把基座模型「無固定音色」的輸出錨定為一致的聲線。

## 5. VRAM 分配 (preset: balanced)

| 組件 | 設備 | 佔用 |
|---|---|---|
| CSM-1B bf16 + Mimi | GPU | ~4.3 GB |
| KV cache / 激活 | GPU | ~1.0 GB |
| qwen2.5:3b q4 (Ollama) | GPU | ~2.2 GB |
| faster-whisper small int8 | CPU | 0 |
| Silero VAD | CPU | 0 |
| **合計** | | **~7.5 / 8 GB** |

若爆顯存：把 Ollama 設為純 CPU（`OLLAMA_NUM_GPU=0`，64GB RAM 跑 3B
非常輕鬆，~15-25 tok/s 對語音對話足夠），GPU 完全留給 CSM。

## 6. 已知限制

- **CSM-1B 僅可靠支持英文**。中文輸入可被識別（whisper），LLM 也能理解，
  但合成輸出設定為英文。如需中文語音輸出，需替換 TTS（超出
  「sesame 官方組件」範圍）或微調 CSM。
- 句級流式：句內不分塊（transformers 實現限制），靠流水線並行掩蓋。
- 單會話設計：一張 4060 同時只服務一個對話。
