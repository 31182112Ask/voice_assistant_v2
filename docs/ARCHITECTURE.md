# 架構說明

## 1. 整體數據流

```
 瀏覽器                                服務端 (RTX 4060 8GB / 64GB RAM)
┌─────────────────────┐              ┌──────────────────────────────────────┐
│ getUserMedia        │   PCM16      │  StreamingVAD (Silero, CPU)          │
│  + AEC/NS/AGC       │   16kHz      │   ├─ 端點檢測 → 完整語音段            │
│ AudioWorklet 採集 ───┼──binary────▶ │   └─ barge-in 檢測 (持續監聽)        │
│                     │   WebSocket  │            │SPEECH_END               │
│ 播放隊列 (可 flush)  │ ◀──binary────┤  WhisperASR (faster-whisper, GPU)   │
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

## 3. TTS 延遲的根因與雙後端設計

### 為什麼 CSM 是結構性瓶頸

CSM 每 80ms 音頻 = 1 次 1B backbone 前向 + 31 次**串行** depth decoder
前向 ≈ 每秒音頻 ~400 次前向, 4060 上 RTF 最好也只在 1.0 附近。
更關鍵的是 transformers 的 CSM 實現**無法在生成中途取出音頻**
(無幀級 streamer, HF 官方未支持), 所以無論塊切多短:

```
首音延遲下界 = 塊音頻時長 × RTF   ← 「整塊生成完才出聲」, 調參無法突破
```

### 解法: 幀級流式後端 (tts.backend: kyutai)

Kyutai TTS 1.6B 與 CSM **同源同原理** (transformer → Mimi codec 音頻碼;
Mimi 正是 Kyutai 為 Moshi 研發、被 CSM 採用的 codec), 但其 Delayed
Streams 架構原生支持文本邊進、音頻邊出: 每解出一幀 (80ms) 立即下發,
攢 2 幀 (~160ms) 即開播。RTF ≤1 時播放永不斷流。

| | CSM-1B (transformers) | Kyutai TTS 1.6B |
|---|---|---|
| 原理 | Llama backbone + Mimi | DSM transformer + Mimi (同 codec) |
| 出聲方式 | 整塊生成完 | **逐幀流式 (80ms)** |
| TTS 首音 | 塊時長 × RTF (~1-2.5s) | **~0.2-0.4s** |
| 韻律上下文 | ✓ 對話音頻條件生成 (最強) | ✗ (音色靠預置 voice) |
| 聲音克隆 | ✓ ref.wav 上下文錨定 | 僅預置庫 (嵌入模型未開源) |
| VRAM (bf16) | ~4.3 GB | ~3.5 GB |
| 許可 | Apache-2.0 (gated) | CC-BY-4.0 |

`config.yaml → tts.backend` 一鍵切換; kyutai 加載失敗自動回退 CSM。

### 其他可選模型 (同為 LLM→codec 流式路線)

- **Orpheus 400M/150M**: Llama+SNAC, 原生流式, 更小更快, 英文為主
- **Kokoro-82M**: StyleTTS2 路線, RTF ~0.03 (CPU 可跑), 無克隆/上下文,
  追求極致穩定低延遲時的兜底選擇
- **CSM + csm-streaming (社區)**: 保住 CSM 音色的幀級流式, 但依賴重
  (torchtune/moshi 原版棧), RTF 硬約束仍在

## 3b. 優化後首音預算 (kyutai 後端)

| 階段 | 耗時 |
|---|---|
| 端點判定 | 480 ms |
| ASR (GPU) | ~150 ms |
| LLM 首塊 (≤5 詞) | ~200–350 ms |
| Kyutai 首幀×2 | **~200–350 ms** |
| **合計** | **~1.0–1.3 s** ≈ 自然對話停頓 |

## 3c. 舊延遲預算 (CSM 後端, 參考)

v2 優化後的首音路徑: **首塊不等整句** —— LLM 流一湊出最早的可朗讀斷點
(句號 / 逗號 / ~5 詞) 就立刻送 CSM。首塊不帶歷史音頻上下文, 只用短
voice prompt 錨定音色, 讓 CSM 只需生成很短的音頻即可開播。

| 階段 | 優化前 | 優化後 (4060 量級) |
|---|---|---|
| 端點判定 | 650 ms | 480 ms |
| ASR (small, CUDA fp16, beam 1, 已預熱) | ~400 ms+ (CPU) | ~200–300 ms |
| LLM 首塊 (1-5 詞, GPU + keep_alive 常駐) | ~700 ms+ (CPU/整句) | ~250–500 ms |
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
低延遲 preset 會把 voice prompt 截到 `max_prompt_s`，並且只在首個
TTS chunk 使用；後續 chunk 使用上一塊生成音頻延續音色，避免每塊都
重做固定前綴 prefill。

## 5. CSM Streaming 現狀

目前 Hugging Face `CsmForConditionalGeneration.generate()` 不是 PCM/audio
streaming。它雖然暴露標準 `streamer` 參數, 但 streamer 收到的是生成中的
audio-code token; `codec_model.decode(...)` 仍是在整段 code 生成完後一次性執行。
因此本專案採用的是 pseudo-streaming:

- LLM token 流即時切成短的可朗讀 chunk
- 首個 chunk 盡可能短, 並且不帶歷史音頻上下文
- 首塊保留短 voice prompt 錨定音色; 後續 chunk 使用上一塊生成音頻延續音色,
  不再每塊重餵 voice prompt

若要真正做到 CSM 邊生成邊播放, 需要自訂 CSM generation loop: 收集若干 Mimi
code frame 後增量呼叫 codec decoder 並處理 overlap / 邊界平滑。官方 Transformers
封裝目前沒有現成 PCM frame streaming API。

## 6. VRAM 分配 (preset: low latency)

| 組件 | 設備 | 佔用 |
|---|---|---|
| CSM-1B bf16 + Mimi | GPU | ~4.3 GB |
| KV cache / 激活 | GPU | ~1.0 GB |
| qwen3.5:2b (Ollama) | GPU | ~2.2 GB |
| faster-whisper small fp16 | GPU | ~0.5–0.9 GB |
| Silero VAD | CPU | 0 |
| **合計** | | **~6.4–7.4 / 8 GB** |

若爆顯存：把 Ollama 設為純 CPU（`OLLAMA_NUM_GPU=0`，64GB RAM 跑 3B
非常輕鬆，~15-25 tok/s 對語音對話足夠），GPU 完全留給 CSM。

## 7. 已知限制

- **CSM-1B 僅可靠支持英文**。中文輸入可被識別（whisper），LLM 也能理解，
  但合成輸出設定為英文。如需中文語音輸出，需替換 TTS（超出
  「sesame 官方組件」範圍）或微調 CSM。
- pseudo-streaming：LLM 是 token streaming, 但 CSM 仍需每個 chunk 生成完
  才能返回音頻；目前靠短 chunk 和流水線並行掩蓋。
- 單會話設計：一張 4060 同時只服務一個對話。

## 7. 主動發話 (Proactive Speech)

像 Sesame / MiniCPM-o 的主動性: 靜默時 AI 自主決定是否開口。

**時鐘 = 音頻流本身, 無系統計時器。** 與 Moshi 等原生全雙工模型
同構: 雙工模型每 80ms 必然收到一幀 (含靜音幀), 時間以幀數形式
存在於流中, 「決定開口」只是下一幀預測從靜音切到語音。本項目同樣:
麥克風每 32ms 產生一個 VAD 幀, on_audio 路徑逐幀數靜音 = 計時。
麥克風斷流 → 時鐘自然停擺。

```
on_audio → 每個 VAD 幀 → _tick_silence (無 asyncio 計時任務)
  ├─ 有聲幀 / 非聆聽態 / 播放中 → 靜默幀計數清零
  └─ 靜默幀數 ≥ delays_s[第n次] × 31.25
       └─ 把 nudge (含靜默秒數) 附加到歷史末尾送 LLM (不寫入歷史)
            ├─ LLM 輸出 "PASS"  → 整輪靜默, 計數重新累計
            └─ LLM 輸出一句話   → 走正常 TTS 流水線播出,
                                  前端標「主動」, 計數+1
```

關鍵設計:
- **決策權在 LLM**: 不是定時播報。LLM 看完整上下文判斷該追問、
  補充、輕聲確認, 還是保持沉默 (對話已自然收尾時輸出 PASS)。
- **退避時間表** `delays_s: [12, 28, 60]`: 越不被回應, 等得越久;
  連續 `max_consecutive` 次無回應後徹底安靜, 等用戶先開口。
- **用戶任何回話 → 計數歸零**; **打斷主動發話 → 冷靜期**
  (`cooldown_after_interrupt_s`), 避免顯得糾纏。
- 主動模式不顯示「思考」狀態 (靜默決策), 開口才切「說話」;
  即時 ack 在主動模式下禁用 (不會憑空冒出 "Okay.")。
- 全雙工不受影響: 主動發話同樣可被秒打斷 (barge-in 同一條路)。

### 7b. 模式 A: 模型自主排程 (mode: model_scheduled, 推薦)

時間的決定權也交給模型 —— 系統連「何時觸發」都不再決定:

```
每輪回覆結束
  └─ LLM 規劃: 回 "NONE" 或 "WAIT=18 | Still mulling it over?"
       └─ WAIT → 立即預合成該句 (GPU 此刻空閒), 緩存 PCM
            └─ 流時鐘數到 18s × 31.25 幀 → 直接播緩存 (零生成延遲)
                 └─ 播完鏈式規劃下一句; 模型回 NONE 即徹底安靜
```

- **時機與內容皆由模型決定**: 任務進行中它會選短等待追問,
  用戶需要思考時選長等待, 對話收尾時回 NONE。
- **零延遲**: 觸發時刻只是播放緩存, 無 LLM/TTS 在關鍵路徑上。
- **計劃以沉默為前提**: 用戶開口 / barge-in / reset → 計劃即作廢
  (預合成中途也會被 interrupt event 中止), 新回覆後重新規劃。
- 模式 B (nudge, 到點即時詢問) 保留, config 一鍵切換。

## 8. 投機轉寫 + 語義端點 (延遲主軸 v3)

```
用戶說話 ──┐
          靜音 180ms ──▶ 投機 ASR (轉寫「語音至此」)
          │                ├─ 文本以 . ! ? 收尾 → 語義端點: force_end
          │                │   立即收束本段 (不等滿額靜音, 省 ~200-300ms)
          │                └─ 否則: 端點到來時轉寫已完成 (ASR=0ms)
          └─ 又開口 → 投機作廢, 重新累計
```

延遲收益 (kyutai 後端疊加後的首音預算):
| 階段 | v2 | v3 |
|---|---|---|
| 端點等待 | 480ms | **~210ms** (語義端點命中時) |
| ASR | ~150-250ms | **~0ms** (投機命中時) |
| LLM 首塊 | ~250-350ms | 不變 |
| TTS 首幀 | ~200-350ms | 不變 |
| **合計** | ~1.1-1.4s | **~0.7-0.9s** ≈ 真人接話節奏 |

備註: 投機在尾隨靜音期間進行, CPU/GPU 此刻本就空閒, 失敗(用戶續說)
僅浪費一次閒置算力; `pipeline.speculative_asr / semantic_endpoint` 可關。

## 9. 前端實用件

文字輸入 (Enter 直通回應流水線, 回覆仍是語音)、靜音開關 (M 鍵;
停發音頻幀 → 服務端流時鐘凍結, 主動發話自然暫停)、每輪首音延遲
徽章 (說話人欄下方, 如 "1.2s")、逐字稿導出 (markdown)、斷線自動
重連 (指數退避, 最大 10s)。

## 7. 投機級聯 (speculative cascade)

把「等待」全部換成「並行」: 用戶句中停頓 ~180ms 起, 各級就提前開跑——

```
用戶語音 ────────────╖ 停頓180ms        端點(192-320ms / 語義端點即刻)
                     ╟─ 投機 ASR ──完成─╢
                     ║                  ╟─ 投機 LLM (primer) ──首塊─╢
                     ║                  ║   端點時首塊多半已在路上    ╟─ TTS 首幀
用戶續說 → 全部作廢 ──╜                  ║                          ║  (~80ms)
                                       ╙── primer 文本≠最終轉寫 → 作廢
```

端點時刻的關鍵路徑只剩: 取已完成的轉寫 (0ms) → 接管已開的 LLM 流 →
Kyutai 幀級出聲。理想命中下 FIRST AUDIO ≈ 300-600ms, 已進入人類
話輪交接區間 (Jacoby 2024: 類人對話要求在說話者結束前就開始理解與生成)。

## 8. Duplex 主動發話 (MiniCPM-o 式決策環)

無系統定時器 —— 時間以「幀」的形式存在於音頻流裡 (32ms/幀), 與
MiniCPM-o 的時分複用同構: 流式上下文按時間片進入模型, 模型每片自主
決定發聲或沉默。差別僅在: 級聯架構下時間片不是固定週期, 而是
**模型上一次自己選擇的 WAIT**。

每個檢查點, 模型收到一個感知塊 (PERCEPTION):

```
local time: Friday 23:41          ← 牆鐘 (深夜自然傾向沉默)
user last spoke: 47s ago          ← 雙方沉默時長
you last spoke: 32s ago
mic ambience: faint sounds — someone seems nearby   ← VAD 概率窗統計
unanswered proactive turns: 1     ← 已嘗試次數 (自然退避)
```

並三選一: `SAY=<一句話>` / `WAIT=<n 秒後再評估>` / `SLEEP` (直到用戶
開口)。所有間隔由模型逐次決定; 用戶開口、打斷、靜音(斷流)都會即時
作廢決策環。`max_consecutive` 是最後的硬保險。

## 9. 接縫處理 (前端)

- 抖動緩衝: 每輪首塊預留 100ms; 斷流恢復補 60ms + 3ms 淡入 (無爆音)
- 打斷淡出: flush 先 18ms 指數衰減總線增益再停源, barge-in 是
  「讓位」而非「掐斷」
- playback_done 去抖: 僅當服務端已宣告 assistant_done 且播放源排空
  才上報 —— 塊間空隙不再誤觸發字幕定格 / 狀態誤切

## 10. 識別起始點與準確率 (本輪修復)

起始點三根因與修法:
1. 採集端「丟樣本」降採樣無抗混疊 → 高頻折疊進語音帶。改為源窗口
   平均 (48k→16k 即 3 點 boxcar, 20kHz 混疊能量降至 24%, ~12dB)。
2. 單閾值 0.55 對輕聲起音偏高, 確認窗 250ms 而 pre-roll 僅 240ms。
   改為遲滯雙閾值 (onset 0.40 / release 0.35 / barge-in 仍按 0.55
   嚴格計時) + pre-roll 480ms; 段間不再清空 pre-roll。
3. whisper 對貼邊起始易丟首詞 → 段首墊 250ms / 段尾墊 150ms 靜音。

準確率 (顯存增量 +0.6GB):
- large-v3-turbo int8_float16 (~1.5GB), 詞錯率約為 small 的一半
- beam 3: 投機轉寫已把 ASR 移出關鍵路徑, beam 耗時被靜默期吸收
- initial_prompt 上下文偏置: 最近對話文本, 專名/話題詞顯著提升
- RMS 能量門限: 噪聲段不再幻聽出文字

## 11. 推理平台: llama.cpp (默認)

| | llama.cpp llama-server | vLLM | Ollama |
|---|---|---|---|
| Windows | 原生 ★ | WSL 限定 | 原生 |
| 顯存控制 | -ngl/-c 細粒度 ★ | 默認預佔 90% 做 KV | 粗粒度 |
| 單流小模型 TTFT | 最低 ★ | 無優勢 (吞吐型) | 多一層調度 |

實現為 OpenAI 兼容客戶端 (`OpenAICompatLLM`), llama.cpp 與 vLLM 共用 ——
想試 vLLM 只需把 `llm.base_url` 指向它。`chat_template_kwargs.
enable_thinking=false` 兩者均支持, 用於關閉 Qwen3 系思考段。

## 12. TTS: CosyVoice3 雙向流式 (當前默認)

Fun-CosyVoice3-0.5B-2512 (Apache 2.0): Qwen2-0.5B LM → 語音 token →
DiT 流匹配 (10-step) → HiFT 聲碼器 → 24kHz。官方 Bi-Streaming:
文本流入 + 音頻流出, 首包 ~150ms。

接入方式 (tts_cosyvoice.py):
- LLM 塊流 → asyncio 隊列 → 同步文本生成器 → inference_zero_shot
  的 tts_text 參數 (同一生成態邊收文本邊出音頻)
- 打斷: interrupt 置位 → 消費循環拋出 → generator.close() 關閉
  底層 LM/flow 生成釋放 GPU; 文本隊列以 None 哨兵雙保險解除阻塞
- 音色: voices/ref.wav + ref.txt 零樣本克隆 (prompt_text 格式為
  "<instruct><|endofprompt|><逐字稿>"), instruct 可附加方言/情緒指令

帶來的能力變化:
- 中文 (含 18 種方言) + 9 語言語音回覆解鎖; system prompt 已改為
  「跟隨用戶語言」
- 切句邏輯升級: CJK 句末標點 (。！？；) 無空格即時成立, 子句含
  ，、；：, 另有 30 字符無標點保險 —— 否則中文流永遠切不出首塊
- VRAM: ~2.5GB, 比 Kyutai 方案省 ~1GB (總預算 ~5.7/8GB)
- kyutai / csm 後端保留, config 一行切換, 加載失敗自動回退
