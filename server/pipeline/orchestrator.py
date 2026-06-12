"""全雙工對話調度器 (orchestrator)。

設計要點 (對齊 Sesame demo 的交互體驗):
  - 麥克風永不關閉: 所有狀態下都跑 VAD
  - 流水線並行: LLM 逐句產出 → 隊列 → CSM 逐句合成 → 即時下發;
    第一句在播放時, 第二句已在生成 (overlap 掩蓋生成延遲)
  - barge-in: AI 說話期間用戶連續發聲超過閾值 → 三路同時剎車:
      1) cancel LLM/TTS asyncio 任務
      2) threading.Event 讓 CSM 在 token 級停止解碼 (秒級→毫秒級)
      3) 通知前端清空播放隊列
  - 打斷後的用戶語音不丟失: VAD 段繼續收集, 結束後正常轉寫回應

狀態機:  LISTENING ──(轉寫完成)──▶ THINKING ──(首句音頻)──▶ SPEAKING
            ▲                                              │
            └────────(播放完成 / barge-in 打斷)─────────────┘
"""
from __future__ import annotations

import asyncio
import collections
import logging
import re
import threading
import time

import numpy as np

from .vad import StreamingVAD, VadEvent

log = logging.getLogger("orchestrator")


class Session:
    def __init__(self, cfg, asr, llm, tts, send_json, send_bytes):
        self.cfg = cfg
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.send_json = send_json
        self.send_bytes = send_bytes

        self.vad = StreamingVAD(
            threshold=cfg.vad.threshold,
            min_speech_ms=cfg.vad.min_speech_ms,
            min_silence_ms=cfg.vad.min_silence_ms,
            pre_roll_ms=cfg.vad.pre_roll_ms,
            fast_min_silence_ms=getattr(
                cfg.vad, "fast_min_silence_ms", None),
            fast_endpoint_after_ms=getattr(
                cfg.vad, "fast_endpoint_after_ms", 900),
            speculate_after_ms=getattr(cfg.vad, "speculate_after_ms", 180),
            onset_threshold=getattr(cfg.vad, "onset_threshold", None),
            release_threshold=getattr(cfg.vad, "release_threshold", None),
        )
        self.state = "listening"
        self.history: list[dict] = []
        self._respond_task: asyncio.Task | None = None
        self._tts_interrupt = threading.Event()
        self._client_playing = False

        # ---- 主動發話 (proactive): 時鐘 = 音頻流本身 ----
        # 不用 asyncio 計時器。麥克風以 ~31.25 幀/秒 (32ms/幀) 連續推流,
        # 數「靜音幀」即是計時 —— 與 Moshi 等全雙工模型的幀時鐘同構。
        self.FPS = 1000 / 32                     # VAD 幀率
        self._silent_frames = 0                  # 連續靜默幀數
        self._quiet_frames = 0                   # 打斷後冷靜期 (幀)
        self._proactive_count = 0                # 連續主動發話次數 (用戶回話即歸零)

        # ---- 模型自主排程 (mode: model_scheduled) ----
        # LLM 在每輪回覆後自行決定「沉默 N 秒後說 X」; X 預先合成,
        # 流時鐘到點直接播緩存 → 主動發話零延遲, 時間決定權在模型
        self._plan_task: asyncio.Task | None = None
        self._plan_interrupt = threading.Event()
        self._scheduled: dict | None = None      # {frames, text, pcm}

        # ---- Duplex 決策環 (mode: duplex, MiniCPM-o 式) ----
        # 模型在幀時鐘上自主選擇「下一個決策檢查點」; 到點時讀取
        # 真實上下文 (牆鐘時刻 / 雙方沉默時長 / 環境聲存在感 / 已嘗試
        # 次數) 並三選一: SAY=說這句 / WAIT=再等 n 秒 / SLEEP=直到用戶
        # 開口前不再主動。沒有任何固定時間表 —— 間隔由模型逐次決定。
        self._next_check_frames: int | None = None   # 模型選定的檢查點
        self._decide_task: asyncio.Task | None = None
        self._duplex_sleep = False                   # SLEEP: 等用戶開口
        self._ambient = collections.deque(            # ~12s 概率窗
            maxlen=int(12 * self.FPS))
        self._last_user_t: float | None = None       # 用戶上次說話 (monotonic)
        self._last_ai_t: float | None = None         # AI 上次說話

        # ---- 投機 LLM 預啟動 (speculative prefill) ----
        # 投機 ASR 完成後, 端點到來前的剩餘靜默是純等待 —— 把 LLM 流
        # 也投機啟動, 首塊在端點時刻多半已就緒 (LLM 移出關鍵路徑)。
        # 用戶續說 → 與投機轉寫一同作廢。
        self._primer: dict | None = None             # {text, queue, task}

        # ---- 投機轉寫 + 語義端點 (延遲主軸優化) ----
        # 句中靜音 ~180ms 即對「語音至此」做投機 ASR; 端點到來時轉寫
        # 多半已完成 → ASR 移出關鍵路徑。若投機文本以句末標點收尾,
        # 直接 force_end 提前端點 (語義端點), 再省 ~200ms。
        self._spec_task: asyncio.Task | None = None
        self._spec_text: str | None = None
        self._spec_len = 0                       # 投機段樣本數 (校驗用)

    # ================= 入口: 上行音頻與控制消息 =================
    async def on_audio(self, audio_16k: np.ndarray) -> None:
        for res in self.vad.process(audio_16k):
            self._ambient.append(res.prob)
            await self._tick_silence(res)   # 主動發話的時鐘: 逐幀推進
            if res.is_voiced:
                self._invalidate_speculation()   # 又開口, 投機作廢
                if self._decide_task is not None:
                    self._decide_task.cancel()   # 用戶開口, 決策作廢
                    self._decide_task = None
            elif res.speculative_segment is not None:
                self._start_speculation(res.speculative_segment)
            # ---- barge-in: AI 思考/說話期間檢測到持續用戶語音 ----
            if self.state in ("thinking", "speaking"):
                if res.voiced_ms >= self.cfg.vad.barge_in_speech_ms:
                    await self._interrupt(reason="barge_in")
            if res.event is VadEvent.SPEECH_START:
                await self.send_json({"type": "vad", "speaking": True})
            elif res.event is VadEvent.SPEECH_END:
                await self.send_json({"type": "vad", "speaking": False})
                asyncio.create_task(self._handle_utterance(res.segment))

    async def on_control(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "playback_done":
            self._client_playing = False
            self._silent_frames = 0
            if self.state == "speaking" and self._respond_task is None:
                await self._set_state("listening")
        elif t == "text_input":
            text = (msg.get("text") or "").strip()
            if text:
                await self._handle_text(text)
        elif t == "reset":
            await self._interrupt(reason="reset")
            self.history.clear()
            self.tts.reset_context()
            await self.send_json({"type": "history_cleared"})

    async def greet(self) -> None:
        if not self.cfg.conversation.greet_on_connect:
            return
        text = self.cfg.conversation.greeting
        self._tts_interrupt.clear()
        await self._set_state("speaking")
        got_audio = False
        async for pcm in self.tts.synthesize_stream(
                text, 0, self._tts_interrupt):
            if not got_audio:
                got_audio = True
                await self.send_json(
                    {"type": "assistant_sentence", "text": text})
            await self._send_audio(pcm)
        if got_audio:
            self.history.append({"role": "assistant", "content": text})
            await self.send_json({"type": "assistant_done"})
            self._schedule_plan()
        # 等客戶端 playback_done 再回 listening; 兜底直接置回
        if not self._client_playing:
            await self._set_state("listening")

    # ============ 投機轉寫 + 語義端點 (延遲主軸) ============
    def _start_speculation(self, segment: np.ndarray) -> None:
        pcfg = getattr(self.cfg, "pipeline", None)
        if pcfg is not None and not getattr(pcfg, "speculative_asr", True):
            return
        self._invalidate_speculation()
        self._spec_len = len(segment)
        self._spec_task = asyncio.create_task(self._speculate(segment))

    def _invalidate_speculation(self) -> None:
        if self._spec_task is not None:
            self._spec_task.cancel()
            self._spec_task = None
        self._spec_text = None
        self._spec_len = 0
        self._cancel_primer()

    # ---------- 投機 LLM 預啟動 ----------
    def _cancel_primer(self) -> None:
        primer, self._primer = self._primer, None
        if primer is not None and primer["task"] is not None:
            primer["task"].cancel()

    def _start_primer(self, text: str) -> None:
        """端點前提前打開 LLM 流, 首塊緩存進隊列等 _respond 接管。"""
        pcfg = getattr(self.cfg, "pipeline", None)
        if pcfg is not None and not getattr(pcfg, "speculative_llm", True):
            return
        if self._respond_task is not None:       # 正有回應在跑, 不投機
            return
        self._cancel_primer()
        q: asyncio.Queue = asyncio.Queue()
        msgs = [*self.history, {"role": "user", "content": text}]
        max_w, min_w = self._first_chunk_limits()

        async def run() -> None:
            try:
                async for chunk in self.llm.stream_speakable_chunks(
                        msgs, max_w, min_w):
                    await q.put(chunk)
                await q.put(None)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("primer failed: %s", e)
                await q.put(None)

        self._primer = {"text": text, "queue": q,
                        "task": asyncio.create_task(run()),
                        "t0": time.monotonic()}
        log.info("[lat] llm primer started (pre-endpoint)")

    def _take_primer(self, text: str) -> dict | None:
        """端點時取走匹配的 primer; 文本不一致則作廢。"""
        primer, self._primer = self._primer, None
        if primer is None:
            return None
        if primer["text"] != text:
            if primer["task"] is not None:
                primer["task"].cancel()
            return None
        log.info("[lat] llm primer hit (started %.0fms before endpoint)",
                 (time.monotonic() - primer["t0"]) * 1000)
        return primer

    def _first_chunk_limits(self) -> tuple[int, int]:
        pcfg = getattr(self.cfg, "pipeline", None)
        max_w = getattr(pcfg, "first_chunk_max_words", 9) if pcfg else 9
        min_w = getattr(pcfg, "first_chunk_min_words", 2) if pcfg else 2
        return max_w, min_w

    async def _speculate(self, segment: np.ndarray) -> None:
        """句中靜音時對「語音至此」做轉寫; 投機期間若用戶續說則被作廢。
        投機文本以句末標點收尾 → 語義端點: 不等滿額靜音, 直接收束本段。
        """
        try:
            text = await self.asr.transcribe(segment)
            self._spec_text = text
            log.info("[lat] spec_asr done (%.1fs audio): %r",
                     len(segment) / 16000, text[:50])
            if text and len(text.strip()) >= 2:
                self._start_primer(text)   # LLM 同步移出關鍵路徑
            pcfg = getattr(self.cfg, "pipeline", None)
            semantic = (getattr(pcfg, "semantic_endpoint", True)
                        if pcfg else True)
            if (semantic and text
                    and text.rstrip()[-1:] in ".!?。！？"):
                # 句子在語義上已完整 → 提前端點 (省掉剩餘靜音等待)
                seg = self.vad.force_end()
                if seg is not None:
                    log.info("[lat] semantic endpoint fired")
                    await self.send_json({"type": "vad", "speaking": False})
                    asyncio.create_task(self._handle_utterance(seg))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("speculative asr failed: %s", e)
            self._spec_text = None

    async def _take_transcript(self, segment: np.ndarray) -> str:
        """端點時取轉寫: 投機命中 → 0ms; 投機進行中 → 等它; 否則現轉。
        命中校驗: 端點段的前綴樣本數與投機段一致 (期間僅靜音幀)。"""
        task, self._spec_task = self._spec_task, None
        text, spec_len = self._spec_text, self._spec_len
        self._spec_text, self._spec_len = None, 0
        valid = spec_len > 0 and len(segment) >= spec_len
        if valid and task is not None and not task.done():
            await asyncio.wait({task})           # 投機進行中, 等收尾
            if text is None:
                text, self._spec_text = self._spec_text, None
        if valid and text:
            log.info("[lat] asr=0ms (speculative hit)")
            return text
        if task is not None:
            task.cancel()
        return await self.asr.transcribe(segment)

    async def _handle_text(self, text: str) -> None:
        """文字輸入: 開會/夜深不便說話時用打字, 回應仍走語音。"""
        log.info("USER (text): %s", text)
        t0 = time.monotonic()
        self._last_user_t = t0
        self._duplex_sleep = False
        self._proactive_count = 0
        self._silent_frames = 0
        self._cancel_plan()
        self._invalidate_speculation()
        await self.send_json({"type": "user_transcript", "text": text})
        await self._interrupt(reason="new_utterance", notify=False)
        self.history.append({"role": "user", "content": text})
        self._trim_history()
        self._tts_interrupt = threading.Event()
        self._respond_task = asyncio.create_task(self._respond(t0))

    # ================= 一句用戶話 → 一輪回應 =================
    async def _handle_utterance(self, segment: np.ndarray) -> None:
        import time
        t0 = time.monotonic()          # 語音端點時刻 = 延遲計時起點
        dur = len(segment) / 16000
        if dur < 0.3:
            return
        text = await self._take_transcript(segment)
        t_asr = time.monotonic()
        if not text or len(text.strip()) < 2:
            return
        log.info("USER (%.1fs): %s", dur, text)
        log.info("[lat] asr=%.0fms", (t_asr - t0) * 1000)
        primer = self._take_primer(text)   # 投機 LLM 命中 → 首塊已在路上
        self._last_user_t = time.monotonic()
        self._duplex_sleep = False         # 用戶開口, 解除 SLEEP
        self._proactive_count = 0          # 用戶回話, 主動退避歸零
        self._silent_frames = 0
        self._cancel_plan()                # 沉默前提失效, 排程作廢
        await self.send_json({"type": "user_transcript", "text": text})

        # 同一時間只允許一輪回應; 新話語覆蓋舊回應 (自然的搶話語義)
        await self._interrupt(reason="new_utterance", notify=False)

        self.history.append({"role": "user", "content": text})
        self._trim_history()
        if hasattr(self.asr, "set_context"):
            self.asr.set_context(self.history)   # 下一輪識別的上下文偏置
        self.tts.add_user_context(text, segment)

        self._tts_interrupt = threading.Event()
        self._respond_task = asyncio.create_task(self._respond(t0, primed=primer))

    async def _respond(self, t0: float | None = None,
                       messages: list[dict] | None = None,
                       proactive: bool = False,
                       primed: dict | None = None) -> None:
        spoken: list[str] = []
        interrupt = self._tts_interrupt
        msgs = messages if messages is not None else self.history
        try:
            if not proactive:
                await self._set_state("thinking")
            # 主動模式: 不顯示「思考」, LLM 先靜默決定要不要開口 (PASS = 沉默)
            queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=4)
            pcfg = getattr(self.cfg, "pipeline", None)
            max_w, min_w = self._first_chunk_limits()
            first_history = (
                getattr(pcfg, "first_chunk_history_context", False)
                if pcfg else False
            )
            vcfg = getattr(self.cfg.tts, "voice_prompt", None)
            voice_every_chunk = (
                getattr(vcfg, "every_chunk", True) if vcfg else True
            )

            async def chunk_source():
                """LLM 塊流; 投機 primer 命中時直接消費已開的流;
                主動模式下首塊若為 PASS 則整輪靜默。"""
                first_chunk = True

                async def raw():
                    if primed is not None:
                        pq = primed["queue"]
                        while True:
                            c = await pq.get()
                            if c is None:
                                return
                            yield c
                    else:
                        async for c in self.llm.stream_speakable_chunks(
                                msgs, max_w, min_w):
                            yield c

                async for chunk in raw():
                    chunk = chunk.strip().strip('"\'')
                    if not chunk:
                        continue
                    if proactive and first_chunk:
                        head = chunk.strip().strip('"\'.,!').upper()
                        if head == "PASS" or head.startswith("PASS"):
                            log.info("proactive: LLM 選擇沉默 (PASS)")
                            return
                    first_chunk = False
                    yield chunk

            if (getattr(pcfg, "tts_stream_input", False)
                    and hasattr(self.tts, "synthesize_text_stream")):
                first_audio = True
                logged_first_token = False

                async def token_stream():
                    nonlocal logged_first_token
                    async for token in chunk_source():
                        if not logged_first_token:
                            logged_first_token = True
                            if t0 is not None:
                                log.info(
                                    "[lat] llm_first_chunk=%.0fms (%r)",
                                    (time.monotonic() - t0) * 1000, token)
                        await self.send_json(
                            {"type": "assistant_delta",
                             "text": token,
                             "proactive": proactive})
                        yield token + " "

                async for pcm, final_text in self.tts.synthesize_text_stream(
                        token_stream(), 0, interrupt):
                    if interrupt.is_set():
                        break
                    if final_text is not None:
                        if final_text.strip():
                            spoken.append(final_text.strip())
                        continue
                    if pcm.size == 0:
                        continue
                    if first_audio:
                        first_audio = False
                        if t0 is not None:
                            ms = (time.monotonic() - t0) * 1000
                            log.info(
                                "[lat] FIRST AUDIO=%.0fms (stream-in)", ms)
                            await self.send_json(
                                {"type": "turn_latency", "ms": int(ms)})
                        await self._set_state("speaking")
                    await self._send_audio(pcm)
                await self.send_json({"type": "assistant_done"})
                return

            async def produce() -> None:
                try:
                    async for chunk in chunk_source():
                        await self.send_json(
                            {"type": "assistant_delta",
                             "text": chunk,
                             "proactive": proactive})
                        await queue.put(chunk)
                finally:
                    await queue.put(None)

            producer = asyncio.create_task(produce())
            try:
                first = True
                logged_llm_first = False
                if getattr(pcfg, "instant_ack_enabled", False) and not proactive:
                    ack = getattr(pcfg, "instant_ack_text", "Okay.").strip()
                    if ack:
                        got_ack_audio = False
                        cached_ack = getattr(pcfg, "instant_ack_pcm", None)
                        if cached_ack is not None and not interrupt.is_set():
                            got_ack_audio = True
                            if t0 is not None:
                                log.info(
                                    "[lat] FIRST AUDIO=%.0fms (cached ack)",
                                    (time.monotonic() - t0) * 1000)
                            await self._set_state("speaking")
                            first = False
                            await self.send_json(
                                {"type": "assistant_delta", "text": ack})
                            await self._send_audio(cached_ack)
                        else:
                            async for pcm in self.tts.synthesize_stream(
                                    ack, 0, interrupt):
                                if interrupt.is_set():
                                    break
                                if not got_ack_audio:
                                    got_ack_audio = True
                                    if t0 is not None:
                                        log.info(
                                            "[lat] FIRST AUDIO=%.0fms (instant ack)",
                                            (time.monotonic() - t0) * 1000)
                                    await self._set_state("speaking")
                                    first = False
                                    await self.send_json(
                                        {"type": "assistant_delta",
                                         "text": ack})
                                await self._send_audio(pcm)
                        if got_ack_audio:
                            spoken.append(ack)
                while True:
                    sent = await queue.get()
                    if sent is None:
                        break
                    if not logged_llm_first and t0 is not None:
                        logged_llm_first = True
                        log.info("[lat] llm_first_chunk=%.0fms (%r)",
                                 (time.monotonic() - t0) * 1000, sent)
                    kwargs = {}
                    if getattr(self.tts, "supports_csm_context", False):
                        kwargs = dict(
                            use_voice_prompt=(first or voice_every_chunk),
                            use_history_context=(
                                first_history if first else True),
                        )
                    got_audio = False
                    async for pcm in self.tts.synthesize_stream(
                            sent, 0, interrupt, **kwargs):
                        if interrupt.is_set():
                            break
                        if not got_audio:
                            got_audio = True
                            if first:
                                if t0 is not None:
                                    ms = (time.monotonic() - t0) * 1000
                                    log.info(
                                        "[lat] FIRST AUDIO=%.0fms (端點→開播)",
                                        ms)
                                    await self.send_json(
                                        {"type": "turn_latency",
                                         "ms": int(ms)})
                                await self._set_state("speaking")
                                first = False
                        await self._send_audio(pcm)
                    if interrupt.is_set():
                        break
                    if got_audio:
                        spoken.append(sent)
            finally:
                producer.cancel()

            await self.send_json({"type": "assistant_done"})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("respond failed")
            await self.send_json({"type": "error", "message": str(e)})
        finally:
            if primed is not None and primed["task"] is not None:
                primed["task"].cancel()
            if spoken:
                content = " ".join(spoken)
                self._last_ai_t = time.monotonic()
                if proactive:
                    self._proactive_count += 1
                    log.info("proactive: 第 %d 次主動發話: %s",
                             self._proactive_count, content)
                self.history.append(
                    {"role": "assistant", "content": content})
                self._trim_history()
            self._silent_frames = 0   # 回應結束後從零開始累計靜默
            self._respond_task = None
            if not self._client_playing and self.state != "listening":
                await self._set_state("listening")
            if spoken and not interrupt.is_set():
                self._schedule_plan()     # 模型自主排程下一句 (mode A)

    # ================= 主動發話 (proactive speech) =================
    DEFAULT_NUDGE = (
        "[SYSTEM NOTE — not spoken by the user] The user has been silent "
        "for about {silence_s} seconds. Based on the conversation so far, "
        "decide whether to speak up naturally — e.g. follow up on the "
        "topic, share a brief related thought, gently check in, or ask a "
        "light question — in your own voice, the way a friend breaks a "
        "lull, never like an assistant checking in. ONE short sentence, "
        "low-pressure. If the conversation has reached a "
        "natural close, the user said goodbye, or silence feels more "
        "appropriate, reply with exactly: PASS"
    )

    async def _tick_silence(self, res) -> None:
        """主動發話的「時鐘」: 由音頻流逐幀驅動, 無任何系統計時器。

        原理與 Moshi 等原生全雙工模型同構 —— 模型每 80ms 必然收到
        一幀 (含靜音幀), 時間以幀數的形式存在於流中。這裡同樣:
        麥克風每 32ms 產生一個 VAD 幀, 數靜音幀 = 計時。
        麥克風斷流 → 時鐘自然停擺, 不會對著空氣自言自語。
        """
        pcfg = getattr(self.cfg, "proactive", None)
        if not pcfg or not getattr(pcfg, "enabled", False):
            return
        if self._quiet_frames > 0:               # 打斷後的冷靜期, 逐幀消耗
            self._quiet_frames -= 1
            return
        if res.is_voiced:
            self._silent_frames = 0
            return
        # 只有「真正的對話空檔」才計時: 聆聽態、無播放、無進行中回應
        if (self.state != "listening" or self._client_playing
                or self._respond_task is not None):
            self._silent_frames = 0
            return
        self._silent_frames += 1

        max_consec = int(getattr(pcfg, "max_consecutive", 3))
        if self._proactive_count >= max_consec:
            return                               # 退避用盡, 安靜等用戶

        mode = getattr(pcfg, "mode", "nudge")

        # ---- 模式 duplex: MiniCPM-o 式決策環 —— 模型自選檢查點 ----
        # 幀時鐘推進到「模型上次自己選定的檢查時刻」→ 喚起一次決策;
        # 決策讀取真實上下文+時間信息, 輸出 SAY / WAIT / SLEEP。
        # 沒有固定時間表: 間隔由模型逐次決定, SLEEP 後時鐘休眠直到用戶開口。
        if mode == "duplex":
            if self._duplex_sleep or self._decide_task is not None:
                return
            if self._next_check_frames is None:
                # 回合剛結束尚未播種 (或冷啟動): 用 min_wait 播種首個檢查點
                min_w = int(getattr(pcfg, "min_wait_s", 6))
                self._next_check_frames = int(min_w * self.FPS)
            if self._silent_frames < self._next_check_frames:
                return
            self._silent_frames = 0
            self._next_check_frames = None
            self._decide_task = asyncio.create_task(self._duplex_decide())
            return

        # ---- 模式 A: model_scheduled — 模型已自主排程, 到點播緩存 ----
        if mode == "model_scheduled":
            plan = self._scheduled
            if plan is None or plan.get("pcm") is None:
                return                           # 模型決定不說 / 還在合成
            if self._silent_frames < plan["frames"]:
                return
            self._scheduled = None
            self._silent_frames = 0
            self._tts_interrupt = threading.Event()
            self._respond_task = asyncio.create_task(
                self._speak_scheduled(plan))
            return

        # ---- 模式 B: nudge — 到點即時詢問 LLM (PASS = 沉默) ----
        delays = list(getattr(pcfg, "delays_s", [12, 28, 60]))
        idx = min(self._proactive_count, len(delays) - 1)
        if self._silent_frames < int(delays[idx] * self.FPS):
            return
        silence_s = int(self._silent_frames / self.FPS)
        self._silent_frames = 0                  # 觸發即清零 (PASS 也重新累計)
        await self._start_proactive(silence_s)

    # ---------- 模式 duplex: MiniCPM-o 式決策環 ----------
    # 每個檢查點, 模型看到一個「感知塊」—— 與 MiniCPM-o 的時分複用思路
    # 同構: 流式上下文按時間片進入模型, 模型每片自主決定發聲或保持沉默。
    # 這裡的時間片不是固定週期, 而是模型上一次自己選擇的 WAIT。
    DEFAULT_DUPLEX_PROMPT = (
        "[PERCEPTION — not spoken by the user]\n"
        "local time: {clock}\n"
        "user last spoke: {user_s}s ago\n"
        "you last spoke: {ai_s}s ago\n"
        "mic ambience: {ambience}\n"
        "unanswered proactive turns: {attempts}\n\n"
        "You are in a live call and the line is quiet. Decide for yourself "
        "whether this silence wants company or space. Consider the hour "
        "(late night usually wants space), the ambience (a fully dead mic "
        "may mean they stepped away), how the last exchange ended, and how "
        "many times you've already spoken into silence.\n"
        "Reply with EXACTLY one line, one of:\n"
        "SAY=<one short natural sentence in your own voice — the way a "
        "friend breaks a lull, never an assistant checking in>\n"
        "WAIT=<seconds, {min_wait}-{max_wait}, your own choice of when to "
        "reconsider>\n"
        "SLEEP   (stay quiet until they speak again)"
    )
    _SAY_RE = re.compile(r"SAY\s*=\s*(.+)", re.S | re.I)
    _WAIT_RE = re.compile(r"WAIT\s*=\s*(\d+)", re.I)

    def _ambience_summary(self) -> str:
        """環境聲存在感: VAD 概率窗的分佈 → 模型可讀的一句話。"""
        if not self._ambient:
            return "no signal"
        arr = np.fromiter(self._ambient, dtype=np.float32)
        faint = float(np.mean((arr > 0.12) & (arr < self.cfg.vad.threshold)))
        if faint > 0.25:
            return "faint sounds — someone seems nearby"
        if faint > 0.06:
            return "occasional rustle"
        return "completely silent"

    async def _duplex_decide(self) -> None:
        pcfg = self.cfg.proactive
        try:
            now = time.monotonic()
            user_s = int(now - self._last_user_t) if self._last_user_t else -1
            ai_s = int(now - self._last_ai_t) if self._last_ai_t else -1
            min_w = int(getattr(pcfg, "min_wait_s", 6))
            max_w = int(getattr(pcfg, "max_wait_s", 120))
            tmpl = getattr(pcfg, "duplex_prompt", None) or self.DEFAULT_DUPLEX_PROMPT
            prompt = (tmpl
                      .replace("{clock}", time.strftime("%A %H:%M"))
                      .replace("{user_s}", str(user_s) if user_s >= 0 else "never")
                      .replace("{ai_s}", str(ai_s) if ai_s >= 0 else "never")
                      .replace("{ambience}", self._ambience_summary())
                      .replace("{attempts}", str(self._proactive_count))
                      .replace("{min_wait}", str(min_w))
                      .replace("{max_wait}", str(max_w)))
            raw = (await self.llm.decide_proactive(self.history, prompt)).strip()
            first = raw.splitlines()[0].strip() if raw else ""
            log.info("duplex: 決策 → %r", first[:80])

            m = self._SAY_RE.search(first)
            if m:
                text = m.group(1).strip().strip('"\'')
                if text:
                    # 仍處於對話空檔才開口 (決策期間用戶可能已說話)
                    if (self.state == "listening"
                            and self._respond_task is None
                            and not self._client_playing):
                        self._tts_interrupt = threading.Event()
                        self._respond_task = asyncio.create_task(
                            self._speak_text(text))
                    return
            m = self._WAIT_RE.search(first)
            if m:
                wait = max(min_w, min(max_w, int(m.group(1))))
                log.info("duplex: 模型選擇 %ds 後再評估", wait)
                self._next_check_frames = int(wait * self.FPS)
                return
            # SLEEP 或無法解析 → 安靜等用戶
            log.info("duplex: 模型選擇沉默 (SLEEP)")
            self._duplex_sleep = True
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("duplex decide failed: %s", e)
            self._next_check_frames = int(
                getattr(pcfg, "max_wait_s", 120) * self.FPS / 2)
        finally:
            self._decide_task = None

    async def _speak_text(self, text: str) -> None:
        """直接把一句既定文本流式合成播出 (duplex SAY 路徑)。"""
        interrupt = self._tts_interrupt
        got = False
        try:
            async for pcm in self.tts.synthesize_stream(text, 0, interrupt):
                if interrupt.is_set():
                    break
                if not got:
                    got = True
                    await self._set_state("speaking")
                    await self.send_json({"type": "assistant_sentence",
                                          "text": text, "proactive": True})
                await self._send_audio(pcm)
            if got and not interrupt.is_set():
                await self.send_json({"type": "assistant_done"})
                self.history.append({"role": "assistant", "content": text})
                self._trim_history()
                self._proactive_count += 1
                self._last_ai_t = time.monotonic()
                log.info("duplex: 第 %d 次主動發話: %s",
                         self._proactive_count, text)
        finally:
            self._respond_task = None
            self._silent_frames = 0
            if not self._client_playing and self.state != "listening":
                await self._set_state("listening")
            self._schedule_plan()              # 播種下一個檢查點

    async def _start_proactive(self, silence_s: int) -> None:
        pcfg = self.cfg.proactive
        tmpl = getattr(pcfg, "nudge_prompt", None) or self.DEFAULT_NUDGE
        nudge = tmpl.replace("{silence_s}", str(silence_s))
        # nudge 只進這次請求, 不寫入對話歷史
        msgs = [*self.history, {"role": "user", "content": nudge}]
        log.info("proactive: 靜默 %ds (流時鐘), 觸發第 %d 輪主動決策",
                 silence_s, self._proactive_count + 1)
        self._tts_interrupt = threading.Event()
        self._respond_task = asyncio.create_task(
            self._respond(None, messages=msgs, proactive=True))

    # ---------- 模式 A: 模型自主排程 ----------
    DEFAULT_PLAN_PROMPT = (
        "[SYSTEM NOTE — not spoken by the user] You just finished replying. "
        "If the user stays silent, would you naturally say something to "
        "continue the conversation? Decide for yourself BOTH whether to "
        "speak and how long to wait. Reply in EXACTLY one of these two "
        "formats and nothing else:\n"
        "NONE\n"
        "WAIT=<seconds> | <one short sentence in your own voice — the way "
        "a friend breaks a lull, never an assistant checking in>\n"
        "Choose WAIT between {min_wait}-{max_wait} seconds based on context "
        "(short if mid-task or a question is pending, long if the user "
        "likely needs time to think). Prefer NONE if the conversation "
        "reached a natural close or the user said goodbye."
    )
    _PLAN_RE = re.compile(r"WAIT\s*=\s*(\d+)\s*\|\s*(.+)", re.S)

    def _schedule_plan(self) -> None:
        """回覆結束後觸發: 按模式播種下一步主動性。"""
        pcfg = getattr(self.cfg, "proactive", None)
        if not pcfg or not getattr(pcfg, "enabled", False):
            return
        if self._proactive_count >= int(getattr(pcfg, "max_consecutive", 3)):
            return
        mode = getattr(pcfg, "mode", "nudge")
        if mode == "duplex":
            # 不立刻問模型 —— 只播種首個檢查點 (min_wait), 屆時模型
            # 帶著「沉默已持續多久」等真實感知再做決策, 更貼近時分複用。
            if not self._duplex_sleep:
                min_w = int(getattr(pcfg, "min_wait_s", 6))
                self._next_check_frames = int(min_w * self.FPS)
            return
        if mode != "model_scheduled":
            return
        self._cancel_plan()
        self._plan_interrupt = threading.Event()
        self._plan_task = asyncio.create_task(self._make_plan())

    def _cancel_plan(self) -> None:
        """用戶開口 / 打斷 / 重置 → 計劃作廢 (它以「持續沉默」為前提)。"""
        self._plan_interrupt.set()               # 中止進行中的預合成
        if self._plan_task is not None:
            self._plan_task.cancel()
            self._plan_task = None
        self._scheduled = None
        if self._decide_task is not None:        # duplex: 決策作廢
            self._decide_task.cancel()
            self._decide_task = None
        self._next_check_frames = None

    async def _make_plan(self) -> None:
        pcfg = self.cfg.proactive
        try:
            min_w = int(getattr(pcfg, "min_wait_s", 6))
            max_w = int(getattr(pcfg, "max_wait_s", 120))
            tmpl = getattr(pcfg, "plan_prompt", None) or self.DEFAULT_PLAN_PROMPT
            prompt = (tmpl.replace("{min_wait}", str(min_w))
                          .replace("{max_wait}", str(max_w)))
            raw = await self.llm.plan_followup(self.history, prompt)
            m = self._PLAN_RE.search(raw)
            if not m or raw.strip().upper().startswith("NONE"):
                log.info("plan: 模型決定不排程 (%r)", raw[:60])
                return
            wait_s = max(min_w, min(max_w, int(m.group(1))))
            text = m.group(2).strip().splitlines()[0].strip()
            if not text:
                return
            log.info("plan: 模型自主排程 — %ds 後說: %s", wait_s, text)
            # 預合成 (GPU 此刻空閒); 用戶開口會經 _cancel_plan 即時中止
            chunks = []
            async for pcm in self.tts.synthesize_stream(
                    text, 0, self._plan_interrupt):
                chunks.append(pcm)
            if self._plan_interrupt.is_set() or not chunks:
                return
            self._scheduled = {
                "frames": int(wait_s * self.FPS),
                "text": text,
                "pcm": np.concatenate(chunks),
            }
            log.info("plan: 預合成完成 (%.1fs 音頻), 等待流時鐘",
                     len(self._scheduled["pcm"]) / 24000)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("plan: 排程失敗 (回退安靜): %s", e)
        finally:
            self._plan_task = None

    async def _speak_scheduled(self, plan: dict) -> None:
        """流時鐘到點: 直接播預合成緩存, 零生成延遲。可被 barge-in 秒停。"""
        try:
            log.info("proactive: 播放模型排程語句 (零延遲): %s", plan["text"])
            await self._set_state("speaking")
            await self.send_json({"type": "assistant_sentence",
                                  "text": plan["text"], "proactive": True})
            await self._send_audio(plan["pcm"])
            await self.send_json({"type": "assistant_done"})
            self.history.append(
                {"role": "assistant", "content": plan["text"]})
            self._trim_history()
            self._proactive_count += 1
        finally:
            self._respond_task = None
            self._silent_frames = 0
            if not self._client_playing and self.state != "listening":
                await self._set_state("listening")
            self._schedule_plan()                # 鏈式: 規劃下一句

    # ================= 打斷 =================
    async def _interrupt(self, reason: str, notify: bool = True) -> None:
        task, self._respond_task = self._respond_task, None
        if task is None and self.state == "listening":
            return
        log.info("INTERRUPT (%s)", reason)
        if reason in ("barge_in", "reset"):
            # 用戶打斷 → 主動發話進入冷靜期, 避免顯得糾纏
            cooldown = getattr(getattr(self.cfg, "proactive", None),
                               "cooldown_after_interrupt_s", 20)
            self._quiet_frames = int(float(cooldown) * self.FPS)
            self._silent_frames = 0
        self._cancel_plan()
        self._tts_interrupt.set()          # CSM token 級停止
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            # 記錄被打斷的部分回應, 保持對話事實一致
            if self.history and self.history[-1]["role"] == "user":
                pass  # 部分文本已在 _respond finally 中寫入
        if notify:
            await self.send_json({"type": "interrupt"})  # 前端清空播放隊列
        self._client_playing = False
        await self._set_state("listening")

    # ================= 工具 =================
    async def _send_audio(self, wav_24k: np.ndarray) -> None:
        from ..utils.audio import float32_to_pcm16
        pcm = float32_to_pcm16(wav_24k)
        self._client_playing = True
        chunk = 24000 * 2 // 5  # 200ms / 塊
        for i in range(0, len(pcm), chunk):
            if self._tts_interrupt.is_set():
                return
            await self.send_bytes(pcm[i:i + chunk])

    async def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            await self.send_json({"type": "state", "state": state})

    def _trim_history(self) -> None:
        limit = self.cfg.conversation.max_history_turns * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]
