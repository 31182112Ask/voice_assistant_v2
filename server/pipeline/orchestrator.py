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
import logging
import threading

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
        )
        self.state = "listening"
        self.history: list[dict] = []
        self._respond_task: asyncio.Task | None = None
        self._tts_interrupt = threading.Event()
        self._client_playing = False

    # ================= 入口: 上行音頻與控制消息 =================
    async def on_audio(self, audio_16k: np.ndarray) -> None:
        for res in self.vad.process(audio_16k):
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
            if self.state == "speaking" and self._respond_task is None:
                await self._set_state("listening")
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
        wav = await self.tts.synthesize(text, 0, self._tts_interrupt)
        if wav is not None:
            await self.send_json({"type": "assistant_sentence", "text": text})
            await self._send_audio(wav)
            self.history.append({"role": "assistant", "content": text})
            await self.send_json({"type": "assistant_done"})
        # 等客戶端 playback_done 再回 listening; 兜底直接置回
        if not self._client_playing:
            await self._set_state("listening")

    # ================= 一句用戶話 → 一輪回應 =================
    async def _handle_utterance(self, segment: np.ndarray) -> None:
        import time
        t0 = time.monotonic()          # 語音端點時刻 = 延遲計時起點
        dur = len(segment) / 16000
        if dur < 0.3:
            return
        text = await self.asr.transcribe(segment)
        t_asr = time.monotonic()
        if not text or len(text.strip()) < 2:
            return
        log.info("USER (%.1fs): %s", dur, text)
        log.info("[lat] asr=%.0fms", (t_asr - t0) * 1000)
        await self.send_json({"type": "user_transcript", "text": text})

        # 同一時間只允許一輪回應; 新話語覆蓋舊回應 (自然的搶話語義)
        await self._interrupt(reason="new_utterance", notify=False)

        self.history.append({"role": "user", "content": text})
        self._trim_history()
        self.tts.add_user_context(text, segment)

        self._tts_interrupt = threading.Event()
        self._respond_task = asyncio.create_task(self._respond(t0))

    async def _respond(self, t0: float | None = None) -> None:
        import time
        spoken: list[str] = []
        interrupt = self._tts_interrupt
        try:
            await self._set_state("thinking")
            queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=4)
            pcfg = getattr(self.cfg, "pipeline", None)
            max_w = getattr(pcfg, "first_chunk_max_words", 9) if pcfg else 9
            min_w = getattr(pcfg, "first_chunk_min_words", 2) if pcfg else 2

            async def produce() -> None:
                try:
                    async for chunk in self.llm.stream_speakable_chunks(
                            self.history, max_w, min_w):
                        await queue.put(chunk)
                finally:
                    await queue.put(None)

            producer = asyncio.create_task(produce())
            try:
                first = True
                while True:
                    sent = await queue.get()
                    if sent is None:
                        break
                    if first and t0 is not None:
                        log.info("[lat] llm_first_chunk=%.0fms (%r)",
                                 (time.monotonic() - t0) * 1000, sent)
                    wav = await self.tts.synthesize(sent, 0, interrupt)
                    if wav is None:  # 被打斷
                        break
                    if first:
                        if t0 is not None:
                            log.info("[lat] FIRST AUDIO=%.0fms (端點→開播)",
                                     (time.monotonic() - t0) * 1000)
                        await self._set_state("speaking")
                        first = False
                    await self.send_json(
                        {"type": "assistant_sentence", "text": sent})
                    await self._send_audio(wav)
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
            if spoken:
                self.history.append(
                    {"role": "assistant", "content": " ".join(spoken)})
                self._trim_history()
            self._respond_task = None
            if not self._client_playing and self.state != "listening":
                await self._set_state("listening")

    # ================= 打斷 =================
    async def _interrupt(self, reason: str, notify: bool = True) -> None:
        task, self._respond_task = self._respond_task, None
        if task is None and self.state == "listening":
            return
        log.info("INTERRUPT (%s)", reason)
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
