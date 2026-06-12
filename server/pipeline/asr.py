"""faster-whisper 語音識別封裝 (在線程池中運行, 不阻塞事件循環)。

準確率增強 (零/低顯存代價):
  - 段首尾各墊靜音: whisper 對「貼邊起始」的音頻容易丟首詞,
    前置 ~250ms 靜音是廉價而有效的起點保護
  - initial_prompt 上下文偏置: 把最近的對話文本餵入, 專名/話題詞
    的識別準確率顯著提升 (condition_on_previous_text 仍關閉,
    不會引入跨段幻聽)
  - 能量門限: RMS 過低的段直接丟棄, 避免噪聲段幻聽出文字
  - beam search: 投機轉寫已把 ASR 移出關鍵路徑 —— beam 的額外
    耗時被端點前的靜默期吸收, 等於免費的準確率
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import numpy as np

log = logging.getLogger("asr")

_PAD_HEAD_S = 0.25   # 段首靜音墊
_PAD_TAIL_S = 0.15   # 段尾靜音墊
_MIN_RMS = 0.0045    # 能量門限 (AGC 後的人聲遠高於此)


class WhisperASR:
    def __init__(self, model: str, device: str, compute_type: str,
                 language: str | None, beam_size: int, warmup: bool = True,
                 context_bias: bool = True,
                 local_files_only: bool = False,
                 fallback_models: Sequence[str] | None = None):
        from faster_whisper import WhisperModel
        candidates = [model, *(fallback_models or [])]
        seen: set[str] = set()
        last_error: Exception | None = None
        self.model = None
        self.active_model = model
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            log.info("Loading faster-whisper '%s' on %s (%s, offline=%s)...",
                     candidate, device, compute_type, local_files_only)
            try:
                self.model = WhisperModel(
                    candidate,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=local_files_only,
                )
                self.active_model = candidate
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                if not local_files_only:
                    raise
                log.warning(
                    "ASR model '%s' is not available in the local cache: %s",
                    candidate, e)
        if self.model is None:
            tried = ", ".join(seen)
            raise RuntimeError(
                "No local faster-whisper model is available. "
                f"Tried: {tried}. Run `python scripts/download_models.py` "
                "with network access, or set `asr.model` to a cached model/path."
            ) from last_error
        self.language = language
        self.beam_size = beam_size
        self.context_bias = context_bias
        self._context_text = ""           # 最近對話文本 (偏置用)
        if warmup:
            try:
                # 預熱, 避免首次請求的編譯/分配延遲 (失敗不影響啟動)
                self.model.transcribe(np.zeros(16000, dtype=np.float32),
                                      beam_size=1)
            except Exception as e:  # noqa: BLE001
                log.warning("ASR warmup failed (non-fatal): %s", e)
        log.info("ASR ready: %s (beam=%d, bias=%s).",
                 self.active_model, beam_size, context_bias)

    # ------------------------------------------------------------------
    def set_context(self, history: list[dict], max_chars: int = 200) -> None:
        """用最近對話文本更新偏置 prompt (調用方在每輪後刷新)。"""
        if not self.context_bias:
            return
        parts: list[str] = []
        for turn in reversed(history):
            t = (turn.get("content") or "").strip()
            if not t:
                continue
            parts.insert(0, t)
            if sum(len(p) for p in parts) > max_chars:
                break
        self._context_text = " ".join(parts)[-max_chars:]

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        if rms < _MIN_RMS:
            log.info("asr: segment below energy gate (rms=%.4f), dropped", rms)
            return ""
        padded = np.concatenate([
            np.zeros(int(_PAD_HEAD_S * 16000), dtype=np.float32),
            audio.astype(np.float32),
            np.zeros(int(_PAD_TAIL_S * 16000), dtype=np.float32),
        ])
        segments, _info = self.model.transcribe(
            padded,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,           # 上游已做 VAD
            condition_on_previous_text=False,
            initial_prompt=self._context_text or None,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono 16kHz"""
        return await asyncio.to_thread(self._transcribe_sync, audio)
