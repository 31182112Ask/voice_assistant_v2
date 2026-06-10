"""faster-whisper 語音識別封裝 (在線程池中運行, 不阻塞事件循環)。"""
from __future__ import annotations

import asyncio
import logging

import numpy as np

log = logging.getLogger("asr")


class WhisperASR:
    def __init__(self, model: str, device: str, compute_type: str,
                 language: str | None, beam_size: int, warmup: bool = True):
        from faster_whisper import WhisperModel
        log.info("Loading faster-whisper '%s' on %s (%s)...", model, device, compute_type)
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size
        if warmup:
            try:
                # 預熱, 避免首次請求的編譯/分配延遲 (失敗不影響啟動)
                self.model.transcribe(np.zeros(16000, dtype=np.float32),
                                      beam_size=1)
            except Exception as e:  # noqa: BLE001
                log.warning("ASR warmup failed (non-fatal): %s", e)
        log.info("ASR ready.")

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,           # 上游已做 VAD
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono 16kHz"""
        return await asyncio.to_thread(self._transcribe_sync, audio)
