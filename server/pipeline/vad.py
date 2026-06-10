"""Silero VAD 串流封裝。

職責:
  1. 端點檢測 (endpointing): 判斷用戶一句話的起點與終點
  2. 打斷檢測 (barge-in): AI 說話期間檢測用戶語音活動

Silero VAD 以 512-sample (16kHz, 32ms) 為一幀, 返回語音概率。
本模塊在其上維護一個小狀態機, 對外發出事件:
  SPEECH_START / SPEECH_END(帶完整語音段) / VOICED_FRAME
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import torch


class VadEvent(Enum):
    NONE = auto()
    SPEECH_START = auto()   # 確認語音開始 (連續語音 > min_speech_ms)
    SPEECH_END = auto()     # 語音結束 (靜音 > min_silence_ms), payload=完整語音段


@dataclass
class VadResult:
    event: VadEvent = VadEvent.NONE
    is_voiced: bool = False                 # 當前幀是否為語音
    voiced_ms: int = 0                      # 當前連續語音時長 (打斷檢測用)
    segment: np.ndarray | None = field(default=None)  # SPEECH_END 時的完整語音段


class StreamingVAD:
    FRAME_SAMPLES = 512  # Silero @16kHz 固定幀長
    SAMPLE_RATE = 16000

    def __init__(self, threshold: float, min_speech_ms: int,
                 min_silence_ms: int, pre_roll_ms: int):
        from silero_vad import load_silero_vad
        self.model = load_silero_vad()  # CPU, 極輕量
        self.threshold = threshold
        self.frame_ms = int(1000 * self.FRAME_SAMPLES / self.SAMPLE_RATE)  # 32ms
        self.min_speech_frames = max(1, min_speech_ms // self.frame_ms)
        self.min_silence_frames = max(1, min_silence_ms // self.frame_ms)
        pre_roll_frames = max(1, pre_roll_ms // self.frame_ms)

        self._pre_roll = collections.deque(maxlen=pre_roll_frames)
        self._residual = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._in_speech = False
        self._voiced_run = 0
        self._silence_run = 0
        self._segment: list[np.ndarray] = []
        self._pre_roll.clear()
        self._residual = np.zeros(0, dtype=np.float32)
        self.model.reset_states()

    def process(self, audio: np.ndarray) -> list[VadResult]:
        """餵入任意長度 float32 16kHz 音頻, 返回逐幀結果列表。"""
        self._residual = np.concatenate([self._residual, audio])
        results: list[VadResult] = []
        while self._residual.shape[0] >= self.FRAME_SAMPLES:
            frame = self._residual[: self.FRAME_SAMPLES]
            self._residual = self._residual[self.FRAME_SAMPLES:]
            results.append(self._step(frame))
        return results

    # ------------------------------------------------------------------
    def _step(self, frame: np.ndarray) -> VadResult:
        prob = self.model(torch.from_numpy(frame), self.SAMPLE_RATE).item()
        voiced = prob >= self.threshold
        res = VadResult(is_voiced=voiced)

        if voiced:
            self._voiced_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            if not self._in_speech:
                self._voiced_run = 0
        res.voiced_ms = self._voiced_run * self.frame_ms

        if not self._in_speech:
            self._pre_roll.append(frame)
            if voiced and self._voiced_run >= self.min_speech_frames:
                # 確認語音開始: 把 pre-roll 一併納入語音段
                self._in_speech = True
                self._segment = list(self._pre_roll)
                res.event = VadEvent.SPEECH_START
        else:
            self._segment.append(frame)
            if self._silence_run >= self.min_silence_frames:
                # 一句話結束
                segment = np.concatenate(self._segment)
                self._in_speech = False
                self._voiced_run = 0
                self._segment = []
                self._pre_roll.clear()
                res.event = VadEvent.SPEECH_END
                res.segment = segment
        return res
