"""音頻基礎工具: PCM16 <-> float32 轉換、重採樣。"""
from __future__ import annotations

import numpy as np


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """bytes (int16 LE) -> float32 [-1, 1]"""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] -> bytes (int16 LE)"""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """輕量線性插值重採樣 (語音場景足夠, 避免引入額外依賴)。"""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.shape[0] / src_rate
    dst_len = int(round(duration * dst_rate))
    src_idx = np.linspace(0.0, audio.shape[0] - 1, dst_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(audio.shape[0]), audio).astype(np.float32)
