"""Kyutai TTS (Delayed Streams Modeling) 幀級流式後端。

為什麼是它: 與 CSM 同源同原理 —— transformer 生成 Mimi codec 音頻碼
(Mimi 正是 Kyutai 為 Moshi 造的, CSM 直接採用), 但 DSM 架構原生支持
「文本邊進、音頻邊出」: 首 token 到首音 ~220ms, 且逐幀 (80ms) 產出,
徹底消除 CSM transformers 實現「整塊生成完才出聲」的結構性延遲。

模型: kyutai/tts-1.6b-en_fr (CC-BY-4.0, bf16 ~3.5GB VRAM)
安裝: pip install -U moshi
音色: 來自 kyutai/tts-voices 預置庫 (聲音嵌入模型未開源,
      不能直接用本地 ref.wav 克隆 —— 倫理限制)

接口與 CSMSynthesizer 對齊: synthesize_stream / add_user_context / reset_context
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import AsyncIterator

import numpy as np
import torch

# Windows CUDA environments commonly do not have a working Triton build.
# Moshi checks this variable at import/decorator time, so set it before
# importing moshi modules in KyutaiTTS.__init__.
os.environ.setdefault("NO_TORCH_COMPILE", "1")

log = logging.getLogger("tts.kyutai")

SAMPLE_RATE = 24000  # Mimi 固定 24kHz
FRAME_S = 0.08       # 12.5 幀/秒


class _Interrupted(Exception):
    pass


class KyutaiTTS:
    def __init__(self, device: str = "cuda", voice: str =
                 "expresso/ex03-ex01_happy_001_channel1_334s.wav",
                 temp: float = 0.6, cfg_coef: float = 2.0,
                 n_q: int = 16, first_emit_frames: int = 2,
                 batch_frames: int = 4, local_files_only: bool = False):
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            from moshi.models.loaders import CheckpointInfo
            from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
        except ImportError as e:
            raise RuntimeError(
                "Kyutai 後端需要 moshi 包: pip install -U moshi\n"
                "或在 config.yaml 設 tts.backend: csm 切回 CSM") from e

        log.info(
            "Loading Kyutai TTS (%s, n_q=%d, torch_compile=disabled, hf_offline=%s) ...",
            device, n_q, bool(os.environ.get("HF_HUB_OFFLINE")),
        )
        info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        self.model = TTSModel.from_checkpoint_info(
            info, n_q=n_q, temp=temp, device=torch.device(device))
        self.cfg_coef = cfg_coef
        self.first_emit_frames = max(1, first_emit_frames)
        self.batch_frames = max(1, batch_frames)
        self._lock = threading.Lock()

        self.voice_path = self.model.get_voice_path(voice)
        self.attributes = self.model.make_condition_attributes(
            [self.voice_path], cfg_coef=cfg_coef)
        self._warmup()
        log.info("Kyutai TTS ready (voice=%s).", voice)

    def _warmup(self) -> None:
        ev = threading.Event()
        self._generate_blocking("Warm up.", ev, emit=lambda pcm: None)

    # ------------------------------------------------------------------
    def _generate_blocking(self, text: str, interrupt: threading.Event,
                           emit) -> None:
        """在線程中運行; 每解出一幀音頻通過 emit(pcm) 回調吐出。"""
        with self._lock:
            entries = self.model.prepare_script([text], padding_between=1)
            t0 = time.monotonic()
            frames_out = 0

            def _on_frame(frame: torch.Tensor) -> None:
                nonlocal frames_out
                if interrupt.is_set():
                    raise _Interrupted
                if (frame == -1).any():
                    return
                # frame: [B, K, 1]; 第 0 流是文本對齊流, 1: 才是音頻碼
                pcm = self.model.mimi.decode(frame[:, 1:, :])
                pcm = pcm.detach().to(torch.float32).cpu().numpy()[0, 0]
                frames_out += 1
                if frames_out == 1:
                    log.info("[lat] kyutai first frame=%.0fms",
                             (time.monotonic() - t0) * 1000)
                emit(np.clip(pcm, -1.0, 1.0))

            try:
                with self.model.mimi.streaming(1):
                    self.model.generate(
                        [entries], [self.attributes], on_frame=_on_frame)
                gen_s = time.monotonic() - t0
                audio_s = frames_out * FRAME_S
                log.info("[lat] kyutai gen=%.2fs audio=%.2fs rtf=%.2f",
                         gen_s, audio_s, gen_s / max(audio_s, 1e-6))
            except _Interrupted:
                log.info("kyutai generation interrupted")

    # ------------------------------------------------------------------
    async def synthesize_stream(
            self, text: str, speaker_id: int,
            interrupt: threading.Event) -> AsyncIterator[np.ndarray]:
        """幀級流式合成: 第 first_emit_frames 幀 (~160ms 音頻) 即產出首塊,
        之後按 batch_frames 聚合, 邊生成邊下發。"""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        buf: list[np.ndarray] = []
        emitted_first = False

        def emit(pcm: np.ndarray) -> None:
            nonlocal emitted_first
            buf.append(pcm)
            need = self.first_emit_frames if not emitted_first \
                else self.batch_frames
            if len(buf) >= need:
                chunk = np.concatenate(buf)
                buf.clear()
                emitted_first = True
                loop.call_soon_threadsafe(q.put_nowait, chunk)

        def run() -> None:
            try:
                self._generate_blocking(text, interrupt, emit)
            finally:
                if buf:
                    tail = np.concatenate(buf)
                    loop.call_soon_threadsafe(q.put_nowait, tail)
                loop.call_soon_threadsafe(q.put_nowait, None)

        worker = loop.run_in_executor(None, run)
        try:
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                if interrupt.is_set():
                    break
                yield chunk
        finally:
            await worker  # 等生成線程收尾 (打斷時它會經 _Interrupted 快速退出)

    # ---- 與 CSM 後端接口對齊 (Kyutai 不吃對話音頻上下文) ----
    def add_user_context(self, text: str, audio_16k: np.ndarray) -> None:
        pass

    def reset_context(self) -> None:
        pass
