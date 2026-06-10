from __future__ import annotations

import asyncio
from collections import deque
import logging
import os
import queue
import re
import threading
import time
from typing import AsyncIterator

import numpy as np
import torch

# Windows CUDA environments commonly do not have a working Triton build.
# Moshi checks this variable at import/decorator time.
os.environ.setdefault("NO_TORCH_COMPILE", "1")

log = logging.getLogger("tts.kyutai")

SAMPLE_RATE = 24000
FRAME_S = 0.08


class _Interrupted(Exception):
    pass


class KyutaiTTS:
    """Kyutai DSM TTS backend.

    Two modes are exposed:
    - synthesize_stream(text): text chunk in, audio frames out.
    - synthesize_text_stream(text_stream): text fragments in, same generation
      state audio frames out. This is the low-latency path used by the
      orchestrator when available.
    """

    def __init__(
        self,
        device: str = "cuda",
        voice: str = "expresso/ex03-ex01_happy_001_channel1_334s.wav",
        temp: float = 0.6,
        cfg_coef: float = 2.0,
        n_q: int = 16,
        first_emit_frames: int = 2,
        batch_frames: int = 4,
        local_files_only: bool = False,
    ):
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            from moshi.models.loaders import CheckpointInfo
            from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
        except ImportError as e:
            raise RuntimeError(
                "Kyutai backend requires moshi. Install with: pip install -U moshi"
            ) from e

        log.info(
            "Loading Kyutai TTS (%s, n_q=%d, torch_compile=disabled, hf_offline=%s) ...",
            device,
            n_q,
            bool(os.environ.get("HF_HUB_OFFLINE")),
        )
        info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        self.model = TTSModel.from_checkpoint_info(
            info, n_q=n_q, temp=temp, device=torch.device(device)
        )
        self.cfg_coef = cfg_coef
        self.first_emit_frames = max(1, first_emit_frames)
        self.batch_frames = max(1, batch_frames)
        self._lock = threading.Lock()

        self.voice_path = self.model.get_voice_path(voice)
        self.attributes = self.model.make_condition_attributes(
            [self.voice_path], cfg_coef=cfg_coef
        )
        self._warmup()
        log.info("Kyutai TTS ready (voice=%s).", voice)

    def _warmup(self) -> None:
        ev = threading.Event()
        self._generate_blocking("Warm up.", ev, emit=lambda _pcm: None)

    def _generate_blocking(self, text: str, interrupt: threading.Event, emit) -> None:
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
                pcm = self.model.mimi.decode(frame[:, 1:, :])
                pcm = pcm.detach().to(torch.float32).cpu().numpy()[0, 0]
                frames_out += 1
                if frames_out == 1:
                    log.info(
                        "[lat] kyutai first frame=%.0fms",
                        (time.monotonic() - t0) * 1000,
                    )
                emit(np.clip(pcm, -1.0, 1.0))

            try:
                with self.model.mimi.streaming(1):
                    self.model.generate(
                        [entries], [self.attributes], on_frame=_on_frame
                    )
                gen_s = time.monotonic() - t0
                audio_s = frames_out * FRAME_S
                log.info(
                    "[lat] kyutai gen=%.2fs audio=%.2fs rtf=%.2f",
                    gen_s,
                    audio_s,
                    gen_s / max(audio_s, 1e-6),
                )
            except _Interrupted:
                log.info("kyutai generation interrupted")

    def _append_completed_text(self, state, text_buf: str, input_done: bool) -> str:
        if not text_buf:
            return ""
        if input_done:
            ready, rest = text_buf, ""
        else:
            match = re.search(r"\s+(?=\S*$)", text_buf)
            if match is None:
                return text_buf
            ready = text_buf[: match.end()]
            rest = text_buf[match.end():]
        ready = ready.strip()
        if ready:
            state.entries.extend(self.model.prepare_script([ready], padding_between=1))
        return rest

    def _generate_text_stream_blocking(
        self,
        text_q: queue.Queue[str | None],
        interrupt: threading.Event,
        emit,
    ) -> str:
        from moshi.models.lm import LMGen
        from moshi.models.tts import Entry

        with self._lock:
            assert self.model.lm.condition_provider is not None
            condition_tensors = self.model.lm.condition_provider.prepare_and_provide(
                [self.attributes]
            )
            if not self.model.multistream:
                self.model.lm.dep_q = self.model.n_q

            state = self.model.machine.new_state([])
            state.entries = deque()
            text_buf = ""
            input_done = False
            text_started = False
            spoken_parts: list[str] = []
            t0 = time.monotonic()
            frames_out = 0
            device = self.model.lm.device

            def drain_text() -> None:
                nonlocal text_buf, input_done, text_started
                while True:
                    try:
                        item = text_q.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        input_done = True
                        break
                    text_buf += item
                before = text_buf
                text_buf = self._append_completed_text(state, text_buf, input_done)
                if before != text_buf:
                    spoken = before[: len(before) - len(text_buf)].strip()
                    if spoken:
                        spoken_parts.append(spoken)
                        text_started = True

            def on_text_logits_hook(text_logits: torch.Tensor) -> None:
                if self.model.padding_bonus:
                    text_logits[..., self.model.machine.token_ids.pad] += (
                        self.model.padding_bonus
                    )

            def on_audio_hook(audio_tokens: torch.Tensor) -> None:
                audio_offset = self.model.lm.audio_offset
                delays = self.model.lm.delays
                for q in range(audio_tokens.shape[1]):
                    delay = delays[q + audio_offset]
                    if offset < delay + self.model.delay_steps:
                        audio_tokens[:, q] = self.model.machine.token_ids.zero

            logged_text_tokens: list[tuple[int, int]] = []

            def on_text_hook(text_tokens: torch.Tensor) -> None:
                drain_text()
                if (
                    not input_done
                    and not state.entries
                    and not state.queued
                    and not state.lookahead_queued
                ):
                    # Keep the DSM state alive while more text is still arriving.
                    state.entries.append(Entry(tokens=[], text="", padding=1))
                token = text_tokens.tolist()[0]
                out_token, _ = self.model.machine.process(offset, state, token)
                if state.end_step is not None and not input_done:
                    state.end_step = None
                logged_text_tokens.append((token, out_token))
                text_tokens[:] = torch.tensor(
                    [out_token], dtype=torch.long, device=text_tokens.device
                )

            lm_gen = LMGen(
                self.model.lm,
                temp=self.model.temp,
                temp_text=self.model.temp,
                cfg_coef=self.model.cfg_coef,
                condition_tensors=condition_tensors,
                on_text_logits_hook=on_text_logits_hook,
                on_text_hook=on_text_hook,
                on_audio_hook=on_audio_hook,
            )
            missing = self.model.lm.n_q - self.model.lm.dep_q
            no_depformer_tokens = torch.full(
                (1, self.model.lm.dep_q, 1),
                self.model.machine.token_ids.zero,
                dtype=torch.long,
                device=device,
            )

            try:
                while not input_done and not state.entries:
                    try:
                        item = text_q.get(timeout=0.01)
                    except queue.Empty:
                        if interrupt.is_set():
                            raise _Interrupted
                        continue
                    if item is None:
                        input_done = True
                    else:
                        text_buf += item
                    drain_text()

                with lm_gen.streaming(1), self.model.mimi.streaming(1):
                    for offset in range(self.model.max_gen_length):
                        if interrupt.is_set():
                            raise _Interrupted
                        drain_text()
                        if input_done and state.end_step is not None:
                            if (
                                offset
                                >= state.end_step
                                + self.model.delay_steps
                                + self.model.final_padding
                            ):
                                break
                        input_tokens = torch.full(
                            (1, missing, 1),
                            self.model.machine.token_ids.zero,
                            dtype=torch.long,
                            device=device,
                        )
                        depformer_replace_tokens = (
                            no_depformer_tokens
                            if offset < self.model.delay_steps
                            else None
                        )
                        frame = lm_gen.step(
                            input_tokens,
                            depformer_replace_tokens=depformer_replace_tokens,
                        )
                        if frame is None or (frame == -1).any() or not text_started:
                            continue
                        pcm = self.model.mimi.decode(frame[:, 1:, :])
                        pcm = pcm.detach().to(torch.float32).cpu().numpy()[0, 0]
                        frames_out += 1
                        if frames_out == 1:
                            log.info(
                                "[lat] kyutai stream-in first frame=%.0fms",
                                (time.monotonic() - t0) * 1000,
                            )
                        emit(np.clip(pcm, -1.0, 1.0))
                gen_s = time.monotonic() - t0
                audio_s = frames_out * FRAME_S
                log.info(
                    "[lat] kyutai stream-in gen=%.2fs audio=%.2fs rtf=%.2f",
                    gen_s,
                    audio_s,
                    gen_s / max(audio_s, 1e-6),
                )
            except _Interrupted:
                log.info("kyutai stream-in generation interrupted")
            return " ".join(part for part in spoken_parts if part)

    async def synthesize_stream(
        self, text: str, speaker_id: int, interrupt: threading.Event
    ) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        buf: list[np.ndarray] = []
        emitted_first = False

        def emit(pcm: np.ndarray) -> None:
            nonlocal emitted_first
            buf.append(pcm)
            need = self.first_emit_frames if not emitted_first else self.batch_frames
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
                    loop.call_soon_threadsafe(q.put_nowait, np.concatenate(buf))
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
            await worker

    async def synthesize_text_stream(
        self,
        text_stream: AsyncIterator[str],
        speaker_id: int,
        interrupt: threading.Event,
    ) -> AsyncIterator[tuple[np.ndarray, str | None]]:
        loop = asyncio.get_running_loop()
        text_q: queue.Queue[str | None] = queue.Queue()
        audio_q: asyncio.Queue[tuple[np.ndarray | None, str | None]] = asyncio.Queue()
        buf: list[np.ndarray] = []
        emitted_first = False

        def emit(pcm: np.ndarray) -> None:
            nonlocal emitted_first
            buf.append(pcm)
            need = self.first_emit_frames if not emitted_first else self.batch_frames
            if len(buf) >= need:
                chunk = np.concatenate(buf)
                buf.clear()
                emitted_first = True
                loop.call_soon_threadsafe(audio_q.put_nowait, (chunk, None))

        def run() -> None:
            spoken = ""
            try:
                spoken = self._generate_text_stream_blocking(text_q, interrupt, emit)
            finally:
                if buf:
                    loop.call_soon_threadsafe(
                        audio_q.put_nowait, (np.concatenate(buf), None)
                    )
                loop.call_soon_threadsafe(audio_q.put_nowait, (None, spoken))

        async def feed_text() -> None:
            try:
                async for text in text_stream:
                    if interrupt.is_set():
                        break
                    text_q.put(text)
            finally:
                text_q.put(None)

        worker = loop.run_in_executor(None, run)
        feeder = asyncio.create_task(feed_text())
        try:
            while True:
                chunk, spoken = await audio_q.get()
                if chunk is None:
                    if spoken:
                        yield np.zeros(0, dtype=np.float32), spoken
                    break
                if interrupt.is_set():
                    break
                yield chunk, None
        finally:
            feeder.cancel()
            await worker

    def add_user_context(self, text: str, audio_16k: np.ndarray) -> None:
        pass

    def reset_context(self) -> None:
        pass
