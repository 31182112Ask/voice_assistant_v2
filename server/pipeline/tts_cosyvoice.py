"""CosyVoice3 雙向流式 TTS 後端 (Fun-CosyVoice3-0.5B-2512, Apache 2.0)。

為什麼是它:
  - 原生 Bi-Streaming: 文本流入 + 音頻流出, 官方首包 ~150ms ——
    與本項目 orchestrator 的 synthesize_text_stream 路徑天然同構
  - 9 語言 + 18 種中文方言 → 解鎖中文語音回覆 (Kyutai/CSM 僅英文)
  - 0.5B (Qwen2 backbone → 語音 token → DiT 流匹配 → HiFT 聲碼器),
    fp16 約 2-2.5GB 顯存, 比 Kyutai 1.6B 還省 ~1GB
  - 24kHz 輸出, 與現有播放鏈路一致, 前端零改動

部署 (官方倉庫不是 pip 包):
  scripts/setup_cosyvoice.ps1   # 克隆 FunAudioLLM/CosyVoice + 下載權重
  config.yaml → tts.cosyvoice.repo_dir / model_dir

接口與 KyutaiTTS 對齊:
  synthesize_stream(text)             → 整句文本, 流式出音頻塊
  synthesize_text_stream(text_stream) → LLM 塊流直餵, 同一生成態流式出聲 ★
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import queue
import sys
import threading
import time
from typing import AsyncIterator

import numpy as np

log = logging.getLogger("tts.cosyvoice")

SAMPLE_RATE = 24000


class _Interrupted(Exception):
    pass


class CosyVoiceTTS:
    def __init__(
        self,
        repo_dir: str,
        model_dir: str,
        device: str = "cuda",
        fp16: bool = True,
        speed: float = 1.0,
        instruct: str = "You are a helpful assistant.",
        voice_wav: str | None = None,
        voice_text: str | None = None,
        root: pathlib.Path | None = None,
        max_prompt_chars: int = 0,
        warmup_text: str = "Warm up.",
    ):
        repo = pathlib.Path(repo_dir).expanduser()
        if not repo.exists():
            raise RuntimeError(
                f"CosyVoice 倉庫不存在: {repo} — 先運行 scripts/setup_cosyvoice.ps1")
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))

        from cosyvoice.cli.cosyvoice import AutoModel

        log.info("Loading CosyVoice3 (%s, fp16=%s) ...", model_dir, fp16)
        try:
            self.model = AutoModel(model_dir=model_dir, fp16=fp16)
        except TypeError:
            # 兼容舊簽名 (load_jit/load_trt 等位置參數差異)
            self.model = AutoModel(model_dir=model_dir)
        self.sample_rate = int(getattr(self.model, "sample_rate", SAMPLE_RATE))
        self.speed = speed
        self.warmup_text = warmup_text
        self._lock = threading.Lock()

        # ---- 音色: 沿用項目慣例 voices/ref.wav + ref.txt ----
        # CosyVoice3 的 prompt_text 格式: "<instruct><|endofprompt|><參考音頻逐字稿>"
        wav_path = pathlib.Path(voice_wav) if voice_wav else None
        if root and wav_path and not wav_path.is_absolute():
            wav_path = root / wav_path
        if wav_path and wav_path.exists():
            self.prompt_wav = str(wav_path)
            ref_text = ""
            txt_path = pathlib.Path(voice_text) if voice_text else None
            if root and txt_path and not txt_path.is_absolute():
                txt_path = root / txt_path
            if txt_path and txt_path.exists():
                ref_text = txt_path.read_text(encoding="utf-8").strip()
            if max_prompt_chars > 0 and len(ref_text) > max_prompt_chars:
                ref_text = ref_text[:max_prompt_chars].rstrip()
            self.prompt_text = f"{instruct}<|endofprompt|>{ref_text}"
            log.info("CosyVoice voice prompt: %s (%d chars transcript)",
                     wav_path.name, len(ref_text))
        else:
            # 回退到官方倉庫自帶的零樣本示例音色
            asset = repo / "asset" / "zero_shot_prompt.wav"
            self.prompt_wav = str(asset)
            self.prompt_text = (f"{instruct}<|endofprompt|>"
                                "希望你以后能够做的比我还好呦。")
            log.info("CosyVoice voice prompt: 官方示例音色 (放置 voices/ref.wav"
                     " + ref.txt 可換成自己的)")
        self._warmup()
        log.info("CosyVoice3 ready (sr=%d).", self.sample_rate)

    # ------------------------------------------------------------------
    def _warmup(self) -> None:
        ev = threading.Event()
        for _ in self._infer_sync(self.warmup_text, ev):
            pass

    def _infer_sync(self, tts_text, interrupt: threading.Event):
        """同步生成器: tts_text 可為 str 或文本生成器 (雙向流式)。
        每次產出 float32 mono 24kHz 音頻塊; interrupt 置位即關閉生成。
        """
        with self._lock:
            t0 = time.monotonic()
            first = True
            gen = self.model.inference_zero_shot(
                tts_text, self.prompt_text, self.prompt_wav,
                stream=True, speed=self.speed,
            )
            try:
                for out in gen:
                    if interrupt.is_set():
                        raise _Interrupted
                    pcm = out["tts_speech"].reshape(-1).numpy()
                    if first:
                        first = False
                        log.info("[lat] cosyvoice first packet=%.0fms",
                                 (time.monotonic() - t0) * 1000)
                    yield np.clip(pcm.astype(np.float32), -1.0, 1.0)
                gen_s = time.monotonic() - t0
                log.info("[lat] cosyvoice gen=%.2fs", gen_s)
            except _Interrupted:
                log.info("cosyvoice generation interrupted")
            finally:
                gen.close()      # 關閉底層 LM/flow 生成, 釋放 GPU

    # ------------------------------------------------------------------
    async def synthesize_stream(
        self, text: str, speaker_id: int, interrupt: threading.Event
    ) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        def run() -> None:
            try:
                for pcm in self._infer_sync(text, interrupt):
                    loop.call_soon_threadsafe(q.put_nowait, pcm)
            finally:
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
        """雙向流式: LLM 塊流直餵 CosyVoice 的同一個生成態。

        橋接: asyncio 文本流 → 線程安全隊列 → 同步文本生成器 (模型消費)。
        模型邊收文本邊出音頻; 文本流結束 (None 哨兵) 後自然收尾。
        """
        loop = asyncio.get_running_loop()
        text_q: queue.Queue[str | None] = queue.Queue()
        audio_q: asyncio.Queue[tuple[np.ndarray | None, str | None]] = asyncio.Queue()
        spoken_parts: list[str] = []

        def text_gen():
            while True:
                item = text_q.get()
                if item is None:
                    return
                if interrupt.is_set():
                    return
                spoken_parts.append(item.strip())
                yield item

        def run() -> None:
            try:
                for pcm in self._infer_sync(text_gen(), interrupt):
                    loop.call_soon_threadsafe(audio_q.put_nowait, (pcm, None))
            finally:
                loop.call_soon_threadsafe(
                    audio_q.put_nowait,
                    (None, " ".join(p for p in spoken_parts if p)))

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
            text_q.put(None)     # 雙保險: 喚醒可能阻塞在 get() 的模型線程
            await worker

    # ------------------------------------------------------------------
    def add_user_context(self, text: str, audio_16k: np.ndarray) -> None:
        pass

    def reset_context(self) -> None:
        pass
