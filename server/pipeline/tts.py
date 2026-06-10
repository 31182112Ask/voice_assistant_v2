"""Sesame CSM-1B 語音合成封裝 (官方開源組件)。

採用 Hugging Face transformers 官方集成 (CsmForConditionalGeneration):
  - Llama backbone + depth decoder 生成 Mimi RVQ codes → 24kHz 音頻
  - 帶「對話上下文」生成: 把最近幾輪的 (文本+音頻) 餵入,
    讓韻律自然銜接 —— 這是 Sesame demo 聽感自然的核心機制
  - 可選 voice prompt: 用一段參考音頻錨定固定音色 (基座模型無固定音色)
  - InterruptStoppingCriteria: barge-in 時在 token 級別立刻中止生成

延遲優化 (8GB VRAM 實測要點):
  - bf16 權重 (~4.3GB 含 Mimi)
  - torch.compile depth decoder (社區實測由 ~0.5x 提升到 >1x 實時)
  - 句級流水線: 第一句生成完即播放, 後續句子在播放期間並行生成
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import threading
from collections import deque

import numpy as np
import torch

log = logging.getLogger("tts")

SAMPLE_RATE = 24000  # Mimi codec 固定輸出


class _InterruptCriteria:
    """StoppingCriteria: 每步檢查中斷旗標, barge-in 時即刻停止解碼。"""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
        return self.event.is_set()


class CSMSynthesizer:
    def __init__(self, model_id: str, device: str, dtype: str,
                 compile_decoder: bool, max_audio_len_ms: int,
                 context_turns: int, voice_prompt_cfg, root: pathlib.Path,
                 local_files_only: bool = False,
                 max_context_audio_s: float = 3.0):
        self.max_context_audio_s = max_context_audio_s
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from transformers import AutoProcessor, CsmForConditionalGeneration

        torch_dtype = getattr(torch, dtype)
        self.torch_dtype = torch_dtype
        log.info("Loading CSM-1B (%s, %s) ...", device, dtype)
        self.processor = AutoProcessor.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self.model = CsmForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.device = device
        self.context_turns = context_turns
        # max_new_tokens: Mimi 為 12.5 codes/秒 (80ms/frame)
        self.max_new_tokens = int(max_audio_len_ms / 80)

        if compile_decoder:
            try:
                self.model.depth_decoder = torch.compile(
                    self.model.depth_decoder, mode="reduce-overhead", fullgraph=True
                )
                log.info("depth decoder compiled (reduce-overhead).")
            except Exception as e:  # noqa: BLE001
                log.warning("torch.compile failed, fallback to eager: %s", e)

        # 韻律上下文: (speaker_id, text, audio float32 24k)
        self._context: deque[tuple[int, str, np.ndarray]] = deque(maxlen=context_turns)
        self._voice_prompt: tuple[int, str, np.ndarray] | None = None
        self._load_voice_prompt(voice_prompt_cfg, root)

        self._lock = threading.Lock()  # GPU 上同時只跑一個生成
        self._warmup()
        log.info("CSM ready.")

    # ------------------------------------------------------------------
    def _load_voice_prompt(self, cfg, root: pathlib.Path) -> None:
        if not cfg or not getattr(cfg, "enabled", False):
            return
        try:
            import soundfile as sf
            wav_path, txt_path = root / cfg.wav, root / cfg.text
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                from ..utils.audio import resample_linear
                audio = resample_linear(audio, sr, SAMPLE_RATE)
            text = txt_path.read_text(encoding="utf-8").strip()
            self._voice_prompt = (int(cfg.speaker_id), text, audio)
            vp_s = len(audio) / SAMPLE_RATE
            log.info("Voice prompt loaded: %s (%.1fs)", wav_path.name, vp_s)
            if vp_s > 6.0:
                log.warning(
                    "voice prompt 長達 %.1fs — 它是每一句合成的固定前綴, "
                    "會拖慢所有 TTS 塊; 強烈建議裁剪到 ≤5 秒", vp_s)
        except Exception as e:  # noqa: BLE001
            log.warning("Voice prompt load failed, using base voice: %s", e)

    def _warmup(self) -> None:
        ev = threading.Event()
        self._generate_sync("Warm up.", speaker_id=0, interrupt=ev, use_context=False)

    # ------------------------------------------------------------------
    def _build_conversation(self, text: str, speaker_id: int,
                            use_context: bool) -> list[dict]:
        conv: list[dict] = []
        items: list[tuple[int, str, np.ndarray]] = []
        if use_context:
            if self._voice_prompt is not None:
                items.append(self._voice_prompt)
            items.extend(self._context)
        for spk, t, audio in items:
            conv.append({
                "role": str(spk),
                "content": [{"type": "text", "text": t},
                            {"type": "audio", "audio": audio}],
            })
        conv.append({
            "role": str(speaker_id),
            "content": [{"type": "text", "text": text}],
        })
        return conv

    def _generate_sync(self, text: str, speaker_id: int,
                       interrupt: threading.Event,
                       use_context: bool = True) -> np.ndarray | None:
        from transformers import StoppingCriteriaList

        with self._lock:
            if interrupt.is_set():
                return None
            t_start = __import__("time").monotonic()
            conv = self._build_conversation(text, speaker_id, use_context)
            inputs = self.processor.apply_chat_template(
                conv, tokenize=True, return_dict=True,
            ).to(self.device)
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    inputs[key] = value.to(device=self.device, dtype=self.torch_dtype)
            with torch.inference_mode():
                audio = self.model.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    depth_decoder_do_sample=True,
                    depth_decoder_temperature=0.9,
                    stopping_criteria=StoppingCriteriaList(
                        [_InterruptCriteria(interrupt)]),
                )
            if interrupt.is_set():
                return None
            wav = audio[0].to(torch.float32).cpu().numpy().reshape(-1)
            gen_s = __import__("time").monotonic() - t_start
            audio_s = len(wav) / SAMPLE_RATE
            log.info("[lat] tts gen=%.2fs audio=%.2fs rtf=%.2f (%d chars)",
                     gen_s, audio_s, gen_s / max(audio_s, 1e-6), len(text))
            return wav

    def _cap(self, audio: np.ndarray) -> np.ndarray:
        """上下文音頻只保留尾部 N 秒 (韻律銜接看的是最近的節奏)。"""
        cap = int(self.max_context_audio_s * SAMPLE_RATE)
        return audio[-cap:] if len(audio) > cap else audio

    # ------------------------------------------------------------------
    async def synthesize(self, text: str, speaker_id: int,
                         interrupt: threading.Event) -> np.ndarray | None:
        """合成一句話。返回 float32 24kHz; 被打斷時返回 None。"""
        wav = await asyncio.to_thread(
            self._generate_sync, text, speaker_id, interrupt)
        if wav is not None:
            self._context.append((speaker_id, text, self._cap(wav)))
        return wav

    def add_user_context(self, text: str, audio_16k: np.ndarray) -> None:
        """把用戶語音也放入韻律上下文 (speaker 1), 讓 AI 回應的語氣呼應用戶。"""
        from ..utils.audio import resample_linear
        audio_24k = resample_linear(audio_16k, 16000, SAMPLE_RATE)
        self._context.append((1, text, self._cap(audio_24k)))

    def reset_context(self) -> None:
        self._context.clear()
