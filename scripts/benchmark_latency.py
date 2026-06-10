from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.config import load_config
from server.pipeline.asr import WhisperASR
from server.pipeline.llm import OllamaLLM
from server.pipeline.tts import CSMSynthesizer
from server.utils.audio import resample_linear


TARGETS = {
    "natural": {
        "endpoint_ms": 200,
        "asr_ms": 150,
        "model_first_audio_ms": 700,
        "estimated_user_stop_to_audio_ms": 1000,
        "tts_first_chunk_ms": 300,
        "tts_rtf": 1.0,
        "max_inter_chunk_gap_ms": 360,
    },
    "local": {
        "endpoint_ms": 320,
        "asr_ms": 400,
        "model_first_audio_ms": 1200,
        "estimated_user_stop_to_audio_ms": 1500,
        "tts_first_chunk_ms": 700,
        "tts_rtf": 1.0,
        "max_inter_chunk_gap_ms": 420,
    },
}

PROMPTS = [
    "Can you hear me?",
    "What's a simple way to make coffee taste better?",
    "I have ten minutes before a meeting. Help me prepare.",
]

TTS_TEXTS = [
    "Sure. I can hear you clearly.",
    "Try grinding fresh beans and using water just off the boil.",
    "Start with the one decision you need from the meeting.",
]


def _ms(start: float, end: float | None = None) -> float:
    return ((end or time.perf_counter()) - start) * 1000


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": round(min(values), 1),
        "median": round(statistics.median(values), 1),
        "max": round(max(values), 1),
    }


def _pass(value: float | None, threshold: float, lower_is_better: bool = True) -> bool:
    if value is None:
        return False
    return value <= threshold if lower_is_better else value >= threshold


async def build_llm(cfg) -> OllamaLLM:
    llm = OllamaLLM(
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        system_prompt=cfg.llm.system_prompt,
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
        think=getattr(cfg.llm, "think", None),
        num_gpu=getattr(cfg.llm, "num_gpu", None),
        num_batch=getattr(cfg.llm, "num_batch", None),
        keep_alive=getattr(cfg.llm, "keep_alive", None),
    )
    await llm.warmup()
    return llm


async def build_tts(cfg):
    backend = getattr(cfg.tts, "backend", "csm")
    if backend == "kyutai":
        from server.pipeline.tts_kyutai import KyutaiTTS

        kcfg = getattr(cfg.tts, "kyutai", None)
        return await asyncio.to_thread(
            KyutaiTTS,
            cfg.tts.device,
            getattr(kcfg, "voice", "expresso/ex03-ex01_happy_001_channel1_334s.wav"),
            getattr(kcfg, "temp", 0.6),
            getattr(kcfg, "cfg_coef", 2.0),
            getattr(kcfg, "n_q", 16),
            getattr(kcfg, "first_emit_frames", 2),
            getattr(kcfg, "batch_frames", 4),
            getattr(kcfg, "local_files_only", getattr(cfg.tts, "local_files_only", False)),
        )
    return await asyncio.to_thread(
        CSMSynthesizer,
        cfg.tts.model_id,
        cfg.tts.device,
        cfg.tts.dtype,
        cfg.tts.compile_decoder,
        cfg.tts.max_audio_len_ms,
        cfg.tts.context_turns,
        cfg.tts.voice_prompt,
        ROOT,
        getattr(cfg.tts, "local_files_only", False),
        getattr(cfg.tts, "max_context_audio_s", 3.0),
    )


async def cache_instant_ack(tts, cfg) -> np.ndarray | None:
    pcfg = getattr(cfg, "pipeline", None)
    if not pcfg or not getattr(pcfg, "instant_ack_enabled", False):
        return None
    if not getattr(pcfg, "instant_ack_cache", False):
        return None
    ack = getattr(pcfg, "instant_ack_text", "Okay.").strip()
    if not ack:
        return None
    chunks: list[np.ndarray] = []
    interrupt = threading.Event()
    async for pcm in tts.synthesize_stream(ack, 0, interrupt):
        chunks.append(pcm.astype(np.float32, copy=False))
    if not chunks:
        return None
    pcfg.instant_ack_pcm = np.concatenate(chunks)
    return pcfg.instant_ack_pcm


def load_audio_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = resample_linear(audio, sr, 16000)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


async def build_asr(cfg) -> WhisperASR:
    return await asyncio.to_thread(
        WhisperASR,
        cfg.asr.model,
        cfg.asr.device,
        cfg.asr.compute_type,
        cfg.asr.language,
        cfg.asr.beam_size,
        getattr(cfg.asr, "warmup", True),
    )


async def measure_asr(asr: WhisperASR, audio: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()
    text = await asr.transcribe(audio)
    return {
        "audio_s": round(len(audio) / 16000, 3),
        "ms": round(_ms(started), 1),
        "text": text,
    }


async def measure_llm_first_chunk(llm: OllamaLLM, cfg, prompt: str) -> dict[str, Any]:
    pcfg = getattr(cfg, "pipeline", None)
    max_w = getattr(pcfg, "first_chunk_max_words", 5) if pcfg else 5
    min_w = getattr(pcfg, "first_chunk_min_words", 1) if pcfg else 1
    started = time.perf_counter()
    stream = llm.stream_speakable_chunks(
        [{"role": "user", "content": prompt}],
        first_chunk_max_words=max_w,
        first_chunk_min_words=min_w,
    )
    try:
        chunk = await anext(stream)
        return {"prompt": prompt, "ms": round(_ms(started), 1), "chunk": chunk}
    except StopAsyncIteration:
        pass
    finally:
        await stream.aclose()
    return {"prompt": prompt, "ms": None, "chunk": ""}


async def measure_tts(tts, text: str) -> dict[str, Any]:
    interrupt = threading.Event()
    started = time.perf_counter()
    first_ms: float | None = None
    last_ms: float | None = None
    gaps: list[float] = []
    chunks = 0
    samples = 0

    async for pcm in tts.synthesize_stream(text, 0, interrupt):
        now_ms = _ms(started)
        if first_ms is None:
            first_ms = now_ms
        if last_ms is not None:
            gaps.append(now_ms - last_ms)
        last_ms = now_ms
        chunks += 1
        samples += len(pcm)

    elapsed_s = _ms(started) / 1000
    audio_s = samples / 24000
    return {
        "text": text,
        "first_chunk_ms": round(first_ms, 1) if first_ms is not None else None,
        "elapsed_ms": round(elapsed_s * 1000, 1),
        "audio_s": round(audio_s, 3),
        "rtf": round(elapsed_s / max(audio_s, 1e-6), 3),
        "chunks": chunks,
        "max_gap_ms": round(max(gaps), 1) if gaps else 0.0,
    }


async def measure_model_first_audio(llm: OllamaLLM, tts, cfg, prompt: str) -> dict[str, Any]:
    pcfg = getattr(cfg, "pipeline", None)
    max_w = getattr(pcfg, "first_chunk_max_words", 5) if pcfg else 5
    min_w = getattr(pcfg, "first_chunk_min_words", 1) if pcfg else 1
    started = time.perf_counter()

    async def first_llm_chunk() -> tuple[str, float | None]:
        stream = llm.stream_speakable_chunks(
            [{"role": "user", "content": prompt}],
            first_chunk_max_words=max_w,
            first_chunk_min_words=min_w,
        )
        try:
            chunk = await anext(stream)
            return chunk, _ms(started)
        except StopAsyncIteration:
            return "", None
        finally:
            await stream.aclose()
        return "", None

    llm_task = asyncio.create_task(first_llm_chunk())
    if getattr(pcfg, "instant_ack_enabled", False):
        ack = getattr(pcfg, "instant_ack_text", "Okay.").strip()
        if ack:
            cached_ack = getattr(pcfg, "instant_ack_pcm", None)
            if cached_ack is not None:
                first_audio_ms = _ms(started)
                first_chunk, llm_ms = await llm_task
                return {
                    "prompt": prompt,
                    "strategy": "cached_ack",
                    "chunk": ack,
                    "llm_chunk": first_chunk,
                    "llm_first_chunk_ms": round(llm_ms, 1) if llm_ms is not None else None,
                    "model_first_audio_ms": round(first_audio_ms, 1),
                    "cached_ack_audio_ms": round(len(cached_ack) / 24, 1),
                }
            else:
                interrupt = threading.Event()
                async for _pcm in tts.synthesize_stream(ack, 0, interrupt):
                    first_audio_ms = _ms(started)
                    first_chunk, llm_ms = await llm_task
                    return {
                        "prompt": prompt,
                        "strategy": "instant_ack",
                        "chunk": ack,
                        "llm_chunk": first_chunk,
                        "llm_first_chunk_ms": round(llm_ms, 1) if llm_ms is not None else None,
                        "model_first_audio_ms": round(first_audio_ms, 1),
                    }

    first_chunk = ""
    llm_ms: float | None = None

    first_chunk, llm_ms = await llm_task

    if not first_chunk:
        return {"prompt": prompt, "llm_first_chunk_ms": None, "model_first_audio_ms": None}

    interrupt = threading.Event()
    async for _pcm in tts.synthesize_stream(first_chunk, 0, interrupt):
        return {
            "prompt": prompt,
            "chunk": first_chunk,
            "llm_first_chunk_ms": round(llm_ms, 1) if llm_ms is not None else None,
            "model_first_audio_ms": round(_ms(started), 1),
        }
    return {
        "prompt": prompt,
        "chunk": first_chunk,
        "llm_first_chunk_ms": round(llm_ms, 1) if llm_ms is not None else None,
        "model_first_audio_ms": None,
    }


def evaluate(results: dict[str, Any], profile: str) -> dict[str, Any]:
    target = TARGETS[profile]
    endpoint_ms = results["config"]["endpoint_ms"]
    asr_ms = results["summary"]["asr_ms"]["median"]
    model_first_audio = results["summary"]["model_first_audio_ms"]["median"]
    estimated_total = results["summary"]["estimated_user_stop_to_audio_ms"]
    tts_first = results["summary"]["tts_first_chunk_ms"]["median"]
    tts_rtf = results["summary"]["tts_rtf"]["median"]
    max_gap = results["summary"]["max_inter_chunk_gap_ms"]["max"]
    checks = {
        "endpoint": _pass(endpoint_ms, target["endpoint_ms"]),
        "asr": _pass(asr_ms, target["asr_ms"]) if asr_ms is not None else True,
        "model_first_audio": _pass(model_first_audio, target["model_first_audio_ms"]),
        "estimated_user_stop_to_audio": _pass(
            estimated_total, target["estimated_user_stop_to_audio_ms"]
        ),
        "tts_first_chunk": _pass(tts_first, target["tts_first_chunk_ms"]),
        "tts_rtf": _pass(tts_rtf, target["tts_rtf"]),
        "inter_chunk_gap": _pass(max_gap, target["max_inter_chunk_gap_ms"]),
    }
    return {"profile": profile, "targets": target, "checks": checks, "pass": all(checks.values())}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Measure voice pipeline latency.")
    parser.add_argument("--profile", choices=sorted(TARGETS), default="local")
    parser.add_argument("--json", type=Path, default=ROOT / "latency_benchmark.json")
    parser.add_argument("--asr-wav", type=Path, default=ROOT / "voices" / "ref.wav")
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if getattr(cfg.tts, "local_files_only", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    asr_runs: list[dict[str, Any]] = []
    if not args.skip_asr and args.asr_wav.exists():
        asr = await build_asr(cfg)
        audio = load_audio_16k(args.asr_wav)
        asr_runs = [await measure_asr(asr, audio) for _ in range(3)]

    llm = await build_llm(cfg)
    tts = await build_tts(cfg)
    await cache_instant_ack(tts, cfg)

    llm_runs = [await measure_llm_first_chunk(llm, cfg, prompt) for prompt in PROMPTS]
    tts_runs = [await measure_tts(tts, text) for text in TTS_TEXTS]
    combined_runs = [await measure_model_first_audio(llm, tts, cfg, prompt) for prompt in PROMPTS]

    results: dict[str, Any] = {
        "config": {
            "endpoint_ms": getattr(
                cfg.vad, "fast_min_silence_ms",
                getattr(cfg.vad, "min_silence_ms", None),
            ),
            "fallback_endpoint_ms": getattr(cfg.vad, "min_silence_ms", None),
            "tts_backend": getattr(cfg.tts, "backend", "csm"),
            "llm_model": cfg.llm.model,
            "instant_ack_cache": bool(
                getattr(getattr(cfg, "pipeline", None),
                        "instant_ack_pcm", None) is not None
            ),
            "first_chunk_max_words": getattr(getattr(cfg, "pipeline", None), "first_chunk_max_words", None),
            "first_chunk_min_words": getattr(getattr(cfg, "pipeline", None), "first_chunk_min_words", None),
        },
        "runs": {
            "asr": asr_runs,
            "llm": llm_runs,
            "tts": tts_runs,
            "model_first_audio": combined_runs,
        },
        "summary": {
            "asr_ms": _summary([r["ms"] for r in asr_runs if r["ms"] is not None]),
            "llm_first_chunk_ms": _summary([r["ms"] for r in llm_runs if r["ms"] is not None]),
            "tts_first_chunk_ms": _summary([r["first_chunk_ms"] for r in tts_runs if r["first_chunk_ms"] is not None]),
            "tts_rtf": _summary([r["rtf"] for r in tts_runs if r["rtf"] is not None]),
            "max_inter_chunk_gap_ms": _summary([r["max_gap_ms"] for r in tts_runs if r["max_gap_ms"] is not None]),
            "model_first_audio_ms": _summary(
                [r["model_first_audio_ms"] for r in combined_runs if r["model_first_audio_ms"] is not None]
            ),
        },
    }
    asr_median = results["summary"]["asr_ms"]["median"] or 0
    first_audio_median = results["summary"]["model_first_audio_ms"]["median"] or 0
    results["summary"]["estimated_user_stop_to_audio_ms"] = round(
        results["config"]["endpoint_ms"] + asr_median + first_audio_median, 1
    )
    results["evaluation"] = evaluate(results, args.profile)

    args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    await llm.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
