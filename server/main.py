"""voice_assistant_v2 服務端入口。

啟動:  python -m server.main   (或 uvicorn server.main:app)
頁面:  http://localhost:8000
協議:  ws://localhost:8000/ws
  ↑ 上行  binary = PCM16 mono 16kHz 麥克風幀 | text = JSON 控制
  ↓ 下行  binary = PCM16 mono 24kHz 合成音頻 | text = JSON 事件
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .config import load_config
from .pipeline.asr import WhisperASR
from .pipeline.llm import OllamaLLM
from .pipeline.orchestrator import Session
from .pipeline.tts import CSMSynthesizer
from .utils.audio import pcm16_to_float32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

ROOT = pathlib.Path(__file__).resolve().parent.parent
cfg = load_config()
app = FastAPI(title="voice_assistant_v2")

asr: WhisperASR | None = None
llm: OllamaLLM | None = None
tts: CSMSynthesizer | None = None


@app.on_event("startup")
async def startup() -> None:
    global asr, llm, tts
    log.info("=== voice_assistant_v2 booting ===")
    if getattr(cfg.tts, "local_files_only", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    asr = WhisperASR(
        model=cfg.asr.model, device=cfg.asr.device,
        compute_type=cfg.asr.compute_type,
        language=cfg.asr.language, beam_size=cfg.asr.beam_size,
        warmup=getattr(cfg.asr, "warmup", True),
    )
    llm = OllamaLLM(
        base_url=cfg.llm.base_url, model=cfg.llm.model,
        system_prompt=cfg.llm.system_prompt,
        max_tokens=cfg.llm.max_tokens, temperature=cfg.llm.temperature,
        think=getattr(cfg.llm, "think", None),
        num_gpu=getattr(cfg.llm, "num_gpu", None),
        num_batch=getattr(cfg.llm, "num_batch", None),
        keep_alive=getattr(cfg.llm, "keep_alive", None),
    )
    await llm.warmup()

    backend = getattr(cfg.tts, "backend", "csm")
    if backend == "kyutai":
        try:
            from .pipeline.tts_kyutai import KyutaiTTS
            kcfg = getattr(cfg.tts, "kyutai", None)
            tts = await asyncio.to_thread(
                KyutaiTTS,
                cfg.tts.device,
                getattr(kcfg, "voice",
                        "expresso/ex03-ex01_happy_001_channel1_334s.wav"),
                getattr(kcfg, "temp", 0.6),
                getattr(kcfg, "cfg_coef", 2.0),
                getattr(kcfg, "n_q", 16),
                getattr(kcfg, "first_emit_frames", 2),
                getattr(kcfg, "batch_frames", 4),
                getattr(kcfg, "local_files_only",
                        getattr(cfg.tts, "local_files_only", False)),
            )
        except Exception as e:  # noqa: BLE001
            log.error("Kyutai 後端加載失敗, 回退到 CSM: %s", e)
            backend = "csm"
    if backend == "csm":
        tts = await asyncio.to_thread(
            CSMSynthesizer,
            cfg.tts.model_id, cfg.tts.device, cfg.tts.dtype,
            cfg.tts.compile_decoder, cfg.tts.max_audio_len_ms,
            cfg.tts.context_turns, cfg.tts.voice_prompt, ROOT,
            getattr(cfg.tts, "local_files_only", False),
            getattr(cfg.tts, "max_context_audio_s", 3.0),
        )
    log.info("TTS backend = %s", backend)
    log.info("=== all models ready, open http://localhost:%d ===",
             cfg.server.port)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    if tts is None:
        await ws.send_text(json.dumps(
            {"type": "error", "message": "models still loading"}))
        await ws.close()
        return

    session = Session(
        cfg, asr, llm, tts,
        send_json=lambda m: ws.send_text(json.dumps(m, ensure_ascii=False)),
        send_bytes=ws.send_bytes,
    )
    await ws.send_text(json.dumps({
        "type": "ready",
        "input_sr": cfg.audio.input_sample_rate,
        "output_sr": cfg.audio.output_sample_rate,
    }))
    greet_task = asyncio.create_task(session.greet())
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                await session.on_audio(pcm16_to_float32(msg["bytes"]))
            elif msg.get("text") is not None:
                await session.on_control(json.loads(msg["text"]))
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        greet_task.cancel()
        await session._interrupt(reason="disconnect", notify=False)
        log.info("session closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host=cfg.server.host,
                port=cfg.server.port, log_level="info")
