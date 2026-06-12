"""一鍵預下載全部模型 (避免首次啟動時等待)。

用法:
  huggingface-cli login        # CSM-1B 為 gated 模型, 需先在 HF 頁面接受授權
  python scripts/download_models.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()

    print(f"[1/3] TTS model: {cfg.tts.model_id} (需已在 HF 接受授權)")
    from huggingface_hub import snapshot_download
    snapshot_download(cfg.tts.model_id)

    print(f"[2/3] faster-whisper {cfg.asr.model}")
    from faster_whisper import WhisperModel
    WhisperModel(cfg.asr.model, device=cfg.asr.device, compute_type=cfg.asr.compute_type)

    print(f"[3/3] Ollama LLM ({cfg.llm.model})")
    try:
        subprocess.run(["ollama", "pull", cfg.llm.model], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  ! 未找到 ollama, 請先安裝: https://ollama.com 然後執行:")
        print(f"    ollama pull {cfg.llm.model}")
        sys.exit(1)

    print("\n全部就緒。啟動服務: python -m server.main")


def cosyvoice_hint() -> None:
    print("\nTTS (CosyVoice3): 運行 scripts/setup_cosyvoice.ps1 完成"
          "倉庫克隆+依賴+權重 (FunAudioLLM/Fun-CosyVoice3-0.5B-2512)")


if __name__ == "__main__":
    main()
    cosyvoice_hint()
