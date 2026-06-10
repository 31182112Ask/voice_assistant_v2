from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.config import load_config


REQUIRED_MODULES = [
    "fastapi",
    "uvicorn",
    "yaml",
    "httpx",
    "transformers",
    "accelerate",
    "soundfile",
    "librosa",
    "faster_whisper",
    "silero_vad",
    "numpy",
    "torch",
    "torchaudio",
]


def require_modules() -> None:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"Missing Python modules: {', '.join(missing)}")


def main() -> None:
    require_modules()
    cfg = load_config()

    import torch
    import torchaudio
    import transformers

    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__} CUDA={torch.version.cuda}")
    print(f"Torchaudio: {torchaudio.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"LLM: {cfg.llm.model} at {cfg.llm.base_url}")
    print(f"TTS: {cfg.tts.model_id} local_files_only={cfg.tts.local_files_only}")
    print("Deployment prerequisites: OK")


if __name__ == "__main__":
    main()
