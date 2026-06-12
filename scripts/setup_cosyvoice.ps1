# Setup CosyVoice3 streaming TTS.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_cosyvoice.ps1
#
# This script:
#   1. Clones FunAudioLLM/CosyVoice into third_party/CosyVoice.
#   2. Installs CosyVoice dependencies into the active Python environment,
#      excluding torch and torchaudio so the existing CUDA build is preserved.
#   3. Downloads Fun-CosyVoice3-0.5B-2512 weights into pretrained_models/.

param(
    [string]$RepoDir = "third_party/CosyVoice",
    [string]$ModelDir = "pretrained_models/Fun-CosyVoice3-0.5B",
    [string]$ModelRepo = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git was not found in PATH."
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python was not found in PATH. Activate .venv first."
}

New-Item -ItemType Directory -Force "third_party" | Out-Null
New-Item -ItemType Directory -Force "pretrained_models" | Out-Null

if (-not (Test-Path $RepoDir)) {
    Write-Host "Cloning CosyVoice into $RepoDir ..."
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $RepoDir
} else {
    Write-Host "$RepoDir already exists; skipping clone."
}

$requirements = Join-Path $RepoDir "requirements.txt"
if (-not (Test-Path $requirements)) {
    throw "Missing requirements file: $requirements"
}

$noVendorRequirements = Join-Path $RepoDir "requirements.novendor.txt"
Get-Content $requirements |
    Where-Object { $_ -notmatch "^(?i:torch|torchaudio|grpcio|openai-whisper)\b" } |
    Set-Content $noVendorRequirements -Encoding UTF8

Write-Host "Installing CosyVoice Python dependencies ..."
python -m pip install -r $noVendorRequirements
python -m pip install "grpcio>=1.62"
python -m pip install --no-build-isolation "openai-whisper==20231117"
python -m pip install --upgrade "onnxruntime-gpu>=1.26.0"

Write-Host "Downloading model weights: $ModelRepo -> $ModelDir"
python -c @"
from huggingface_hub import snapshot_download
snapshot_download("$ModelRepo", local_dir="$ModelDir", local_dir_use_symlinks=False)
print("weights ready")
"@

Write-Host ""
Write-Host "CosyVoice3 setup complete."
Write-Host "Expected config:"
Write-Host "  tts.backend: cosyvoice3"
Write-Host "  tts.cosyvoice.repo_dir: $RepoDir"
Write-Host "  tts.cosyvoice.model_dir: $ModelDir"
Write-Host ""
Write-Host "Optional voice prompt:"
Write-Host "  voices/ref.wav  (10-30s clean speech)"
Write-Host "  voices/ref.txt  (matching transcript)"
