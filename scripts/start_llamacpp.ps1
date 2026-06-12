# 啟動 llama.cpp 推理服務 (llama-server, OpenAI 兼容)
# 用法:  .\scripts\start_llamacpp.ps1 -Model "C:\models\qwen3.5-2b-instruct-q4_k_m.gguf"
#
# 安裝 llama.cpp (Windows, CUDA):
#   winget install ggml.llamacpp        # 或從 github.com/ggml-org/llama.cpp/releases
#                                       # 下載 cudart 版 zip 解壓
#
# 旗標說明 (RTX 4060 8GB, 與 Kyutai TTS 同卡):
#   -ngl 99       全部層上 GPU (2B Q4 約 1.4GB)
#   -c 4096       上下文窗口; 語音對話足夠, KV cache 顯存可控 (~0.3GB)
#   -fa           flash attention, 首 token 更快
#   --no-mmap     模型直接載入內存 (64GB RAM), 避免按需分頁的首輪抖動
#   -t 8          CPU 線程 (僅 CPU 回退時相關)
#   --port 8080   與 config.yaml 的 llm.base_url 對應
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [int]$Port = 8080,
    [int]$Ngl = 99,
    [int]$Ctx = 4096
)
$ErrorActionPreference = "Stop"
llama-server -m $Model -ngl $Ngl -c $Ctx -fa on --no-mmap --port $Port `
    --host 127.0.0.1 --log-disable
