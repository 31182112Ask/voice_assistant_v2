"""LLM 串流客戶端 — 雙後端。

  llamacpp / vllm  → OpenAICompatLLM (/v1/chat/completions, SSE 流)
  ollama           → OllamaLLM      (/api/chat, NDJSON 流)

平台選型 (RTX 4060 8GB, Windows, 與 Kyutai TTS 共享顯存):
  - llama.cpp llama-server: Windows 原生、TTFT 低、`-ngl`/`-c` 顯存
    粒度可控 —— 本項目默認 ★
  - vLLM: 吞吐王者但 Linux/WSL 限定, 且默認預佔 90% 顯存做 KV cache,
    與 TTS 同卡時必須調低 gpu_memory_utilization; 單流小模型場景
    對比 llama.cpp 無優勢
  - Ollama: 易用但多一層調度開銷, 首 token 略慢

共享核心: stream_speakable_chunks() 把 token 流切成「可朗讀的塊」——
首塊極短 (壓首音延遲), 其後逐句。
"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator

import httpx

log = logging.getLogger("llm")

# 句子邊界: 拉丁句末標點需後跟空白/結尾; CJK 句末標點 (。！？；) 後無空格,
# 即時成立。最短句長避免碎片過多佔用 TTS 啟動開銷。
_SENT_RE = re.compile(
    r"(.+?(?:[.!?…]['\")\]]?(?=\s|$)|[。！？；]['\")\]」』]?))\s*", re.S)
_MIN_SENT_CHARS = 12
# 子句邊界 (首塊提前切出用): 拉丁逗號類 + CJK 頓逗冒
_CLAUSE_RE = re.compile(r"(?:[,;:—–]\s|[，、；：])")
# 無標點保險: 連續超過此字符數仍無邊界 → 強制切 (主要服務 CJK 長句)
_FIRST_CHUNK_MAX_CHARS = 30


class BaseLLM:
    """子類需實現 stream_tokens() 與 _collect()。"""

    system_prompt: str = ""

    # ---------------- 共享: 切塊邏輯 ----------------
    async def stream_sentences(self, messages: list[dict]) -> AsyncIterator[str]:
        buf = ""
        async for token in self.stream_tokens(messages):
            buf += token
            while True:
                m = _SENT_RE.match(buf)
                if not m:
                    break
                candidate = m.group(1).strip()
                rest = buf[m.end():]
                if len(candidate) < _MIN_SENT_CHARS and rest:
                    break
                buf = rest
                if candidate:
                    yield candidate
        tail = buf.strip()
        if tail:
            yield tail

    async def stream_speakable_chunks(
        self, messages: list[dict],
        first_chunk_max_words: int = 9,
        first_chunk_min_words: int = 2,
    ) -> AsyncIterator[str]:
        """首塊不等整句: 句末標點 / 子句邊界 / 詞數達標, 最早者勝。"""
        buf = ""
        first_done = False
        async for token in self.stream_tokens(messages):
            buf += token
            if not first_done:
                chunk, rest = self._cut_first_chunk(
                    buf, first_chunk_max_words, first_chunk_min_words)
                if chunk is not None:
                    first_done = True
                    buf = rest
                    yield chunk
                continue
            while True:
                m = _SENT_RE.match(buf)
                if not m:
                    break
                candidate = m.group(1).strip()
                rest = buf[m.end():]
                if len(candidate) < _MIN_SENT_CHARS and rest:
                    break
                buf = rest
                if candidate:
                    yield candidate
        tail = buf.strip()
        if tail:
            yield tail

    @staticmethod
    def _cut_first_chunk(buf: str, max_words: int,
                         min_words: int) -> tuple[str | None, str]:
        words = buf.split()
        m = _SENT_RE.match(buf)
        if m and len(m.group(1).split()) >= 1:
            return m.group(1).strip(), buf[m.end():]
        cm = _CLAUSE_RE.search(buf)
        if cm:
            head = buf[: cm.end()].strip()
            # CJK 子句按字符數放行 (split() 對中文無效)
            if len(head.split()) >= min_words or len(head) >= 2:
                return head, buf[cm.end():]
        if len(words) > max_words:
            head = " ".join(words[:max_words])
            idx = buf.find(head) + len(head)
            return head.strip(), buf[idx:]
        # 無標點保險: CJK 長句 / 無標點英文流, 超長即強制切
        if len(buf) > _FIRST_CHUNK_MAX_CHARS:
            return buf[:_FIRST_CHUNK_MAX_CHARS].strip(), buf[_FIRST_CHUNK_MAX_CHARS:]
        return None, buf

    # ---------------- 共享: 主動性接口 ----------------
    async def decide_proactive(self, history: list[dict],
                               decision_prompt: str,
                               max_tokens: int = 48) -> str:
        """Duplex 決策: SAY=<句子> / WAIT=<秒> / SLEEP, 單行輸出。"""
        msgs = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": decision_prompt},
        ]
        return (await self._collect(msgs, max_tokens=max_tokens)).strip()

    async def plan_followup(self, history: list[dict],
                            plan_prompt: str,
                            max_tokens: int = 60) -> str:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": plan_prompt},
        ]
        return (await self._collect(msgs, max_tokens=max_tokens)).strip()

    async def warmup(self) -> None:
        try:
            await self._collect([{"role": "user", "content": "Hi"}],
                                max_tokens=4)
            log.info("LLM warmed up.")
        except Exception as e:  # noqa: BLE001
            log.warning("LLM warmup failed (is the server running?): %s", e)

    # 子類實現
    async def stream_tokens(self, messages):  # pragma: no cover
        raise NotImplementedError
        yield ""

    async def _collect(self, messages, max_tokens):  # pragma: no cover
        raise NotImplementedError


# ════════════════ OpenAI 兼容後端 (llama.cpp / vLLM) ════════════════
class OpenAICompatLLM(BaseLLM):
    """llama-server (llama.cpp) 與 vLLM 共用的 OpenAI 兼容客戶端。

    - SSE 流式 /v1/chat/completions
    - chat_template_kwargs.enable_thinking=false:
      llama.cpp 與 vLLM 均支持, 用於關閉 Qwen3 系思考段
    """

    def __init__(self, base_url: str, model: str, system_prompt: str,
                 max_tokens: int, temperature: float,
                 think: bool | None = None, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.think = think
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=5), headers=headers)

    def _payload(self, messages: list[dict], stream: bool,
                 max_tokens: int | None = None) -> dict:
        chat_messages = (
            messages
            if messages and messages[0].get("role") == "system"
            else [{"role": "system", "content": self.system_prompt}, *messages]
        )
        payload = {
            "model": self.model,
            "messages": chat_messages,
            "stream": stream,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature,
        }
        if self.think is False:
            # llama.cpp ≥ b5400 / vLLM ≥ 0.8 均接受; 舊版會忽略該鍵
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    async def stream_tokens(self, messages: list[dict]) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST", f"{self.base_url}/v1/chat/completions",
            json=self._payload(messages, stream=True),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                token = delta.get("content") or ""
                if token:
                    yield token

    async def _collect(self, messages: list[dict], max_tokens: int) -> str:
        r = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=self._payload(messages, stream=False, max_tokens=max_tokens),
        )
        r.raise_for_status()
        msg = (r.json().get("choices") or [{}])[0].get("message", {})
        return msg.get("content") or ""


# ════════════════ Ollama 後端 (保留兼容) ════════════════
class OllamaLLM(BaseLLM):
    def __init__(self, base_url: str, model: str, system_prompt: str,
                 max_tokens: int, temperature: float,
                 think: bool | None = None,
                 num_gpu: int | None = None,
                 num_batch: int | None = None,
                 keep_alive: int | str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.think = think
        self.num_gpu = num_gpu
        self.num_batch = num_batch
        self.keep_alive = keep_alive
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5))

    def _options(self, num_predict: int | None = None) -> dict:
        options = {
            "num_predict": self.max_tokens if num_predict is None else num_predict,
            "temperature": self.temperature,
        }
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        if self.num_batch is not None:
            options["num_batch"] = self.num_batch
        return options

    def _decorate(self, payload: dict) -> dict:
        if self.think is not None:
            payload["think"] = self.think
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    async def stream_tokens(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = self._decorate({
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt},
                         *messages],
            "stream": True,
            "options": self._options(),
        })
        async with self.client.stream(
            "POST", f"{self.base_url}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("done"):
                    break
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token

    async def _collect(self, messages: list[dict], max_tokens: int) -> str:
        payload = self._decorate({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._options(max_tokens),
        })
        r = await self.client.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")


def create_llm(cfg) -> BaseLLM:
    """按 config.llm.backend 構造後端。llamacpp 與 vllm 走同一客戶端。"""
    backend = getattr(cfg, "backend", "ollama").lower()
    if backend in ("llamacpp", "llama.cpp", "llama_cpp", "vllm", "openai"):
        llm = OpenAICompatLLM(
            base_url=cfg.base_url, model=cfg.model,
            system_prompt=cfg.system_prompt,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
            think=getattr(cfg, "think", None),
            api_key=getattr(cfg, "api_key", None),
        )
        log.info("LLM backend: %s (OpenAI-compatible @ %s)",
                 backend, cfg.base_url)
        return llm
    llm = OllamaLLM(
        base_url=cfg.base_url, model=cfg.model,
        system_prompt=cfg.system_prompt,
        max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        think=getattr(cfg, "think", None),
        num_gpu=getattr(cfg, "num_gpu", None),
        num_batch=getattr(cfg, "num_batch", None),
        keep_alive=getattr(cfg, "keep_alive", None),
    )
    log.info("LLM backend: ollama @ %s", cfg.base_url)
    return llm
