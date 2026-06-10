"""LLM 串流客戶端 (Ollama)。

核心: stream_sentences() 把 token 流即時切成「可朗讀的句子」,
讓 TTS 不必等整段回覆生成完 —— 這是壓低首音延遲的關鍵。
"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator

import httpx

log = logging.getLogger("llm")

# 句子邊界: 句末標點後跟空白/結尾。最短句長避免 "Hi." 之類碎片過多佔用 TTS 啟動開銷
_SENT_RE = re.compile(r"(.+?[.!?…]['\")\]]?)(?:\s+|$)", re.S)
_MIN_SENT_CHARS = 12


class OllamaLLM:
    def __init__(self, base_url: str, model: str, system_prompt: str,
                 max_tokens: int, temperature: float,
                 think: bool | None = None,
                 num_gpu: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.think = think
        self.num_gpu = num_gpu
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5))

    async def warmup(self) -> None:
        """觸發 Ollama 加載模型進顯存/內存。"""
        try:
            await self._collect([{"role": "user", "content": "Hi"}], max_tokens=4)
            log.info("LLM '%s' warmed up.", self.model)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM warmup failed (is Ollama running?): %s", e)

    async def stream_tokens(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt}, *messages],
            "stream": True,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        if self.think is not None:
            payload["think"] = self.think
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

    async def stream_sentences(self, messages: list[dict]) -> AsyncIterator[str]:
        """將 token 流切分為句子流。取消安全: 外層 task.cancel() 即停止生成。"""
        buf = ""
        async for token in self.stream_tokens(messages):
            buf += token
            while True:
                m = _SENT_RE.match(buf)
                if not m:
                    break
                candidate = m.group(1).strip()
                rest = buf[m.end():]
                # 句子太短且後面還有內容 → 與下一句合併, 減少 TTS 碎片
                if len(candidate) < _MIN_SENT_CHARS and rest:
                    break
                buf = rest
                if candidate:
                    yield candidate
        tail = buf.strip()
        if tail:
            yield tail

    async def _collect(self, messages: list[dict], max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        if self.think is not None:
            payload["think"] = self.think
        r = await self.client.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
