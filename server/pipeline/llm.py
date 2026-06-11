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
            "options": self._options(),
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
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

    async def stream_speakable_chunks(
        self, messages: list[dict],
        first_chunk_max_words: int = 9,
        first_chunk_min_words: int = 3,
    ) -> AsyncIterator[str]:
        """壓低首音延遲的核心: 首塊不等整句。

        階段 1 (首塊): token 一邊到一邊掃描, 只要滿足任一條件立刻產出:
            a) 遇到句末標點         b) 遇到子句邊界 (,;:—) 且 ≥ min_words
            c) 詞數達到 max_words (在最後一個空格處截斷)
          → CSM 只需生成 ~1 秒音頻即可開播, 而不是等 3-5 秒的整句。
        階段 2 (其後): 退回逐句模式, 在首塊播放期間並行合成, 聽感無縫。
        """
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
        """在 token 流前綴上找最早的可朗讀斷點。返回 (首塊|None, 剩餘)。"""
        words = buf.split()
        # a) 句末標點
        m = _SENT_RE.match(buf)
        if m and len(m.group(1).split()) >= 1:
            return m.group(1).strip(), buf[m.end():]
        # b) 子句邊界
        cm = re.search(r"[,;:—–]\s", buf)
        if cm:
            head = buf[: cm.end()].strip()
            if len(head.split()) >= min_words:
                return head, buf[cm.end():]
        # c) 詞數達標 (保留最後一個可能不完整的詞在剩餘部分)
        if len(words) > max_words:
            head = " ".join(words[:max_words])
            idx = buf.find(head) + len(head)
            return head.strip(), buf[idx:]
        return None, buf

    async def plan_followup(self, history: list[dict],
                            plan_prompt: str,
                            max_tokens: int = 60) -> str:
        """讓 LLM 在回覆結束後自主排程下一句主動發話。

        返回原始單行計劃文本, 由調用方解析:
          "NONE"                          → 不排程
          "WAIT=18 | Still mulling it over?" → 18 秒後說這句
        """
        msgs = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": plan_prompt},
        ]
        return (await self._collect(msgs, max_tokens=max_tokens)).strip()

    async def _collect(self, messages: list[dict], max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._options(max_tokens),
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        r = await self.client.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
