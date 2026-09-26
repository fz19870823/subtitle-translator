"""OpenAI 兼容 Chat Completions 翻译后端。

适用于任何暴露 ``POST {base_url}/chat/completions`` 的服务：
OpenAI、各类中转/聚合网关（含 grok2api 这类自建中继）、以及本地推理服务。

只依赖标准库 urllib，避免为一个 HTTP 调用引入额外依赖。
"""
from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence

from app.config import TranslationConfig
from app.core.translator import (
    TranslationError,
    TranslationRequest,
    Translator,
    looks_like_verbatim_echo,
    register,
)

# 从模型回复里抠出 JSON 数组：优先整段解析，失败再退回首个 [...] 片段。
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

#: 用可见记号传递换行。让模型理解 JSON 字符串里的 \n 是换行并原样保留，
#: 实测不可靠（模型会把两行合并成一行）。换成一个看得见的字符后稳定得多。
_LINE_MARK = "\u23ce"  # ⏎

_SYSTEM_PROMPT = (
    "你是一名专业的字幕翻译引擎。用户会给你一个 JSON 数组，"
    "请把数组里的每个字符串翻译成{target}，逐元素一一对应。\n"
    "硬性要求：\n"
    "1. 只输出一个 JSON 数组，元素是翻译后的字符串，数量必须与输入完全一致；\n"
    "2. 不要输出任何解释、编号、Markdown 代码块或额外文字；\n"
    "3. 字符串里的 {mark} 是「换行」记号，必须在译文中对应的位置原样保留，"
    "既不要删掉也不要新增；\n"
    "4. 保留原文中形如 <i>、</i> 的标签；\n"
    "5. 字幕要口语、简洁，不要合并或拆分数组元素。"
)


def explain_http_error(detail: str) -> str:
    """把服务端的错误体压成一行可读信息，便于定位。"""
    try:
        err = json.loads(detail).get("error")
    except (json.JSONDecodeError, AttributeError):
        return detail.strip() or "(无响应体)"
    if isinstance(err, dict):
        code = err.get("code") or err.get("type") or "?"
        return f"[{code}] {err.get('message', '')}".strip()
    return str(err) if err else detail.strip()


def extract_model_ids(payload: Any) -> List[str]:
    """从 /models 响应里取出模型 id。

    兼容 ``{"data": [{"id": ...}]}``（OpenAI 标准）、``{"models": [...]}``
    以及直接给字符串数组的几种写法。
    """
    if not isinstance(payload, dict):
        entries = payload if isinstance(payload, list) else []
    else:
        entries = payload.get("data") or payload.get("models") or []

    ids: List[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            value = entry.get("id") or entry.get("name")
        else:
            value = entry
        if isinstance(value, str) and value and value not in ids:
            ids.append(value)
    return ids


def fetch_models(base_url: str, api_key: str, *, timeout: int = 30) -> List[str]:
    """``GET {base_url}/models``，返回模型 id 列表。

    独立成函数（而不是 translator 的方法），这样界面在用户还没选定模型时
    也能拉列表 —— 拉列表本来就不需要 model 参数。
    """
    if not base_url:
        raise TranslationError("base_url 未配置")
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "subtitle-translator/0.1")
    try:
        with urllib.request.urlopen(
            req, timeout=int(timeout), context=ssl.create_default_context()
        ) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise TranslationError(
            f"拉取模型列表失败: HTTP {exc.code} —— {explain_http_error(detail)}"
        ) from None
    except urllib.error.URLError as exc:
        raise TranslationError(f"连接 {url} 失败: {exc.reason}") from None
    except TimeoutError:
        raise TranslationError(f"拉取模型列表超时（{timeout}s）") from None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise TranslationError("模型列表响应不是合法 JSON") from None
    return extract_model_ids(payload)


class OpenAICompatTranslator(Translator):
    """通过 OpenAI 兼容接口做字幕翻译。"""

    name = "openai"

    #: 需要 base_url / 密钥 / 模型这些外部参数，界面据此决定是否启用模型选择。
    requires_api = True

    #: 检测到整批原样回抄时的额外重试次数（共尝试 echo_retries + 1 次）。
    #: 实测某些中继会把请求路由到能力不足的上游，把整个数组原样吐回来，
    #: 且是间歇性的 —— 不重试就会静默交付一份没翻译的字幕。
    echo_retries = 2

    def __init__(
        self,
        *,
        base_url: str = "",
        model: str = "",
        api_key: str = "",
        timeout: int = 120,
        temperature: float = 0.0,
        style_hint: str = "",
        batch_size: int = 20,
        preserve_line_breaks: bool = True,
        echo_retries: int | None = None,
    ) -> None:
        if not base_url:
            raise TranslationError("base_url 未配置")
        if not model:
            raise TranslationError("model 未配置")
        if not api_key:
            raise TranslationError("api_key 未配置")
        if int(batch_size) <= 0:
            # 配置是人工写的，写错了要报错，不要静默兜底成一个别的值。
            raise TranslationError(f"batch_size 必须为正整数，收到 {batch_size!r}")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = int(timeout)
        self.temperature = float(temperature)
        self.style_hint = style_hint
        self.batch_size = int(batch_size)
        self.preserve_line_breaks = bool(preserve_line_breaks)
        if echo_retries is not None:
            self.echo_retries = max(0, int(echo_retries))
        self._api_key = api_key  # 私有：不参与 repr，也不写进日志
        #: 因模型丢换行而被单独重译的条目数，便于观测提示词是否退化
        self.line_repair_count = 0
        #: 因整批原样回抄而重发的批次数
        self.echo_retry_count = 0

    # 防止密钥经由 repr/日志外泄
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"model={self.model!r}, api_key=<hidden>)"
        )

    # ------------------------------------------------------------------ 配置工厂
    @classmethod
    def from_config(
        cls,
        cfg: TranslationConfig,
        *,
        engine_name: str | None = None,
    ) -> "OpenAICompatTranslator":
        """从 :class:`~app.config.TranslationConfig` 构造实例。"""
        return cls(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.resolve_api_key(),
            timeout=cfg.timeout,
            temperature=cfg.temperature,
            style_hint=cfg.style_hint,
            batch_size=cfg.batch_size,
            preserve_line_breaks=cfg.preserve_line_breaks,
        )

    # ------------------------------------------------------------------ HTTP
    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")
        req.add_header("User-Agent", "subtitle-translator/0.1")

        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout, context=ssl.create_default_context()
            ) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise TranslationError(
                f"HTTP {exc.code} from {url} —— {explain_http_error(detail)}"
            ) from None
        except urllib.error.URLError as exc:
            raise TranslationError(f"连接 {url} 失败: {exc.reason}") from None
        except TimeoutError:
            raise TranslationError(f"请求 {url} 超时（{self.timeout}s）") from None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TranslationError(f"响应不是合法 JSON（前 300 字符）: {raw[:300]}") from None
        if not isinstance(payload, dict):
            raise TranslationError(f"响应顶层不是 JSON 对象: {raw[:300]}")
        return payload

    def list_models(self) -> List[str]:
        """拉取可用模型 id，用于配置自检。"""
        return fetch_models(self.base_url, self._api_key, timeout=min(self.timeout, 60))

    # ------------------------------------------------------------------ 翻译
    def _system_prompt(self, target_lang: str, *, strict: bool = False) -> str:
        target = target_lang or "中文"
        prompt = _SYSTEM_PROMPT.format(target=target, mark=_LINE_MARK)
        if self.style_hint:
            prompt += f"\n6. 额外要求：{self.style_hint}"
        if strict:
            # 上一次整批被原样退回，再问一次时把要求说死，别让对方继续偷懒。
            prompt += (
                "\n\n【重试】上一次回答把原文原样返回了，等于没有翻译。"
                f"这次请逐条改写成{target}，"
                "不得原样返回原文，也不得只改标点、空格或语序了事。"
            )
        return prompt

    def _encode(self, text: str) -> str:
        """送出前把换行换成可见记号。"""
        if not self.preserve_line_breaks:
            return text
        return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", _LINE_MARK)

    def _decode(self, text: str) -> str:
        """收回来后把记号还原成换行。"""
        if not self.preserve_line_breaks:
            return text
        # 模型偶尔会在记号两侧加空格，一并吃掉
        return re.sub(rf"\s*{re.escape(_LINE_MARK)}\s*", "\n", text).strip()

    def _chat(self, messages: Sequence[Dict[str, str]], *, max_tokens: int = 4096) -> str:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "stream": False,
            # 注意：部分中继会忽略 max_tokens，这里只作为"建议上限"。
            "max_tokens": max(256, max_tokens),
        }
        payload = self._post("/chat/completions", body)

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TranslationError(f"响应缺少 choices: {json.dumps(payload, ensure_ascii=False)[:300]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise TranslationError("响应 message.content 不是字符串")
        return content

    @staticmethod
    def _parse_array(content: str, expected: int) -> List[str] | None:
        """把模型回复解析成长度为 expected 的字符串数组；失败返回 None。"""
        text = _FENCE_RE.sub("", content.strip()).strip()

        def coerce(value: Any) -> List[str] | None:
            if isinstance(value, dict):
                # 容忍 {"translations": [...]} 这类包装
                for candidate in value.values():
                    if isinstance(candidate, list):
                        value = candidate
                        break
                else:
                    return None
            if not isinstance(value, list):
                return None
            if not all(isinstance(v, str) for v in value):
                return None
            return value if len(value) == expected else None

        try:
            parsed = coerce(json.loads(text))
            if parsed is not None:
                return parsed
        except json.JSONDecodeError:
            pass

        match = _ARRAY_RE.search(text)
        if match:
            try:
                parsed = coerce(json.loads(match.group(0)))
                if parsed is not None:
                    return parsed
            except json.JSONDecodeError:
                pass
        return None

    def translate_batch(self, requests: Sequence[TranslationRequest]) -> List[str]:
        if not requests:
            return []

        source_lang = requests[0].source_lang
        target_lang = requests[0].target_lang

        # 空串不送模型，原样返回，省 token 也避免模型"补一句"。
        results: List[str] = ["" for _ in requests]
        todo = [(i, r.text) for i, r in enumerate(requests) if r.text.strip()]
        if not todo:
            return results

        texts = [self._encode(t) for _, t in todo]

        parsed: List[str] = []
        for attempt in range(self.echo_retries + 1):
            parsed = self._request_batch(
                texts, source_lang, target_lang, strict=attempt > 0
            )
            if not looks_like_verbatim_echo(texts, parsed, source_lang, target_lang):
                break
            if attempt < self.echo_retries:
                self.echo_retry_count += 1
        else:
            raise TranslationError(
                f"连续 {self.echo_retries + 1} 次拿到的译文与原文完全相同，"
                f"{source_lang} -> {target_lang} 未真正执行翻译。"
                "这通常是上游把请求路由到了不支持跨语言翻译的通道，"
                "稍后重试或换一个模型即可。"
            )

        for (index, original), translation in zip(todo, parsed):
            fixed = self._decode(translation)
            # 记号被模型吃掉时兜底：按原文的行结构重译，保证行数不丢。
            if self.preserve_line_breaks and "\n" in original and "\n" not in fixed:
                repaired = self._translate_lines(original, source_lang, target_lang)
                if repaired is not None:
                    self.line_repair_count += 1
                    fixed = repaired
            results[index] = fixed
        return results

    def _request_batch(
        self,
        texts: Sequence[str],
        source_lang: str,
        target_lang: str,
        *,
        strict: bool = False,
    ) -> List[str]:
        """发一次批量请求并解析；批量协议被破坏时退回逐条。"""
        user_content = json.dumps(list(texts), ensure_ascii=False)
        if source_lang and source_lang != "auto":
            user_content = f"源语言: {source_lang}\n{user_content}"

        content = self._chat(
            [
                {
                    "role": "system",
                    "content": self._system_prompt(target_lang, strict=strict),
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=min(8192, 256 + 120 * len(texts)),
        )

        parsed = self._parse_array(content, len(texts))
        if parsed is None:
            # 批量协议被破坏（模型加了解释/数量不符）时退回逐条，宁可慢也不丢内容。
            parsed = [self._translate_one(t, source_lang, target_lang) for t in texts]
        return parsed

    def _translate_lines(
        self, text: str, source_lang: str, target_lang: str
    ) -> str | None:
        """按行重译并保留换行结构；无法完成时返回 None（调用方保留原译文）。"""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        keep = [i for i, line in enumerate(lines) if line.strip()]
        if len(keep) < 2:
            return None

        payload = [lines[i] for i in keep]
        content = self._chat(
            [
                {"role": "system", "content": self._system_prompt(target_lang)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=min(4096, 128 + 120 * len(payload)),
        )
        parsed = self._parse_array(content, len(payload))
        if parsed is None:
            parsed = [self._translate_one(line, source_lang, target_lang) for line in payload]

        rebuilt = list(lines)
        for index, translation in zip(keep, parsed):
            rebuilt[index] = translation.strip()
        return "\n".join(rebuilt)

    def _translate_one(self, text: str, source_lang: str, target_lang: str) -> str:
        content = self._chat(
            [
                {"role": "system", "content": self._system_prompt(target_lang)},
                {"role": "user", "content": json.dumps([self._encode(text)], ensure_ascii=False)},
            ],
            max_tokens=1024,
        )
        parsed = self._parse_array(content, 1)
        if parsed is not None:
            return self._decode(parsed[0])
        # 单条也解析不出就退到纯文本：去掉可能的代码围栏与首尾引号。
        cleaned = _FENCE_RE.sub("", content.strip()).strip().strip('"').strip()
        return self._decode(cleaned) or text


def build_from_config(cfg: TranslationConfig) -> OpenAICompatTranslator:
    """注册表工厂：让 ``create_engine("openai", config=cfg)`` 可用。"""
    return OpenAICompatTranslator.from_config(cfg)


register(OpenAICompatTranslator)
