"""OpenAI 兼容 Chat Completions 翻译后端。

适用于任何暴露 ``POST {base_url}/chat/completions`` 的服务：
OpenAI、各类中转/聚合网关（含 grok2api 这类自建中继）、以及本地推理服务。

只依赖标准库 urllib，避免为一个 HTTP 调用引入额外依赖。
"""
from __future__ import annotations

import json
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from app.config import TranslationConfig
from app.core.translator import (
    TranslationError,
    TranslationRequest,
    Translator,
    is_translatable,
    locate_untranslated,
    looks_like_verbatim_echo,
    register,
    should_check_echo,
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


# ------------------------------------------------------------------ 链路重试
#
# 中继链路上的 502/503/504 是**瞬时**故障：上游节点被踢下线、后端池在扩容重启、
# 网关健康检查把某台摘掉 —— 同一份请求过一两秒再发往往就成功了。它和
# 「模型翻不动」（见 ``_recover_echoes``）是两回事，但后果一样：一批字幕直接
# 卡死、整个任务中断。所以同样要自动重试。

#: 重发有意义的状态码。
#:
#: 4xx 里的 400/401/403/404/422 是**确定性**失败（密钥错、模型名错、参数非法），
#: 重发一百次也是同样的结果，只会白烧配额 —— 刻意不在这一列里。
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: 第一次失败后大约等这么久，之后逐次翻倍（实际会带抖动，见 ``_backoff``）。
_RETRY_BASE_DELAY = 0.6
#: 单次等待上限，别让一个坏掉的网关卡住整段字幕。
_RETRY_MAX_DELAY = 8.0
#: 服务端 ``Retry-After`` 的封顶值。
_RETRY_MAX_WAIT = 30.0

#: 等待由它执行 —— 抽成模块级变量，测试里替换掉就不必真的干等。
_SLEEP = time.sleep


def _backoff(attempt: int) -> float:
    """指数退避 + 抖动。

    抖动是给「服务端刚恢复、所有客户端同时重试」留的余量：固定等待会让
    它们齐步冲上去，把刚起来的节点再打挂一次。取 ``[base/2, base]`` 而不是
    ``[0, base]``，是为了保证至少等了一半，别把重试退化成紧凑轮询。
    """
    base = min(_RETRY_BASE_DELAY * (2 ** max(0, attempt)), _RETRY_MAX_DELAY)
    return base / 2 + random.uniform(0, base / 2)


def _is_cert_error(exc: BaseException) -> bool:
    """证书校验 / 主机名不匹配 —— 确定性失败，重发无用。"""
    candidates = (exc, getattr(exc, "reason", None))
    return any(isinstance(item, ssl.SSLCertVerificationError) for item in candidates)


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """服务端显式要求的等待秒数；只认纯数字形式的 ``Retry-After``。"""
    headers = getattr(exc, "headers", None)
    raw = headers.get("Retry-After") if headers else None
    if not raw:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        # HTTP-date 形式的 Retry-After 这里不解析：中继基本只用秒数，
        # 解析日期还得把系统时钟的偏差一起背进来。
        return None


def _retry_delay(exc: BaseException, attempt: int) -> float | None:
    """这次失败要不要重发、等多久；返回 None 表示「重发也没用」。

    注：``urllib.error.HTTPError`` 是 ``URLError`` 的子类，``URLError`` 又是
    ``OSError`` 的子类，所以判断顺序必须是「从具体到笼统」。
    """
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code not in RETRYABLE_STATUS:
            return None
        explicit = _retry_after(exc)
        # 服务端说了等多久就照办（封顶，免得被一个离谱的值卡死）。
        return min(explicit, _RETRY_MAX_WAIT) if explicit is not None else _backoff(attempt)
    if _is_cert_error(exc):
        return None
    if isinstance(exc, OSError):
        # 连接被重置、握手失败、DNS 抖动、读超时 —— 都可能下一秒就好了。
        return _backoff(attempt)
    return None


def _open_with_retry(
    req: urllib.request.Request,
    *,
    timeout: int,
    retries: int,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> str:
    """发一个已构造好的请求，对链路临时故障自动重发，返回解码后的响应体。

    重试用尽或遇到确定性错误时**原样抛出最后一次的异常**，由调用方按各自场景
    包装成 :class:`TranslationError` —— 「拉取模型列表失败」和「翻译请求失败」
    的措辞不一样，不该在这里写死。

    ``on_retry(第几次重试, 异常, 等待秒数)`` 每次重发前回调，供调用方记账。
    """
    retries = max(0, int(retries))
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(
                req, timeout=int(timeout), context=ssl.create_default_context()
            ) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except OSError as exc:
            delay = _retry_delay(exc, attempt)
            if delay is None or attempt >= retries:
                raise
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            _SLEEP(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def fetch_models(
    base_url: str, api_key: str, *, timeout: int = 30, retries: int = 2
) -> List[str]:
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
        raw = _open_with_retry(req, timeout=int(timeout), retries=retries)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise TranslationError(
            f"拉取模型列表失败: HTTP {exc.code} —— {explain_http_error(detail)}"
        ) from None
    except urllib.error.URLError as exc:
        raise TranslationError(f"连接 {url} 失败: {exc.reason}") from None
    except TimeoutError:
        raise TranslationError(f"拉取模型列表超时（{timeout}s）") from None
    except OSError as exc:
        raise TranslationError(f"连接 {url} 失败: {exc}") from None

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
    #:
    #: 取值依据（2026-09-26 判别实验）：`grok-chat-fast` 这种免费档是
    #: **请求级**随机失败，实测单次回抄率约 1/3，且与批量大小、提示词写法、
    #: 是否带 system 角色都无关（同一份请求体发 8 次，2 次退回原文）。
    #: 于是「连续 N+1 次都撞上」的概率是 (1/3)^(N+1)：
    #: N=2 时 3.7%，N=3 时 1.2% —— 加一次只多花约 2.6% 的请求，很划算。
    echo_retries = 3

    #: 整批重发仍救不回来时，是否把没翻的条目拎出来逐条重译。
    #: 单条请求更不容易被上游“整批偷懒”，实测能救回大部分残留。
    echo_downgrade = True

    #: 逐条重译时每个条目最多试几次。单条同样是约 1/3 的随机失败，
    #: 只试一次会留下「每批零星几条」的残留（一部 1000 条的字幕累计能到几十条），
    #: 试两次把单条残留率从 1/3 压到 1/9。只对已判定失败的条目发请求，代价极小。
    echo_item_retries = 2

    #: 链路临时故障（502/503/504、429 限流、连接抖动）的自动重发次数。
    #:
    #: 取值说明：与 ``echo_retries`` 不同，这个数**不是**从实测失败率推出来的 ——
    #: 502 无法按需复现（见 ``scripts/verify_http_retry.py --live`` 对真实中继的
    #: 采样），用的是常规默认。首次失败等 0.3–0.6s，之后逐次翻倍，3 次重试
    #: 总共最多多等约 3s：足够跨过「上游节点被摘掉、几秒后重新挂上」这类抖动，
    #: 又不会在网关真挂了的时候把整段任务拖住（那时早报错比干等有用）。
    http_retries = 3

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
        echo_downgrade: bool | None = None,
        echo_item_retries: int | None = None,
        http_retries: int | None = None,
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
        if echo_downgrade is not None:
            self.echo_downgrade = bool(echo_downgrade)
        if echo_item_retries is not None:
            self.echo_item_retries = max(0, int(echo_item_retries))
        if http_retries is not None:
            self.http_retries = max(0, int(http_retries))
        self._api_key = api_key  # 私有：不参与 repr，也不写进日志
        #: 因模型丢换行而被单独重译的条目数，便于观测提示词是否退化
        self.line_repair_count = 0
        #: 因整批原样回抄而重发的批次数
        self.echo_retry_count = 0
        #: 因残留未翻译条目而触发的逐条重译**条目数**（去重）
        self.echo_item_count = 0
        #: 逐条重译实际发出的**请求次数**（含对同一条目的重复尝试），用于算配额开销
        self.echo_item_attempts = 0
        #: 逐条重译救回来的条数
        self.echo_repaired_count = 0
        #: 用尽所有手段后仍未翻译的条数（> 0 说明这批译文不完整）
        self.untranslated_count = 0
        #: 因链路临时故障（502/503/504、限流、连接抖动）自动重发的**请求次数**
        self.http_retry_count = 0
        #: 重发原因分布，如 ``{"HTTP 502": 2}`` —— 界面据此说清"到底怎么了"
        self.http_retry_reasons: Dict[str, int] = {}
        #: 重发前累计等待的秒数，用于判断中继是不是在持续抖动
        self.http_retry_waited = 0.0

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
    def _note_http_retry(self, attempt: int, exc: BaseException, delay: float) -> None:
        """记一次链路重试：谁、为什么、等了多久。"""
        self.http_retry_count += 1
        self.http_retry_waited += delay
        if isinstance(exc, urllib.error.HTTPError):
            reason = f"HTTP {exc.code}"
        elif isinstance(exc, TimeoutError):
            reason = "响应超时"
        else:
            reason = "连接中断"
        self.http_retry_reasons[reason] = self.http_retry_reasons.get(reason, 0) + 1

    def _retry_hint(self, retryable: bool) -> str:
        """给错误信息补一句「已经重试过 N 次了」，免得用户以为是偶发失败。"""
        if not retryable or not self.http_retries:
            return ""
        return f"（已自动重试 {self.http_retries} 次）"

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST 并解析 JSON 响应；链路临时故障（502/503/504…）自动重发。

        重试与「回抄重试」是两件事：这里处理的是**请求根本没被正常处理**
        （网关 502、限流 429、连接被重置），重发同一份请求就是正确做法；
        回抄则是请求被处理了但没翻译，得换提示词（见 ``_recover_echoes``）。
        """
        url = f"{self.base_url}{path}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")
        req.add_header("User-Agent", "subtitle-translator/0.1")

        try:
            raw = _open_with_retry(
                req,
                timeout=self.timeout,
                retries=self.http_retries,
                on_retry=self._note_http_retry,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            hint = self._retry_hint(exc.code in RETRYABLE_STATUS)
            raise TranslationError(
                f"HTTP {exc.code} from {url} —— {explain_http_error(detail)}{hint}"
            ) from None
        except urllib.error.URLError as exc:
            raise TranslationError(
                f"连接 {url} 失败: {exc.reason}{self._retry_hint(not _is_cert_error(exc))}"
            ) from None
        except TimeoutError:
            raise TranslationError(
                f"请求 {url} 超时（{self.timeout}s）{self._retry_hint(True)}"
            ) from None
        except OSError as exc:
            raise TranslationError(f"连接 {url} 失败: {exc}{self._retry_hint(True)}") from None

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

        parsed = self._request_batch(texts, source_lang, target_lang)
        # 源语言写 auto 时按文本本身的书写族判断，别把最常见的用法漏在门外。
        if should_check_echo(source_lang, target_lang, samples=texts):
            parsed = self._recover_echoes(texts, parsed, source_lang, target_lang)

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

    def _recover_echoes(
        self,
        texts: Sequence[str],
        parsed: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """把「没翻译」压到最少，全都压不下去才报错。

        三级处置，代价由小到大：

        1. 整批被原样退回 —— 换成把要求说死的提示词整批重发（最多 ``echo_retries`` 次）；
        2. 重发后仍有零星条目一字未改 —— 只把这几个拎出来单独重译
           （每个条目最多 ``echo_item_retries`` 次），比其他条目贵不了多少；
        3. 单条重译仍纹丝不动 —— 计入 ``untranslated_count``；若整批可比条目**无一**
           翻出来，说明这条通道是真不干活，直接报错，绝不把没翻译的字幕当成品交出去。
        """
        attempts = 1  # 已经发过的那一次
        for _ in range(self.echo_retries):
            if not looks_like_verbatim_echo(texts, parsed, source_lang, target_lang):
                break
            self.echo_retry_count += 1
            attempts += 1
            parsed = self._request_batch(texts, source_lang, target_lang, strict=True)

        leftover = locate_untranslated(texts, parsed, target_lang)
        if leftover and self.echo_downgrade:
            self.echo_item_count += len(leftover)
            pending = list(leftover)
            for _ in range(self.echo_item_retries):
                if not pending:
                    break
                still_stuck: list[int] = []
                for index in pending:
                    self.echo_item_attempts += 1
                    retried = self._translate_one(
                        texts[index], source_lang, target_lang, strict=True
                    )
                    if retried.strip() != texts[index].strip():
                        parsed[index] = retried
                        self.echo_repaired_count += 1
                    else:
                        still_stuck.append(index)
                pending = still_stuck

        remaining = locate_untranslated(texts, parsed, target_lang)
        if remaining:
            self.untranslated_count += len(remaining)
            comparable = sum(1 for text in texts if is_translatable(text))
            if comparable and len(remaining) >= comparable:
                # 报**实际**发过几次，不是重试预算。
                # 单条批次走不到整批判定（min_items=3），这里只有 1 次 ——
                # 旧文案写死 echo_retries + 1，会谎报「连续 4 次」，误导定位。
                raise TranslationError(
                    f"请求 {attempts} 次，{source_lang} -> {target_lang} 的译文"
                    "始终与原文完全相同，未真正执行翻译。"
                    "这通常是上游把请求路由到了不支持跨语言翻译的通道，"
                    "稍后重试或换一个模型即可。"
                )
        return parsed

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
            parsed = [
                self._translate_one(t, source_lang, target_lang, strict=strict)
                for t in texts
            ]
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
            rebuilt[index] = self._decode(translation)
        return "\n".join(rebuilt)

    def _translate_one(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        *,
        strict: bool = False,
    ) -> str:
        """单条翻译。

        ``text`` 与返回值都保持「线上形式」（换行仍是 ⏎ 记号），
        由 ``translate_batch`` 统一做编解码 —— 这样批量路径与逐条路径
        产出的结果在同一个坐标系里，回抄判定才能直接比较。
        """
        content = self._chat(
            [
                {
                    "role": "system",
                    "content": self._system_prompt(target_lang, strict=strict),
                },
                {"role": "user", "content": json.dumps([text], ensure_ascii=False)},
            ],
            max_tokens=1024,
        )
        parsed = self._parse_array(content, 1)
        if parsed is not None:
            return parsed[0]
        # 单条也解析不出就退到纯文本：去掉可能的代码围栏与首尾引号。
        cleaned = _FENCE_RE.sub("", content.strip()).strip().strip('"').strip()
        if not cleaned or cleaned[0] in "[{":
            # 解析不出来的结构体绝不能当译文 —— 上层会把它认成"翻好了"，
            # 于是垃圾内容被静默塞进字幕。宁可原样交回，让上层判为"没翻"。
            return text
        return cleaned


def build_from_config(cfg: TranslationConfig) -> OpenAICompatTranslator:
    """注册表工厂：让 ``create_engine("openai", config=cfg)`` 可用。"""
    return OpenAICompatTranslator.from_config(cfg)


register(OpenAICompatTranslator)
