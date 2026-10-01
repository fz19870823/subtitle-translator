"""Ollama 本地推理后端（走原生 ``POST /api/chat``）。

## 为什么不用 Ollama 自带的 OpenAI 兼容层

``/v1/chat/completions`` 看起来能用 —— 拉模型列表、非流式、SSE 流式都返回 200。
但它**静默丢弃** Ollama 的专有字段。2026-10-01 在 192.168.1.50:11434（Ollama 0.34.0）
上用 ``GET /api/ps`` 里模型**实际加载时**的 ``context_length`` 做的判据
（「返回 200」不能当证据，丢弃字段一样返回 200）：

| 送出的字段 | 期望 | ``/api/ps`` 实测 | 结论 |
| --- | --- | --- | --- |
| ``options.num_ctx=8192`` | 8192 | 40960（与不传时相同） | 丢弃 |
| 不传任何 options（对照） | 模型默认 | 40960 | —— |
| ``keep_alive="30m"`` | 30 分钟后卸载 | ``expires_at`` − now = 5 分 02 秒 | 丢弃 |
| ``think=false`` | 不产出思维链 | 照样产出 | 丢弃 |
| 原生 ``/api/chat`` + ``num_ctx=16384`` | 16384 | 16384 | **生效** |

而这四件事里至少三件是必须的：

- **``think``** —— qwen3 这类会「想」的模型默认会先写一大段思维链，两个后果都实测过：
  - **慢**：同一份三行字幕，think 开着 4.93s、关掉 0.58s（8.5 倍），译文质量看不出差别。
  - **可能一个字都拿不到**：把输出预算卡紧时思维链会把预算吃光，
    ``content`` 返回**空串**、``done_reason="length"``（探针实测 ``num_predict=256``
    时 ``content=""`` 而 ``thinking`` 有 439 字；同一个问题 7.20s / 256 token，
    关掉后 0.23s / 6 token）。空串到了解析层就是「整批没翻出来」→ 逐条重译 →
    每条还是空 → 最后报一句和原因毫无关系的「译文与原文完全相同」。
    预算紧恰恰发生在上下文被压短的时候，所以这条不是理论风险。

  非思考模型收到 ``think: false`` 不会报错（实测 200、正常出正文），所以无条件发。
- **``keep_alive``** —— 默认 5 分钟卸载模型。队列在两个文件之间停一会儿、用户接个
  电话，下一个请求就要重新加载：实测冷加载 11.2s、热 0.2s，差 50 倍。
- **``num_ctx``** —— 见下。

## 关于 num_ctx 的取舍

不传 ``num_ctx`` 时 Ollama 用的是**模型自身的上限**（实测 qwen2.5-14b→32768、
qwen3-8b→40960），这比任何手填的值都合理，所以默认就是**不传**。
只有用户在配置里显式写了 ``num_ctx`` 时我们才带上它（通常是为了压住显存）。

但不管传不传，都要防一件事：**Ollama 在提示词超限时会静默截断**。
实测往一个 ``num_ctx=4096`` 的模型里塞约 11500 个汉字，HTTP 200、没有报错、
``done_reason="stop"``，而 ``prompt_eval_count`` 只有 2050 —— 提示词被砍掉了八成，
砍掉的还是**开头**，也就是 system prompt 里那段「只输出 JSON 数组」的要求。
后果不是报错，而是一份看起来翻了、协议其实已经崩掉的译文。

所以每批发出去之前先做两件事（见 :meth:`OllamaTranslator._build_chat_body`
与 :meth:`OllamaTranslator._check_truncation`）：
发前按保守估计拦住放不下的批次；收回来后拿 ``prompt_eval_count`` 与
**token 数的下界**比对 —— 真实 token 数不可能低于那个下界，低太多就说明被砍了。
"""
from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterator, List, Sequence

from app.config import ConfigError, TranslationConfig

# 下划线开头的几个是 openai_compat 的内部件，这里刻意跨模块复用：读取线程、
# 连接释放、取消时机这三件事写两遍一定会慢慢漂开，而它们恰恰是最难写对的部分。
from app.core.engines.openai_compat import (
    _STREAM_POLL,
    _close_response,
    _open_with_retry,
    ArrayStreamParser,
    OpenAICompatTranslator,
    explain_http_error,
    iter_stream_lines,
)
from app.core.translator import TranslationCancelled, TranslationError, register

#: 本地没有密钥这回事。但 OpenAI 兼容层的构造器要求 api_key 非空，给它一个
#: 无害的占位串 —— Ollama 不校验 Authorization 头，带上也不会出错。
PLACEHOLDER_KEY = "ollama"

DEFAULT_BASE_URL = "http://127.0.0.1:11434"

#: 模型列表接口的请求超时。列表不该让人干等。
_TAGS_TIMEOUT = 15

#: 用户可能把 ``.../v1``、``.../api`` 甚至整个 ``.../v1/chat/completions``
#: 粘进「API 地址」。这些后缀都剥掉，只留服务根，否则拼出来的路径是双份的。
_KNOWN_SUFFIXES = (
    "/v1/chat/completions",
    "/api/chat",
    "/v1/models",
    "/api/tags",
    "/v1",
    "/api",
)

#: 留给生成的最低预算。低于这个数，返回的一定是半截 JSON。
_MIN_OUTPUT_TOKENS = 256

#: 估算与实际之间留的余量（BOS、role 包装等零碎开销）。
_CONTEXT_MARGIN = 64


def normalise_base_url(raw: str) -> str:
    """把用户填的地址规整成 Ollama 服务根。

    空值回落到本机默认（``http://127.0.0.1:11434``），这样「装了 Ollama 就跑」
    的人不用填任何东西。
    """
    url = (raw or "").strip().rstrip("/")
    if not url:
        return DEFAULT_BASE_URL
    if "://" not in url:
        # 大多数人填的是「192.168.1.50:11434」这种形式。不补 scheme 的话 urllib
        # 会把主机名当成协议名（"unknown url type"），错误信息还完全指不到点上。
        url = f"http://{url}"
    for suffix in _KNOWN_SUFFIXES:
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            break
    return url


def _is_cjk(char: str) -> bool:
    """这个字符算不算「一个字符约等于一个 token」的书写系统。"""
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF      # 日文假名
        or 0x3400 <= code <= 0x9FFF   # 中日韩汉字（含扩展 A）
        or 0xAC00 <= code <= 0xD7AF   # 韩文音节
        or 0xF900 <= code <= 0xFAFF   # 兼容汉字
        or 0x3000 <= code <= 0x303F   # 中日韩标点
        or 0xFF00 <= code <= 0xFFEF   # 全角形式
    )


def count_cjk(text: str) -> int:
    return sum(1 for char in text if _is_cjk(char))


def estimate_tokens(text: str) -> int:
    """token 数的**保守上界**：宁可多估，也不要放行一个注定被截断的请求。

    汉字/假名/韩文按 1 字 1 token（真实约 0.6–0.7），其余按 2 字符 1 token
    （英文真实约 4 字符 1 token）。两个方向都往多了估。
    """
    cjk = count_cjk(text)
    other = len(text) - cjk
    return cjk + other // 2 + 8


def lower_bound_tokens(text: str) -> int:
    """token 数的**下界**：真实值不该低于它。

    用来判「提示词是不是被服务端砍了」（见 ``_check_truncation``）。
    汉字按 3 字 1 token、其余按 6 字符 1 token —— 都比真实密度松，所以正常
    情况下真实值稳稳高于它，只有被砍过才会掉下来。
    """
    cjk = count_cjk(text)
    other = len(text) - cjk
    return cjk // 3 + other // 6 + 2


def extract_tag_names(payload: Any) -> List[str]:
    """从 ``/api/tags`` 的响应里取模型名。

    形状是 ``{"models": [{"name": "qwen2.5:14b", ...}]}``；也容忍直接给字符串数组，
    免得服务端换个包装就整个不可用。
    """
    if isinstance(payload, dict):
        entries = payload.get("models") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []

    names: List[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            value = entry.get("name") or entry.get("model")
        else:
            value = entry
        if isinstance(value, str) and value and value not in names:
            names.append(value)
    return names


def fetch_models(
    base_url: str,
    api_key: str = "",
    *,
    timeout: int = _TAGS_TIMEOUT,
    retries: int = 2,
) -> List[str]:
    """``GET {base_url}/api/tags``，返回本地已有的模型名。

    签名里保留 ``api_key`` 只是为了和 OpenAI 兼容那边**同一个形状** ——
    界面层是按引擎注入取数函数的，两边参数不一致会逼出一堆分支。
    这里不用它。
    """
    root = normalise_base_url(base_url)
    url = f"{root}/api/tags"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "subtitle-translator/0.1")

    try:
        raw = _open_with_retry(req, timeout=int(timeout), retries=int(retries))
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
    return extract_tag_names(payload)


def _is_ndjson(resp: object) -> bool:
    """服务端是不是按 NDJSON 回答。

    Ollama 流式返回 ``Content-Type: application/x-ndjson``。照着 SSE 去解析
    会一条都读不出来（那边有 ``data:`` 前缀，这边没有），所以必须先认出来。
    没有 headers 的对象一律按「不是 NDJSON」处理：退回整段接收对任何响应都成立，
    反过来则可能一个字都拿不到。
    """
    headers = getattr(resp, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return False
    return "ndjson" in str(headers.get("Content-Type") or "").lower()


def iter_ollama_stream(
    resp,
    *,
    stop: Callable[[], bool] | None = None,
    poll: float = _STREAM_POLL,
    stall_timeout: float = 120.0,
    stats: Dict[str, Any] | None = None,
) -> Iterator[str]:
    """逐行读 NDJSON，产出 ``message.content`` 片段。

    每行是一个完整的 JSON 对象（``{"message":{"content":"甲"},"done":false}``），
    最后一行 ``done: true`` 里带着 ``prompt_eval_count`` 这类统计 —— 传了
    ``stats`` 就顺手抄进去，供调用方判断提示词有没有被截断。

    坏行直接跳过：一行读坏了不该毁掉整批字幕。``{"error": ...}`` 则是服务端
    明确报错，原样抛出去。
    """
    for line in iter_stream_lines(
        resp, stop=stop, poll=poll, stall_timeout=stall_timeout
    ):
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        error = payload.get("error")
        if error:
            raise TranslationError(f"Ollama: {error}")
        message = payload.get("message")
        if isinstance(message, dict):
            piece = message.get("content")
            if isinstance(piece, str) and piece:
                yield piece
        if payload.get("done"):
            if stats is not None:
                for key in ("prompt_eval_count", "eval_count", "done_reason"):
                    if key in payload:
                        stats[key] = payload[key]
            return


@register
class OllamaTranslator(OpenAICompatTranslator):
    """通过 Ollama 原生 ``/api/chat`` 做字幕翻译。"""

    name = "ollama"

    #: 需要地址和模型（但不要密钥）。
    requires_api = True

    #: 本地服务没有密钥这回事。界面据此不再用「请先填写密钥」把人拦在门外。
    api_key_required = False

    #: 模型加载后保持多久。默认 5 分钟一卸载，队列里换个文件都要重新加载：
    #: 实测冷 11.2s / 热 0.2s。
    keep_alive = "30m"

    #: 关掉思维链。qwen3 这类模型开着的话，思维链会把输出预算吃光、
    #: ``content`` 返回空串（实测）。非思考模型收到它不会报错，所以无条件发。
    think = False

    #: 显式的上下文长度。``None`` 表示「交给 Ollama」—— 不传这个参数时它用的是
    #: 模型自身的上限（实测 32768 / 40960），比手填的值都合理，也谈不上截断。
    #: 只有用户想压住显存时才需要填。
    num_ctx: int | None = None

    #: 是否在收完回复后核对 ``prompt_eval_count``，判断提示词有没有被截断。
    #: 关掉它需要用户在配置里显式写 ``check_truncation: false``。
    check_truncation = True

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
        keep_alive: str | None = None,
        think: bool | None = None,
        num_ctx: int | None = None,
        check_truncation: bool | None = None,
        echo_retries: int | None = None,
        echo_downgrade: bool | None = None,
        echo_item_retries: int | None = None,
        http_retries: int | None = None,
        stream: bool | None = None,
    ) -> None:
        super().__init__(
            base_url=normalise_base_url(base_url),
            model=model,
            # 本地服务没有密钥；占位串只为满足父类「必须有密钥」的构造约定。
            api_key=api_key or PLACEHOLDER_KEY,
            timeout=timeout,
            temperature=temperature,
            style_hint=style_hint,
            batch_size=batch_size,
            preserve_line_breaks=preserve_line_breaks,
            echo_retries=echo_retries,
            echo_downgrade=echo_downgrade,
            echo_item_retries=echo_item_retries,
            http_retries=http_retries,
            stream=stream,
        )
        if keep_alive is not None:
            self.keep_alive = str(keep_alive)
        if think is not None:
            self.think = bool(think)
        if num_ctx is not None:
            self.num_ctx = int(num_ctx) if int(num_ctx) > 0 else None
        if check_truncation is not None:
            self.check_truncation = bool(check_truncation)

        #: 因为放不下而被压过生成长度的批次数（不是错误，但要说一声）
        self.context_clamped_count = 0
        #: 服务端实际只吃了一部分提示词的次数，> 0 说明有批次被截断过
        self.truncated_context_count = 0

    # ------------------------------------------------------------------ 配置工厂
    @classmethod
    def from_config(
        cls,
        cfg: TranslationConfig,
        *,
        engine_name: str | None = None,
    ) -> "OllamaTranslator":
        """从 :class:`~app.config.TranslationConfig` 构造实例。

        引擎专属的旋钮放在 ``translation.ollama`` 这个子对象里，例如::

            {"translation": {"engine": "ollama",
                             "base_url": "http://192.168.1.50:11434",
                             "model": "qwen2.5:14b",
                             "ollama": {"num_ctx": 16384,
                                        "keep_alive": "1h",
                                        "think": false}}}

        为什么不给它们开一等字段：这几个键只有本地引擎认，摆进共用配置里
        会让「换个引擎」变成一堆看不懂的残留。``extra`` 本来就是为此留的
        （见 :meth:`TranslationConfig.to_dict`，未知键原样保留、不会被抹掉）。
        """
        try:
            key = cfg.resolve_api_key()
        except ConfigError:
            # 本地服务没有密钥。这不是配置错误，别在这里拦人。
            key = ""

        section: Dict[str, Any] = {}
        extra = cfg.extra if isinstance(cfg.extra, dict) else {}
        nested = extra.get("ollama")
        if isinstance(nested, dict):
            section.update(nested)
        # 也允许把旋钮直接平铺在 translation 下（extra 里那些）
        for key_name in ("num_ctx", "keep_alive", "think", "check_truncation"):
            if key_name in extra and key_name not in section:
                section[key_name] = extra[key_name]

        return cls(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=key,
            timeout=cfg.timeout,
            temperature=cfg.temperature,
            style_hint=cfg.style_hint,
            batch_size=cfg.batch_size,
            preserve_line_breaks=cfg.preserve_line_breaks,
            stream=cfg.stream,
            keep_alive=section.get("keep_alive"),
            think=section.get("think"),
            num_ctx=section.get("num_ctx"),
            check_truncation=section.get("check_truncation"),
        )

    @staticmethod
    def fetch_models(
        base_url: str = "", api_key: str = "", *, timeout: int = _TAGS_TIMEOUT
    ) -> List[str]:
        """拉取本机已有模型（``GET {base_url}/api/tags``）。

        界面层按引擎取用这个方法 —— Ollama 的模型列表在 ``/api/tags``，
        照着 OpenAI 那套打到 ``/models`` 上会 404。
        同时覆盖父类的同名方法，否则它会拿着 Ollama 的地址去请求 ``/models``。

        注意这里是调用模块级同名函数，不是递归。
        """
        return fetch_models(base_url, api_key, timeout=timeout)

    def list_models(self) -> List[str]:
        """拉取本机已有的模型名（``/api/tags``）。

        不用 ``/v1/models``：同一个服务上两者返回的 id 一样，但原生接口是
        权威的，也少一层「兼容层哪天改了」的风险。
        """
        return fetch_models(self.base_url, timeout=min(self.timeout, _TAGS_TIMEOUT))

    def quality_notes(self, *, include_untranslated: bool = True) -> List[str]:
        notes = super().quality_notes(include_untranslated=include_untranslated)
        if self.context_clamped_count:
            # 压过生成长度就可能被截在半句上，必须让用户知道 —— 这正是
            # 「上下文窗口调太大了」的第一个信号。
            notes.append(
                f"{self.context_clamped_count} 批因上下文放不下被压短了生成长度"
                "（把「上下文窗口」调小，或调大 ollama.num_ctx）"
            )
        return notes

    # ------------------------------------------------------------------ 请求构造
    @staticmethod
    def _prompt_tokens(messages: Sequence[Dict[str, Any]]) -> int:
        return sum(
            estimate_tokens(str(message.get("content") or "")) + 8
            for message in messages
        )

    def _build_chat_body(
        self, messages: Sequence[Dict[str, Any]], max_tokens: int
    ) -> Dict[str, Any]:
        """组装原生请求体，并把「放不下」这件事在发出去之前就处理掉。

        Ollama 超限时不报错、直接砍掉提示词的开头（实测），而被砍掉的正好是
        system prompt 里那段输出格式要求。所以宁可在本地先拦。

        上下文够大时不碰 ``num_predict``；不够时把它压到放得下为止（并记一笔，
        让用户知道自这次生成长度被压短了）；连最低预算都放不下才报错。
        """
        requested = max(_MIN_OUTPUT_TOKENS, int(max_tokens))
        num_predict = requested

        window = self.num_ctx
        if window is not None:
            prompt = self._prompt_tokens(messages)
            budget = window - prompt - _CONTEXT_MARGIN
            if budget < _MIN_OUTPUT_TOKENS:
                raise TranslationError(
                    f"这批提示词估计要 {prompt} tokens，加上最低 {_MIN_OUTPUT_TOKENS} "
                    f"tokens 的生成长度已经超过 ollama.num_ctx={window}。"
                    "Ollama 在超限时会**静默截断**提示词（连「只输出 JSON 数组」的"
                    "格式要求一起砍掉），所以这里直接拦下。"
                    "请把「上下文窗口」调小，或把配置里的 ollama.num_ctx 调大。"
                )
            num_predict = min(requested, budget)
            if num_predict < requested:
                self.context_clamped_count += 1

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": bool(self.stream),
            "think": bool(self.think),
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict,
            },
        }
        if window is not None:
            body["options"]["num_ctx"] = window
        return body

    def _check_truncation(
        self, payload: Dict[str, Any], messages: Sequence[Dict[str, Any]]
    ) -> None:
        """核对服务端到底吃了多少提示词 —— 判断有没有被静默截断。

        判据不是「和估计值比」：我们的估计本来就是保守上界，比实际高是正常的。
        这里用**下界**：真实 token 数不可能低于它。低太多只有一个解释 ——
        服务端为了塞进上下文把提示词砍了。

        这个检查在默认配置下几乎不会触发（不传 ``num_ctx`` 时用的是模型的
        自身上限），它保的是「显式把 num_ctx 调小」和「模型本身上下文很小」
        这两种情况：那时宁可停下来报清楚，也不要交付一份协议已经崩掉的译文。
        """
        if not self.check_truncation:
            return
        used = payload.get("prompt_eval_count")
        if not isinstance(used, int) or used <= 0:
            return
        floor = sum(
            lower_bound_tokens(str(message.get("content") or ""))
            for message in messages
        )
        if used >= floor * 0.9:
            return
        self.truncated_context_count += 1
        window = self.num_ctx
        hint = (
            f"当前 ollama.num_ctx={window}，"
            if window is not None
            else "当前没设 num_ctx（用的是模型自身上限），"
        )
        raise TranslationError(
            f"服务端只处理了 {used} tokens 的提示词，而这段提示词至少有 {floor} "
            "tokens —— 内容被截断了，而 Ollama 不会为此报错。"
            f"{hint}请把「上下文窗口」调小。截断会砍掉开头的 system prompt"
            "（正是「只输出 JSON 数组」那段要求），继续跑只会得到协议崩掉的译文。"
        )

    @staticmethod
    def _reply_content(payload: Dict[str, Any]) -> str:
        """从原生响应里取出助手回复。"""
        message = payload.get("message")
        if not isinstance(message, dict):
            raise TranslationError(
                "响应里没有 message 对象: "
                f"{json.dumps(payload, ensure_ascii=False)[:300]}"
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise TranslationError("响应 message.content 不是字符串")
        if not content.strip():
            thinking = message.get("thinking") or ""
            if thinking:
                # 思维链把预算吃光了。这种情况解析层会当成「空回复」→ 逐条重译
                # → 每条还是空，最后报一句莫名其妙的「译文与原文完全相同」。
                # 不如在这里说清原因。
                raise TranslationError(
                    "模型把整个输出预算都花在思维链上了，正文是空的（本次生成了 "
                    f"{len(thinking)} 字思维链，done_reason="
                    f"{payload.get('done_reason')}）。把 ollama.think 设为 false，"
                    "或调大「上下文窗口」对应的输出预算。"
                )
            raise TranslationError(
                "Ollama 返回了空的 message.content"
                f"（done_reason={payload.get('done_reason')}）"
            )
        return content

    # ------------------------------------------------------------------ 传输
    def _chat(self, messages: Sequence[Dict[str, str]], *, max_tokens: int = 4096) -> str:
        body = self._build_chat_body(messages, max_tokens)
        if not self.stream:
            payload = self._post("/api/chat", body)
            content = self._reply_content(payload)
            self._check_truncation(payload, messages)
            return content
        return self._chat_stream(body, messages)

    def _chat_stream(
        self, body: Dict[str, Any], messages: Sequence[Dict[str, Any]]
    ) -> str:
        """用 ``stream: true`` 发一次请求，边收边解析。

        三条降级路径与 OpenAI 兼容层一致：中继没理会 ``stream``、流中途断掉、
        声明了流却一个分片都没给 —— 都退回整段接收。用户手动「导出…」、
        断点续传这些功能不该因为流式坏掉就不可用（只记一笔供界面提示）。

        区别只在**怎么解释这些行**：Ollama 是 NDJSON，没有 ``data:`` 前缀。
        """
        url = f"{self.base_url}/api/chat"
        req = self._build_request("/api/chat", body)
        parser = ArrayStreamParser()
        parts: List[str] = []
        stats: Dict[str, Any] = {}

        # 不直接 self._open：Ollama 也可能把响应头憋到第一个 token 就绪，
        # 同步等在那儿的话取消要干等（见 openai_compat._open_cancellable）。
        # 也不用 with：它的退出动作是 close()，而读线程那一刻握着 BufferedReader
        # 的锁，close 会把调用方卡住 —— 连接的释放统一交给 iter_stream_lines 的收尾。
        resp = self._open_cancellable(req, url)
        if not _is_ndjson(resp):
            self.stream_fallback_count += 1
            try:
                raw = resp.read().decode("utf-8", errors="replace")
            finally:
                _close_response(resp)
            payload = self._payload_from(raw, url)
            content = self._reply_content(payload)
            self._check_truncation(payload, messages)
            return content

        stream = iter_ollama_stream(
            resp,
            stop=self.should_stop,
            stall_timeout=float(self.timeout),
            stats=stats,
        )
        try:
            for delta in stream:
                parts.append(delta)
                parser.feed(delta)
        except TranslationCancelled as exc:
            # 把**已经完整收到**的元素一起交出去。取消多半落在一批的中段，
            # 丢掉这一批等于让用户白等前面那十几秒 —— 本地模型尤其等不起。
            # （``items`` 最终由 translate_batch 换算成请求下标并落盘。）
            raise TranslationCancelled(str(exc), items=parser.items) from None
        except (OSError, http.client.HTTPException):
            # 中途断流（连接重置、读超时、IncompleteRead）。
            # 半截内容没法确认完整性，重发一次拿完整的。
            self.stream_fallback_count += 1
            payload = self._post("/api/chat", {**body, "stream": False})
            content = self._reply_content(payload)
            self._check_truncation(payload, messages)
            return content
        finally:
            stream.close()

        content = "".join(parts)
        if not content.strip():
            # 声明了流却一个分片都没给 —— 退回整段接收再问一次。
            self.stream_fallback_count += 1
            payload = self._post("/api/chat", {**body, "stream": False})
            content = self._reply_content(payload)
            self._check_truncation(payload, messages)
            return content
        self._check_truncation(stats, messages)
        return content


def build_from_config(cfg: TranslationConfig) -> OllamaTranslator:
    """注册表工厂：让 ``create_engine("ollama", config=cfg)`` 可用。"""
    return OllamaTranslator.from_config(cfg)
