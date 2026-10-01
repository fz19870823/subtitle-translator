"""OpenAI 兼容 Chat Completions 翻译后端。

适用于任何暴露 ``POST {base_url}/chat/completions`` 的服务：
OpenAI、各类中转/聚合网关（含 grok2api 这类自建中继）、以及本地推理服务。

只依赖标准库 urllib，避免为一个 HTTP 调用引入额外依赖。
"""
from __future__ import annotations

import http.client
import json
import queue
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple

from app.config import TranslationConfig
from app.core.translator import (
    TranslationCancelled,
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


def _open_stream(
    req: urllib.request.Request,
    *,
    timeout: int,
    retries: int,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    stop: Callable[[], bool] | None = None,
):
    """发一个已构造好的请求，对链路临时故障自动重发，返回**尚未读完**的响应。

    重试用尽或遇到确定性错误时**原样抛出最后一次的异常**，由调用方按各自场景
    包装成 :class:`TranslationError` —— 「拉取模型列表失败」和「翻译请求失败」
    的措辞不一样，不该在这里写死。

    ``on_retry(第几次重试, 异常, 等待秒数)`` 每次重发前回调，供调用方记账。

    与 :func:`_open_with_retry` 的唯一区别是**不把响应读完** —— 流式读取要的
    就是这个句柄，好让调用方在分片之间响应取消。句柄由调用方负责关闭。

    ``stop`` 是「用户是不是已经点了取消」：重试前先问一句。一边声称「取消很快
    生效」一边自己把退避等满，说不过去。
    """
    retries = max(0, int(retries))
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(
                req, timeout=int(timeout), context=ssl.create_default_context()
            )
        except OSError as exc:
            delay = _retry_delay(exc, attempt)
            if delay is None or attempt >= retries:
                raise
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            if stop is not None and stop():
                raise TranslationCancelled("已取消（重试等待期间）") from None
            _SLEEP(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _open_with_retry(
    req: urllib.request.Request,
    *,
    timeout: int,
    retries: int,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> str:
    """同 :func:`_open_stream`，但把响应体读完并解码返回。"""
    with _open_stream(req, timeout=timeout, retries=retries, on_retry=on_retry) as resp:
        return resp.read().decode("utf-8", errors="replace")


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


# ------------------------------------------------------------------ 流式读取
#
# 为什么值得为它写一整套：一次请求可能携带 200 条字幕（上下文窗口上限），
# 模型吐完得十几秒甚至更久。以前取消只在**批与批之间**生效，用户点下取消之后
# 还得干等这一批跑完 —— 屏幕上什么都没发生，很像程序卡死。
#
# 打开 ``stream: true`` 之后，响应变成一个个 SSE 分片。我们在分片之间查一次取消：
# 命中就立刻关连接（服务端随之停止生成，省下的额度是实打实的），并把**已经完整
# 收到**的那几条交回去。于是「取消」的代价从「一批」变成一个分片。
#
# 但「分片之间」还不够快 —— 见 ``_StreamReader``：读取必须离开调用方线程。

#: 一行 SSE 最长允许多少字节。坏掉的中继可能只发不换行，兜住内存。
_MAX_SSE_LINE = 1 << 20

#: 调用方多久醒一次去「查取消 / 看有没有新数据」。它同时就是取消延迟的上限。
#:
#: 真实中继实测：模型可能先「想」 2.5–5.7 秒才吐第一个分片（占整段耗时的 45%）。
#: 取消检查若和读取挤在同一个线程里，这几秒内根本轮不到它 —— 实测点下取消要
#: 2.0s 才有反应（整段 12.5s）。
_STREAM_POLL = 0.2

#: 读取线程交给调用方的「流到此为止」信号（与真实数据行区分开）。
_STREAM_EOF = object()


def _quiet_close(closer: Callable[[], None]) -> None:
    """关连接；失败也不吭声 —— 收尾动作不该盖住真正的错。"""
    try:
        closer()
    except (OSError, ValueError):
        pass


def _close_response(resp) -> None:
    """关掉响应；没有 ``close`` 的对象（测试替身、极简包装）直接放过。"""
    closer = getattr(resp, "close", None)
    if callable(closer):
        _quiet_close(closer)


def _release_stream(reader: threading.Thread, resp) -> None:
    """收掉读取线程和连接，**绝不阻塞调用方**。

    ⚠️ **不能在这里直接 ``resp.close()``**：读线程阻塞在 ``readline`` 时握着
    ``BufferedReader`` 的内部锁，而 ``close()`` 要拿同一把锁 —— 于是「取消」会卡在
    关闭连接这一步，一直等到读线程读回数据为止。实测（``scripts/diag_cancel_latency.py``）：
    服务端只睡 2.0s，取消延迟却成了 1.68s，而且这期间取消回调**一次都没再被调用**
    （主线程根本没在跑）。
    把关闭丢给后台线程，调用方立刻返回；读线程随后读到数据或连接超时，锁自然放开。

    读线程已经结束时（正常读完、或它自己就撞了错）就没这层顾虑，直接关。
    """
    closer = getattr(resp, "close", None)
    if not callable(closer):
        return
    if reader.is_alive():
        threading.Thread(
            target=_quiet_close, args=(closer,), name="sse-close", daemon=True
        ).start()
    else:
        _quiet_close(closer)


class _StreamReader(threading.Thread):
    """把阻塞的 SSE 读取放在后台线程里，读到的整行投进队列。

    为什么不让调用方自己读：``resp.readline`` 在想读的那一秒里是**不可打断**的，
    而取消检查必须在那段时间里也能执行。读取挪进线程后，调用方只在队列上等
    ``_STREAM_POLL`` 秒，于是取消延迟与「模型想多久」彻底无关。

    ⚠️ **不要试图用「调短 socket 读超时」在主线程里轮询**（这个实现的第一版就是）。
    那条路在明文 HTTP 上成立（``scripts/probe_stream_read.py`` 的路线 C），但真实
    中继走 TLS：同一个中继、同一份请求，把读超时调成 0.2s 之后，超时 12 次就在
    t=2.55s 提前收到 EOF，整段内容一个字都收不到；不碰超时则 4.65s 正常返回三行译文。
    原因是 ``http.client.HTTPResponse`` 的读路径在 OSError（超时是它的子类）时会
    关掉连接，而 TLS 上的超时重试又和明文 socket 的语义不同。
    证据脚本：``scripts/probe_live_stream.py``（A/B 对照）、
    ``scripts/diag_https_stream.py``（失败那一刻的内部状态）。
    线程方案不碰 socket 的任何内部状态，HTTP / HTTPS 行为一致。
    """

    def __init__(self, resp, out: "queue.Queue", abort: threading.Event) -> None:
        super().__init__(daemon=True, name="sse-reader")
        self._resp = resp
        self._out = out
        self._abort = abort

    def run(self) -> None:
        try:
            while not self._abort.is_set():
                try:
                    line = self._resp.readline(_MAX_SSE_LINE)
                except (OSError, ValueError) as exc:
                    # 取消时调用方会 close 掉响应，这里多半收到 ValueError；
                    # 反正调用方已经不要了，丢进队列让它自然结束即可。
                    self._out.put(exc)
                    return
                if not line:
                    self._out.put(_STREAM_EOF)
                    return
                self._out.put(line)
        except BaseException as exc:  # noqa: BLE001 - 兜底：绝不让线程静默死掉
            self._out.put(exc)


class _Sentinel:
    """增量解析的两种「不再往下切」信号（用单例对象，避免和真实译文撞车）。"""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:  # pragma: no cover - 只为调试时好认
        return f"<{self._name}>"


#: 数据还不够，等下一个分片
_NEED_MORE = _Sentinel("need-more")
#: 这个形状不打算增量解析了（数组结束、或开头不是字符串数组）
_GIVE_UP = _Sentinel("give-up")


def _is_event_stream(resp: object) -> bool:
    """服务端是不是真的按事件流回答。

    ``stream: true`` 只是个请求，中继完全可以忽略它、直接把整个 JSON 甩回来。
    那时按 SSE 去解析会一个字都读不到 —— 必须靠 Content-Type 认出来并降级。
    """
    headers = getattr(resp, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        # 没有 headers 的对象一律按「没流式」处理：降级路径是「读全文再解析」，
        # 对任何响应都成立；反过来把普通响应当流式解析则可能一个字都拿不到。
        return False
    return "event-stream" in str(headers.get("Content-Type") or "").lower()


def iter_stream_lines(
    resp,
    *,
    stop: Callable[[], bool] | None = None,
    poll: float = _STREAM_POLL,
    stall_timeout: float = 120.0,
) -> Iterator[str]:
    """逐行读一个流式响应，产出**去掉首尾空白**的整行。

    只负责「把行拿进来、让取消查得到、卡死了要报错」，怎么解释这些行交给调用方。
    抽出来是因为流式协议不止一种：OpenAI 兼容层用 SSE（``data: {...}``），
    Ollama 用 NDJSON（一行一个 JSON 对象，没有前缀）。两者的**读取**部分一模一样，
    而这段恰恰是最难写对的一段（线程、取消时机、连接释放），没必要写两遍、
    更没必要让两份实现慢慢漂开。

    读取跑在后台线程（见 :class:`_StreamReader`），本函数只从队列里取行 ——
    所以 ``stop()`` 不但在数据行之间查得到，在「模型还在想」的那几秒里也查得到。
    取消延迟是 ``poll``，不是「等第一个字节」。

    ``stall_timeout`` 是「多久没有任何数据就算死了」：有它兜底，一个中途不再
    说话的流不会把读取线程永远挂住。

    ⚠️ 取消检查在 ``yield`` **之后**，也就是「调用方处理完这一行、回来要下一行」
    的时候。顺序不能反：读取线程可能提前把几行塞进了队列，若在交出当前行之前
    就抛取消，那些已经到手的行会被当成「还没收到」而丢掉 ——
    而用户点取消时最想留住的恰恰就是它们。
    """
    out: "queue.Queue" = queue.Queue()
    abort = threading.Event()
    reader = _StreamReader(resp, out, abort)
    reader.start()
    last_data = time.monotonic()
    try:
        while True:
            try:
                item = out.get(timeout=poll)
            except queue.Empty:
                # 等不到数据 —— 模型「还在想」的那几秒正是这里。取消检查必须
                # 轮到它，否则用户点下取消要干等到模型开口（实测 2.0s）。
                if stop is not None and stop():
                    raise TranslationCancelled("已取消（流式接收中）")
                if time.monotonic() - last_data > stall_timeout:
                    raise TimeoutError(
                        f"流式响应 {stall_timeout:.0f} 秒没有任何数据"
                    ) from None
                continue
            if item is _STREAM_EOF:
                return
            if isinstance(item, BaseException):
                raise item
            last_data = time.monotonic()
            line = item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item
            yield line.strip()
            if stop is not None and stop():
                raise TranslationCancelled("已取消（流式接收中）")
    finally:
        # 收尾：告诉读取线程别再读了，并释放连接 —— 注意 _release_stream 保证
        # 这一步不会把调用方拖住（见它的注释）。
        abort.set()
        _release_stream(reader, resp)


def iter_sse_deltas(
    resp,
    *,
    stop: Callable[[], bool] | None = None,
    poll: float = _STREAM_POLL,
    stall_timeout: float = 120.0,
) -> Iterator[str]:
    """逐行读 SSE，产出 ``delta.content`` 片段。

    只认 ``data:`` 行；``:`` 开头的心跳、空行、解析不出来的行一律跳过 ——
    一行坏数据不该毁掉整批字幕。

    读取部分在 :func:`iter_stream_lines` 里（后台线程 + 取消 + 卡死保护）。
    """
    lines = iter_stream_lines(
        resp, stop=stop, poll=poll, stall_timeout=stall_timeout
    )
    try:
        for line in lines:
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = None
            if payload is not None:
                yield from _deltas_from(payload)
    finally:
        # 显式收掉内层生成器：让它的 finally（叫停读取线程、释放连接）跑到，
        # 而不是等 GC 回收 —— 那样时机不可控。
        lines.close()


def _deltas_from(payload: Any) -> List[str]:
    """从一个 SSE 事件里取出内容片段。

    同时认 ``delta.content``（标准流式）与 ``message.content``（有些中继在流式
    模式下仍按完整消息回复）。推理模型的 ``reasoning_content`` 不要 ——
    那是思维链，不是译文。
    """
    if not isinstance(payload, dict):
        return []
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []
    pieces: List[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for key in ("delta", "message"):
            holder = choice.get(key)
            if not isinstance(holder, dict):
                continue
            content = holder.get("content")
            if isinstance(content, str) and content:
                pieces.append(content)
                break
    return pieces


class ArrayStreamParser:
    """边收边从 JSON 数组里切出**已经完整**的字符串元素。

    它买到的正是「不用等返回全部内容」：一次请求 200 条字幕，等整段收完再解析
    的话，用户中途取消就等于把这 200 条全丢了；这里每收完一个字符串就交出一条，
    取消时已经到手的那部分能直接落盘。

    刻意**不**追求完整的 JSON 语义：它只负责「切出候选片段」，最终结果仍由
    ``_parse_array`` 用标准 ``json.loads`` 定稿 —— 增量解析少切几条没关系，
    切出错的才是最糟的。
    """

    #: 起始 ``[`` 之前允许跳过的字符：空白、代码围栏
    _SKIP = " \t\r\n`"

    def __init__(self) -> None:
        self._buf = ""
        self._cursor = 0
        self._started = False
        self._stopped = False
        #: 已经完整切出来的元素（按出现顺序）
        self.items: List[str] = []

    def feed(self, chunk: str) -> List[str]:
        """吃进一段新内容，返回**本次**新切出来的元素。"""
        if self._stopped or not chunk:
            return []
        self._buf += chunk
        fresh: List[str] = []
        while not self._stopped:
            value = self._take_one()
            if value is _NEED_MORE:
                break
            if value is _GIVE_UP:
                self._stopped = True
                break
            self.items.append(value)
            fresh.append(value)
        return fresh

    def _take_one(self):
        """尝试从游标处取一个完整元素。"""
        buf = self._buf
        size = len(buf)
        index = self._cursor

        if not self._started:
            while index < size:
                char = buf[index]
                if char in self._SKIP:
                    index += 1
                    continue
                if buf.startswith("json", index):
                    index += 4  # ```json 这种围栏，连词一起跳掉
                    continue
                break
            if index >= size:
                self._cursor = index
                return _NEED_MORE
            if buf[index] != "[":
                # 形状不认识（对象、纯文本解释…）。交给整段解析那条路 ——
                # 它会把 ``[...]`` 片段找出来，这里别自作聪明。
                return _GIVE_UP
            self._started = True
            index += 1

        while index < size:
            char = buf[index]
            if char in " \t\r\n,":
                index += 1
                continue
            if char == "]":
                self._cursor = index
                return _GIVE_UP  # 数组结束，增量解析的活儿干完了
            if char != '"':
                # 非字符串元素（嵌套对象/数字）：不猜，交给整段解析。
                return _GIVE_UP
            end = _find_string_end(buf, index)
            if end is None:
                self._cursor = index
                return _NEED_MORE  # 这个字符串还没收尾，等下一个分片
            raw = buf[index: end + 1]
            self._cursor = end + 1
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return _GIVE_UP
            if not isinstance(value, str):
                return _GIVE_UP
            return value

        self._cursor = index
        return _NEED_MORE


def _find_string_end(text: str, start: int) -> int | None:
    """``start`` 指向起始引号，返回收尾引号的位置；还没收尾则返回 None。"""
    index = start + 1
    size = len(text)
    while index < size:
        char = text[index]
        if char == "\\":
            # 跳过整个转义序列。\" \\ \uXXXX 都不会在中间夹出真的引号。
            index += 2
            continue
        if char == '"':
            return index
        index += 1
    return None


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

    #: 逐条重译时每个条目最多**再**试几次（合计：1 次批量 + N 次逐条）。
    #:
    #: 「有问题的部分重试三次」就落在这里。单条同样是约 1/3 的随机失败，只试一次
    #: 会留下「每批零星几条」的残留（一部 1000 条的字幕累计能到几十条）：
    #: 两次 → (1/3)³ ≈ 3.7%，三次 → (1/3)⁴ ≈ 1.2%。只对**已经判定失败**的条目
    #: 发请求（一批里通常只有个位数条），代价极小。
    #:
    #: 三轮之后仍没救回来的，**不再当译文收下** —— 由调度层置空并打上「未翻译」
    #: 标记（见 ``translator._apply_results``），等整条队列跑完再让用户手动重试。
    echo_item_retries = 3

    #: 链路临时故障（502/503/504、429 限流、连接抖动）的自动重发次数。
    #:
    #: 取值说明：与 ``echo_retries`` 不同，这个数**不是**从实测失败率推出来的 ——
    #: 502 无法按需复现（见 ``scripts/verify_http_retry.py --live`` 对真实中继的
    #: 采样），用的是常规默认。首次失败等 0.3–0.6s，之后逐次翻倍，3 次重试
    #: 总共最多多等约 3s：足够跨过「上游节点被摘掉、几秒后重新挂上」这类抖动，
    #: 又不会在网关真挂了的时候把整段任务拖住（那时早报错比干等有用）。
    http_retries = 3

    #: 是否用 ``stream: true`` 接收响应。
    #:
    #: 收益全在**取消**上：一次请求可能带 200 条字幕、要十几秒才吐完，
    #: 流式让我们在分片之间就发现「用户点了取消」，立刻关连接（服务端也随之
    #: 停止生成），并把已经完整收到的几条留下来。非流式只能等整段返回，
    #: 取消的代价就是这一整批的时间。
    #:
    #: 默认开。中继不支持时会自动降级成一次性接收（见 ``_chat_stream``），
    #: 不需要人工干预；真遇到疑难中继也可以在 config.local.json 里设
    #: ``"stream": false`` 关掉。
    stream = True

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
        stream: bool | None = None,
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
        if stream is not None:
            self.stream = bool(stream)
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
        #: 中继没理会 stream / 流中途断掉，退回一次性接收的次数
        self.stream_fallback_count = 0

    # 防止密钥经由 repr/日志外泄
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"model={self.model!r}, api_key=<hidden>)"
        )

    # ------------------------------------------------------------------ 配置工厂
    @staticmethod
    def fetch_models(
        base_url: str = "", api_key: str = "", *, timeout: int = 30
    ) -> List[str]:
        """拉取模型 id 列表（``GET {base_url}/models``）。

        界面层按引擎取用这个方法，所以每个引擎都得提供一个同名同签名的静态方法 ——
        拉列表的地址**随引擎而异**（Ollama 走 ``/api/tags``），写死成某一个引擎的
        实现会让另一个引擎的「拉取模型」按钮打到不存在的路径上。

        注意这里是调用模块级同名函数，不是递归：类属性不参与函数体内的全局名查找。
        """
        return fetch_models(base_url, api_key, timeout=timeout)

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
            stream=cfg.stream,
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

    def _build_request(self, path: str, body: Dict[str, Any]) -> urllib.request.Request:
        """构造请求。``Accept`` 跟着 ``stream`` 走 —— 有些中继靠它才肯走流式。"""
        streaming = bool(body.get("stream"))
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream" if streaming else "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")
        req.add_header("User-Agent", "subtitle-translator/0.1")
        return req

    def _open(self, req: urllib.request.Request, url: str):
        """打开连接（可重试），把链路错误统一翻成 :class:`TranslationError`。

        :class:`TranslationCancelled` 不在此列 —— 它是用户的正常选择，
        不该被包装成「失败」。
        """
        try:
            return _open_stream(
                req,
                timeout=self.timeout,
                retries=self.http_retries,
                on_retry=self._note_http_retry,
                stop=self.should_stop,
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

    def _open_cancellable(self, req: urllib.request.Request, url: str):
        """发请求，但**在线程里** —— 于是「等响应头」那段也响应取消。

        为什么连打开连接都要挪走：真实中继实测 ``urlopen`` 要 **3.55s** 才返回，
        因为网关把响应头一直憋到第一个分片就绪才发。这几秒里同步的 ``urlopen``
        是阻塞的，取消检查根本没有机会跑 —— 线上表现就是「0.5s 点取消，
        3.5s 才有反应」，和「分片之间才查取消」一个样。
        挪进线程后，调用方每 ``_STREAM_POLL`` 秒醒一次问一句。

        取消之后这个响应就没人要了，由线程自己关掉（``abandoned``）。
        """
        out: "queue.Queue" = queue.Queue()
        abandoned = threading.Event()

        def worker() -> None:
            try:
                resp = self._open(req, url)
            except BaseException as exc:  # noqa: BLE001 - 原样交回主线程再抛
                out.put(exc)
                return
            if abandoned.is_set():
                _close_response(resp)  # 已经没人要了
                return
            out.put(resp)

        threading.Thread(target=worker, name="sse-open", daemon=True).start()
        try:
            while True:
                try:
                    item = out.get(timeout=_STREAM_POLL)
                except queue.Empty:
                    if self.should_stop():
                        raise TranslationCancelled("已取消（等待响应）") from None
                    continue
                if isinstance(item, BaseException):
                    raise item
                return item
        finally:
            abandoned.set()

    @staticmethod
    def _payload_from(raw: str, url: str) -> Dict[str, Any]:
        if not raw.strip():
            raise TranslationError(f"{url} 返回了空响应体")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise TranslationError(f"响应不是合法 JSON（前 300 字符）: {raw[:300]}") from None
        if not isinstance(payload, dict):
            raise TranslationError(f"响应顶层不是 JSON 对象: {raw[:300]}")
        return payload

    @staticmethod
    def _content(payload: Dict[str, Any]) -> str:
        """从**完整**响应体里取出助手回复。"""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TranslationError(
                f"响应缺少 choices: {json.dumps(payload, ensure_ascii=False)[:300]}"
            )
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise TranslationError("响应 message.content 不是字符串")
        return content

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST 并解析 JSON 响应；链路临时故障（502/503/504…）自动重发。

        重试与「回抄重试」是两件事：这里处理的是**请求根本没被正常处理**
        （网关 502、限流 429、连接被重置），重发同一份请求就是正确做法；
        回抄则是请求被处理了但没翻译，得换提示词（见 ``_recover_echoes``）。
        """
        url = f"{self.base_url}{path}"
        req = self._build_request(path, body)
        with self._open(req, url) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return self._payload_from(raw, url)

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
            "stream": bool(self.stream),
            # 注意：部分中继会忽略 max_tokens，这里只作为"建议上限"。
            "max_tokens": max(256, max_tokens),
        }
        if not self.stream:
            return self._content(self._post("/chat/completions", body))
        return self._chat_stream(body)

    def _chat_stream(self, body: Dict[str, Any]) -> str:
        """用 ``stream: true`` 发一次请求：边收边解析，取消时立刻断开。

        两条降级路径都通向「一次性接收」，区别只在触发条件：

        - 中继压根没理会 ``stream``（Content-Type 不是 event-stream）；
        - 流读到一半断了 —— 半截内容没法确认完整性，不如重发一次拿完整的
          （代价是多花一个请求，但那本来就要重发；不许比这更差）。

        流式只是让取消更快生效的优化，它坏掉不该让翻译整个不可用，
        所以两条路都自动退回，只记一笔 ``stream_fallback_count`` 供界面提示。
        """
        req = self._build_request("/chat/completions", body)
        url = f"{self.base_url}/chat/completions"
        parser = ArrayStreamParser()
        parts: List[str] = []
        # 刻意不用 ``with``：它的退出动作就是 ``close()``，而取消的那一刻读线程
        # 正握着 ``BufferedReader`` 的锁，close 会把调用方卡住（见 _release_stream）。
        # 连接的释放统一交给 iter_sse_deltas 的收尾逻辑。
        #
        # 也不直接 ``self._open``：中继可能把响应头憋到第一个分片就绪（实测 3.55s），
        # 同步等在那儿的话取消照样要干等（见 _open_cancellable）。
        resp = self._open_cancellable(req, url)
        if not _is_event_stream(resp):
            self.stream_fallback_count += 1
            try:
                raw = resp.read().decode("utf-8", errors="replace")
            finally:
                _close_response(resp)
            return self._content(self._payload_from(raw, url))
        stream = iter_sse_deltas(
            resp, stop=self.should_stop, stall_timeout=float(self.timeout)
        )
        try:
            for delta in stream:
                parts.append(delta)
                parser.feed(delta)
        except TranslationCancelled as exc:
            # 把已经完整收到的元素一起交出去。取消多半落在一批的中段，
            # 丢掉这一批等于让用户白等前面那十几秒。
            raise TranslationCancelled(str(exc), items=parser.items) from None
        except (OSError, http.client.HTTPException):
            # 中途断流（连接重置、读超时、IncompleteRead）。
            self.stream_fallback_count += 1
            return self._content(
                self._post("/chat/completions", {**body, "stream": False})
            )
        finally:
            # 显式收掉生成器：让它的 finally 跑到（释放连接、叫停读取线程），
            # 而不是等 GC 回收 —— 那样时机不可控。
            stream.close()
        content = "".join(parts)
        if not content.strip():
            # 声明了事件流却一个分片都没给。真实中继上这是**偶发**的 ——
            # 同一个中继、同一份请求，多数时候流得好好的，偶尔整条流空着回来。
            # 直接判死等于把中继的抖动转嫁给用户，所以退到整段接收再问一次
            # （与「中继根本没理会 stream」走同一条兜底路径）。
            # 整段也拿不到东西时，_payload_from 会报「返回了空响应体」。
            self.stream_fallback_count += 1
            return self._content(
                self._post("/chat/completions", {**body, "stream": False})
            )
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

        try:
            parsed = self._request_batch(texts, source_lang, target_lang)
            # 源语言写 auto 时按文本本身的书写族判断，别把最常见的用法漏在门外。
            if should_check_echo(source_lang, target_lang, samples=texts):
                parsed = self._recover_echoes(texts, parsed, source_lang, target_lang)
        except TranslationCancelled as exc:
            # 流式读取中途被取消：把已经完整拿到的条目换算成「本批请求内的下标」
            # 交回调度层。换算必须在这里做 —— 只有 translate_batch 知道 ``todo``
            # （空条目不送模型，元素下标 ≠ 请求下标）。"重试"路径也包在里面：
            # 取消落在整批重发或逐条重译上时，同样要能带走战果。
            raise TranslationCancelled(
                str(exc), partial=self._salvage(todo, exc.items, target_lang)
            ) from None

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

    def _salvage(
        self,
        todo: Sequence[Tuple[int, str]],
        items: Sequence[str],
        target_lang: str,
    ) -> List[Tuple[int, str]]:
        """取消时把已经完整收到的元素转成 ``(请求下标, 译文)``。

        两件必须做的事：

        1. **换算下标** —— ``items`` 的顺序对应送出去的数组，而数组里只有非空条目
           （``todo``）。直接拿元素下标当请求下标，会把译文写到别的条目上。
        2. **剔掉没翻译的** —— 取消发生在回抄检查之前，直接把回抄值当成果落盘，
           等于用「保住了几条」换来一份假译文。这里用和 ``locate_untranslated``
           同一套判据：只认书写族根本不对的，同形汉字词照旧放过。
        """
        if not items:
            return []
        sources = [text for _, text in todo]
        aligned = list(items[: len(sources)])
        bad = set(locate_untranslated(sources, aligned, target_lang))
        picked: List[Tuple[int, str]] = []
        for position, raw in enumerate(aligned):
            if position in bad:
                continue
            text = self._decode(raw)
            if text.strip():
                picked.append((todo[position][0], text))
        return picked

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
