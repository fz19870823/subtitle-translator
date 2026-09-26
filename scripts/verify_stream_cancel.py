"""验证「流式接收让取消更快生效」：本地起一个真的会分片吐字的 HTTP 服务端。

为什么不用桩：桩只能证明「我写的分支被走到了」，证明不了取消到底要等多久、
连接有没有真的提前关掉、已经收到的条目有没有留下来。这里起一个真实的
``http.server``，用真实的 ``urllib`` 走完整的 ``translate_cues``，
只在「服务端怎么回答」这一点上做手脚 —— 零配额、可重复。

三个场景（同一份字幕、同一个服务端，只改「怎么收」和「怎么切批」）：

===================  ========  =======  ==========================================
场景                   收法       批大小    说明
===================  ========  =======  ==========================================
``stream``            流式        1 批     取消只需等到下一个轮询，已收到的条目留下
``stream-slow-start`` 流式        1 批     模型先思考 2s 才开口 —— 取消也不用等它
``whole-batched``     整段接收    多批     取消要等到**这一批**跑完（批与批之间才查）
``whole-onebatch``    整段接收    1 批     取消完全无效 —— 跑完才发现，为时已晚
===================  ========  =======  ==========================================

服务端会记录：每一条是什么时候发出去的。于是「取消那一刻服务端已经发到第几条」
就成了一项硬证据 —— 客户端说自己提前停手，服务端的时间戳能对上。

``--live N`` 另跑一件事：对配置里的真实中继发一次流式请求，量「第一个分片什么时候
到」与「整段什么时候收完」。两者差得明显说明中继**真的**在流式；几乎同时则说明它
把整个回复憋到最后才发（那时流式的收益只剩「关连接」）。

用法::

    python scripts/verify_stream_cancel.py
    python scripts/verify_stream_cancel.py --items 20 --delay 0.3 --cancel-after 0.8
    python scripts/verify_stream_cancel.py --live 6
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config  # noqa: E402
from app.core import subtitle_io  # noqa: E402
from app.core.engines import openai_compat  # noqa: E402
from app.core.engines.openai_compat import OpenAICompatTranslator  # noqa: E402
from app.core.translator import TranslationCancelled, TranslationError  # noqa: E402

#: 假译文必须落在**目标语言的书写族**里，否则会被回抄防护当成没翻，
#: 污染传输层的结论（源语言是日文假名，译文是中文汉字，正是正常的跨语言结果）。
FAKE_TRANSLATION = "这是第 {} 条译文。"

#: 每条的填充字符数。让响应体足够大，好让「客户端断开后服务端立刻写不进去」
#: 这件事真的发生 —— 小响应会被 socket 发送缓冲区整个吞下去，断开检测不出来。
PAD_CHARS = 400

JA_TEXTS = [
    "何度も言ったはずだ。",
    "そんなの無理だよ。",
    "お前、何を考えてるんだ？",
    "分かった。任せるよ。",
    "ここで待っていてくれ。",
    "明日また来る。",
]


def build_srt(count: int) -> str:
    blocks = []
    for index in range(count):
        start = index * 2
        blocks.append(
            f"{index + 1}\n"
            f"{subtitle_io.format_timestamp(start)} --> "
            f"{subtitle_io.format_timestamp(start + 1.5)}\n"
            f"{JA_TEXTS[index % len(JA_TEXTS)]}"
        )
    return "\n\n".join(blocks) + "\n"


def _translation(index: int) -> str:
    return FAKE_TRANSLATION.format(index + 1) + "。" * PAD_CHARS


def _extract_texts(raw: bytes) -> list[str]:
    """从请求体里取出待翻译的数组（user 消息的内容）。"""
    try:
        payload = json.loads(raw.decode("utf-8"))
        content = payload["messages"][-1]["content"]
        if content.startswith("源语言: "):
            content = content.split("\n", 1)[1]
        value = json.loads(content)
        return value if isinstance(value, list) else []
    except Exception:  # noqa: BLE001 - 假服务端，取不出来就按 0 条算
        return []


class _Recorder:
    """把服务端的回答过程记下来，供事后对照。"""

    def __init__(self, items: int, delay: float) -> None:
        self.items = items
        self.delay = delay
        #: 「模型先思考」多久才吐第一个分片（真实中继实测 2.5–5.7s）
        self.first_delay = 0.0
        self.records: list[dict[str, Any]] = []

    def begin(self, *, stream: bool, count: int) -> dict[str, Any]:
        record = {
            "stream": stream,
            "count": count,
            "started": time.monotonic(),
            #: 每发出一条的时刻
            "sent_at": [],
            "broke_at": None,
            "finished": False,
        }
        self.records.append(record)
        return record


def _make_handler(recorder: _Recorder):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # noqa: D102 - 静音默认的 stderr 日志
            pass

        def _squeeze_socket(self) -> None:
            """把发送缓冲区调小：这样客户端一走，下一次写就会立刻报错。

            默认 64KB 的缓冲区会把整个响应吞下去，断开根本体现不出来 ——
            我们就会把「服务端还在傻发」误读成「它还在正常工作」。
            """
            try:
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
            except OSError:
                pass

        def _write(self, record: dict[str, Any], blob: bytes) -> bool:
            """写一段并冲一下；客户端已经走了就记一笔并返回 False。"""
            try:
                self.wfile.write(blob)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                record["broke_at"] = len(record["sent_at"])
                return False
            return True

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的约定
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)  # 必须读完，否则下一个请求会读到脏数据
            body = json.loads(raw.decode("utf-8"))
            texts = _extract_texts(raw)
            count = len(texts) or 1
            streaming = bool(body.get("stream"))
            record = recorder.begin(stream=streaming, count=count)
            self._squeeze_socket()
            if streaming:
                self._serve_stream(record, count)
            else:
                self._serve_whole(record, count)

        def _serve_stream(self, record: dict[str, Any], count: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            if recorder.first_delay:
                # 模拟「模型先思考」：这段时间客户端一个字节都收不到。
                # 取消要等多久，全看能不能在阻塞的读期间发现它 ——
                # 这正是「只把检查写在分片之间」会露马脚的地方。
                time.sleep(recorder.first_delay)
            for index in range(count):
                time.sleep(recorder.delay)  # 模拟模型逐条吐字
                prefix = "[" if index == 0 else ""
                suffix = "]" if index == count - 1 else ", "
                piece = json.dumps(_translation(index), ensure_ascii=False)
                payload = {"choices": [{"delta": {"content": f"{prefix}{piece}{suffix}"}}]}
                event = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                if not self._write(record, event.encode("utf-8")):
                    return
                record["sent_at"].append(time.monotonic())
            if self._write(record, b"data: [DONE]\n\n"):
                record["finished"] = True
            self.close_connection = True

        def _serve_whole(self, record: dict[str, Any], count: int) -> None:
            # 整段接收：模型必须全部写完才有响应，客户端在拿到响应前一直干等。
            time.sleep(recorder.delay * count)
            content = json.dumps(
                [_translation(index) for index in range(count)], ensure_ascii=False
            )
            blob = json.dumps(
                {"choices": [{"message": {"content": content}}]}, ensure_ascii=False
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            if self._write(record, blob):
                record["sent_at"].append(time.monotonic())
                record["finished"] = True
            self.close_connection = True

    return Handler


class _Server:
    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(recorder))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *exc_info) -> bool:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        return False


def _run_once(
    *,
    base_url: str,
    items: int,
    batch_size: int,
    stream: bool,
    cancel_after: float,
) -> dict[str, Any]:
    """跑一次「翻译到一半点取消」，返回可对照的一组数字。"""
    engine = OpenAICompatTranslator(
        base_url=base_url,
        model="fake-model",
        api_key="local-test-key",
        timeout=30,
        batch_size=batch_size,
        stream=stream,
        echo_downgrade=False,  # 只验传输层，别让回抄救援混进来
    )
    cues = subtitle_io.parse_srt(build_srt(items))

    flag = threading.Event()
    marks: dict[str, float] = {}

    def cancel() -> None:
        marks["stop"] = time.monotonic()
        flag.set()

    timer = threading.Timer(cancel_after, cancel)
    timer.start()
    started = time.monotonic()
    outcome = "跑完了（取消没生效）"
    error = ""
    try:
        engine.translate_cues(
            cues,
            source_lang="ja",
            target_lang="zh-CN",
            batch_size=batch_size,
            should_stop=flag.is_set,
        )
    except TranslationCancelled as exc:
        outcome = "已取消"
        error = str(exc)
    except TranslationError as exc:
        outcome = "失败"
        error = str(exc)
    elapsed = time.monotonic() - started
    timer.cancel()

    latency = elapsed - cancel_after if "stop" in marks else float("nan")
    return {
        "outcome": outcome,
        "error": error,
        "elapsed": elapsed,
        "latency": latency,
        "done": sum(1 for cue in cues if cue.is_translated),
        "items": items,
        "stream": stream,
        "batch_size": batch_size,
    }


def _verdict(
    results: dict[str, dict[str, Any]], recorder: _Recorder, slow_start: float
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    stream = results["stream"]
    slow = results["stream-slow-start"]
    batched = results["whole-batched"]
    onebatch = results["whole-onebatch"]
    records = {row["stream"]: row for row in recorder.records}

    checks.append(
        (
            "流式：取消在分片之间生效（不等整批）",
            stream["latency"] < batched["latency"],
            f"流式 {stream['latency']:.2f}s vs 整批接收 {batched['latency']:.2f}s",
        )
    )
    checks.append(
        (
            "流式：模型先思考时，取消也能马上生效",
            slow["latency"] < slow_start / 2,
            f"服务端先睡 {slow_start:.1f}s 才吐第一个分片，取消延迟 {slow['latency']:.2f}s",
        )
    )
    checks.append(
        (
            "流式：取消时保住了已经收到的条目",
            stream["done"] > 0,
            f"已翻 {stream['done']}/{stream['items']} 条（整批接收时这个数只会是 0 或全部）",
        )
    )
    stream_record = records.get(True, {})
    checks.append(
        (
            "流式：取消的那一刻服务端确实没发完",
            bool(stream_record.get("sent_at")) and len(stream_record["sent_at"]) < stream["items"],
            f"服务端发了 {len(stream_record.get('sent_at', []))}/{stream['items']} 条时客户端就返回了",
        )
    )
    checks.append(
        (
            "整批接收：取消要等到这一批跑完",
            batched["latency"] > 0.1,
            f"延迟 {batched['latency']:.2f}s（≈ 一批剩下的时间）",
        )
    )
    checks.append(
        (
            "整批接收：只有一批时取消根本来不及",
            onebatch["outcome"].startswith("跑完"),
            f"{onebatch['outcome']}，耗时 {onebatch['elapsed']:.2f}s",
        )
    )
    broke = [row for row in recorder.records if row.get("broke_at") is not None]
    checks.append(
        (
            "客户端提前断开被服务端察觉",
            bool(broke) or not records.get(True),
            "服务端在写第 "
            f"{broke[0]['broke_at']} 条时报了连接断开" if broke else "（本轮没观察到写失败）",
        )
    )
    return checks


def run_local(args, out_lines: list[str]) -> int:
    recorder = _Recorder(args.items, args.delay)
    with _Server(recorder) as server:
        # (名字, 收法, 批大小, 「模型先思考」多久)
        plan = [
            ("stream", True, args.items, 0.0),
            ("stream-slow-start", True, args.items, args.slow_start),
            ("whole-batched", False, max(1, args.items // 3), 0.0),
            ("whole-onebatch", False, args.items, 0.0),
        ]
        results: dict[str, dict[str, Any]] = {}
        for name, stream, batch, first_delay in plan:
            recorder.first_delay = first_delay
            results[name] = _run_once(
                base_url=server.base_url,
                items=args.items,
                batch_size=batch,
                stream=stream,
                cancel_after=args.cancel_after,
            )
            time.sleep(0.2)  # 让服务端的记录写完

    out_lines.append("")
    out_lines.append(
        f"服务端：本地 http.server，{args.items} 条字幕、每条间隔 {args.delay:.2f}s"
        f"（整批要 {args.items * args.delay:.1f}s）；在第 {args.cancel_after:.2f}s 点取消"
    )
    out_lines.append("")
    out_lines.append(
        "场景 | 收法 | 批大小 | 结果 | 取消→返回 | 总耗时 | 已翻条目"
    )
    out_lines.append("-" * 78)
    for name, stream, batch, _first_delay in plan:
        row = results[name]
        latency = "—" if row["latency"] != row["latency"] else f"{row['latency']:.2f}s"
        out_lines.append(
            f"{name} | {'流式' if stream else '整段'} | {batch} | {row['outcome']} | "
            f"{latency} | {row['elapsed']:.2f}s | {row['done']}/{row['items']}"
        )
    out_lines.append("")
    out_lines.append("「取消→返回」= 从按下取消到 translate_cues 抛出来：")
    out_lines.append("  · 流式 —— 一个轮询间隔（0.2s），模型在思考也一样；")
    out_lines.append("  · 整段接收 —— 要等当前这一批请求整个返回；")
    out_lines.append("  · 整段接收 + 只有一批 —— 压根没机会中断，程序把整批翻完了。")
    out_lines.append("")
    out_lines.append("服务端记录（每条什么时刻发出去的）：")
    for index, row in enumerate(recorder.records):
        sent = row["sent_at"]
        window = (
            f"第 1 条 {sent[0] - row['started']:.2f}s、最后一条 {sent[-1] - row['started']:.2f}s"
            if sent
            else "一条都没发出去"
        )
        broke = (
            f"；写第 {row['broke_at']} 条时发现客户端已断开"
            if row.get("broke_at") is not None
            else ""
        )
        out_lines.append(
            f"  #{index + 1} {'流式' if row['stream'] else '整段'}"
            f"（{row['count']} 条）已写出 {len(sent)} 条　{window}"
            f"　{'全部发完' if row['finished'] else '被中途打断'}{broke}"
        )

    checks = _verdict(results, recorder, args.slow_start)
    out_lines.append("")
    failures = 0
    for label, passed, detail in checks:
        out_lines.append(f"  [{'PASS' if passed else 'FAIL'}] {label}　— {detail}")
        failures += 0 if passed else 1
    return failures


def run_live(rounds: int, out_lines: list[str]) -> int:
    """对真实中继量两件事：第一个分片多快到、整段多久收完。

    两者差得明显 → 中继真的在流式，取消就是「等到下一个分片」；
    几乎同时 → 它把整个回复憋到最后才发，流式的收益只剩「提前关连接」。
    """
    cfg = load_config().translation
    out_lines.append("")
    out_lines.append("=" * 96)
    out_lines.append(f"真实中继：model = {cfg.model}，{rounds} 条字幕、一条请求，stream=True")
    out_lines.append("=" * 96)
    if not cfg.model or not cfg.base_url:
        out_lines.append("  配置里没有 model / base_url，跳过。")
        return 0
    try:
        engine = OpenAICompatTranslator.from_config(cfg)
    except Exception as exc:  # noqa: BLE001 - 密钥没配好也算「跳过」
        out_lines.append(f"  构造引擎失败（{type(exc).__name__}: {exc}），跳过。")
        return 0

    marks: dict[str, float | None] = {"first": None}
    started = time.monotonic()
    original = openai_compat.iter_sse_deltas

    def timed(resp, **kwargs):
        # 透传全部关键字参数（stop / poll / stall_timeout），免得引擎签名一变就报错
        for delta in original(resp, **kwargs):
            if marks["first"] is None:
                marks["first"] = time.monotonic()
            yield delta

    openai_compat.iter_sse_deltas = timed  # type: ignore[assignment]
    try:
        cues = subtitle_io.parse_srt(build_srt(rounds))
        started = time.monotonic()
        engine.translate_cues(
            cues, source_lang="ja", target_lang="zh-CN", batch_size=rounds
        )
        elapsed = time.monotonic() - started
        out_lines.append(f"  整段收完 {elapsed:.2f}s，已翻 {sum(c.is_translated for c in cues)}/{rounds} 条")
        if marks["first"] is not None:
            first = marks["first"] - started
            out_lines.append(f"  第一个分片 {first:.2f}s 到（占整段的 {first / elapsed:.0%}）")
            if first < elapsed * 0.5:
                out_lines.append("  → 中继确实在流式：取消的生效延迟 ≈ 一个轮询间隔。")
            else:
                out_lines.append(
                    "  → 第一个分片来得晚（该中继慢启动），但取消不必等它 —— "
                    "打开连接和读都在后台线程里，延迟仍是一个轮询间隔。"
                )
        else:
            out_lines.append("  没收到任何 SSE 分片 —— 它没理会 stream=True。")
        out_lines.append(f"  降级次数 stream_fallback_count = {engine.stream_fallback_count}")
        if engine.stream_fallback_count:
            out_lines.append(
                "  → 有请求自动退回了整段接收（中继忽略 stream、中途断流、或偶发空流），"
                "翻译本身不受影响；这是 _STREAM / 降级路径起作用，不是失败。"
            )

        # 第二轮：翻到一半取消，量真实中继上的取消延迟
        out_lines.append("")
        out_lines.append("  再跑一次并在 0.5s 后取消：")
        cues2 = subtitle_io.parse_srt(build_srt(rounds))
        flag = threading.Event()
        threading.Timer(0.5, flag.set).start()
        t0 = time.monotonic()
        outcome = "跑完了（取消没赶上）"
        try:
            engine.translate_cues(
                cues2,
                source_lang="ja",
                target_lang="zh-CN",
                batch_size=rounds,
                should_stop=flag.is_set,
            )
        except TranslationCancelled as exc:
            outcome = "已取消"
            out_lines.append(f"    {exc}")
        except TranslationError as exc:
            outcome = f"失败：{exc}"
        elapsed2 = time.monotonic() - t0
        kept = sum(1 for c in cues2 if c.is_translated)
        out_lines.append(
            f"    {outcome}，耗时 {elapsed2:.2f}s，保住 {kept}/{rounds} 条已翻好的字幕"
        )
    finally:
        openai_compat.iter_sse_deltas = original  # type: ignore[assignment]
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="流式接收让取消更快生效的验证")
    parser.add_argument("--items", type=int, default=12, help="一次请求里放多少条字幕")
    parser.add_argument("--delay", type=float, default=0.25, help="服务端每条之间的间隔（秒）")
    parser.add_argument("--cancel-after", type=float, default=0.6, help="第几秒点取消")
    parser.add_argument(
        "--slow-start", type=float, default=2.0,
        help="「模型先思考」多久才吐第一个分片（秒）",
    )
    parser.add_argument("--live", type=int, default=0, help="对真实中继跑 N 条（消耗配额）")
    parser.add_argument("--out", default="output/verify_stream_cancel.txt")
    args = parser.parse_args()

    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("流式接收 —— 取消要等多久才生效（本地真实 HTTP，零配额）")
    lines.append("=" * 96)
    lines.append("服务端真的按 SSE 分片吐字，客户端是真的 urllib，走完整的 translate_cues。")
    lines.append("只验证传输层：echo_downgrade=False，排除回抄救援的干扰。")

    failures = run_local(args, lines)

    lines.append("")
    lines.append("=" * 96)
    lines.append(f"本地场景：{'全部符合预期' if failures == 0 else f'{failures} 项不符合预期'}")
    lines.append("=" * 96)
    lines.append("注：「服务端随之停止生成」在本地只能证明到「客户端确实提前断开、不再读」；")
    lines.append("    真实中继上是否立刻停手，看 --live 那一轮的输出。")

    if args.live:
        run_live(args.live, lines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
