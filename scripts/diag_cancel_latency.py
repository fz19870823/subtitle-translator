# -*- coding: utf-8 -*-
"""诊断：流式接收时，「模型先思考」那几秒里取消检查到底有没有在跑。

``scripts/verify_stream_cancel.py`` 的 ``stream-slow-start`` 场景实测取消延迟
1.68s（服务端只睡 2.0s），而普通流式场景只有 0.17s —— 两者用的都是同一套
``_STREAM_POLL = 0.2`` 轮询，差值不该存在。

本脚本给 ``should_stop`` 回调插桩，把「它被调用的时刻」和「抛出时刻」全部记下来，
一次看清是「没被调用」还是「调用了但返回 False」，还是「卡在别的地方」。

用法::

    python scripts/diag_cancel_latency.py            # 服务端先睡 2.0s
    python scripts/diag_cancel_latency.py --sleep 0  # 对照：不睡
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.core.engines.openai_compat as oc  # noqa: E402
from app.core.subtitle_io import parse_srt  # noqa: E402
from app.core.translator import TranslationCancelled, TranslationError  # noqa: E402

ITEMS = 6
DELAY = 0.25
CANCEL_AFTER = 0.6
FIRST_DELAY = 2.0


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    first_delay = FIRST_DELAY
    #: (名字, 时刻) —— 服务端的时间线，用来对齐客户端
    events: list[tuple[str, float]] = []
    t0 = 0.0

    def log_message(self, *args) -> None:  # noqa: D102 - 静音
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        type(self).events.append(("收到请求", time.monotonic() - self.t0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        type(self).events.append(("响应头已发", time.monotonic() - self.t0))
        if self.first_delay:
            time.sleep(self.first_delay)
            type(self).events.append(("思考结束", time.monotonic() - self.t0))
        for index in range(ITEMS):
            if index:
                time.sleep(DELAY)
            prefix = "[" if index == 0 else ""
            suffix = "]" if index == ITEMS - 1 else ", "
            payload = {
                "choices": [{"delta": {"content": f'{prefix}"译{index}"{suffix}'}}]
            }
            line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            try:
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            except OSError as exc:
                type(self).events.append((f"写失败({type(exc).__name__})", time.monotonic() - self.t0))
                self.close_connection = True
                return
            type(self).events.append((f"发第{index + 1}条", time.monotonic() - self.t0))
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass
        self.close_connection = True


def _srt(count: int) -> str:
    blocks = []
    for index in range(1, count + 1):
        start = f"00:00:{index:02d},000"
        end = f"00:00:{index + 1:02d},000"
        blocks.append(f"{index}\n{start} --> {end}\n原文第 {index} 条\n")
    return "\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description="取消检查在「模型思考」期间的运行情况")
    parser.add_argument("--sleep", type=float, default=FIRST_DELAY, help="服务端先睡多久")
    args = parser.parse_args()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"

    engine = oc.OpenAICompatTranslator(
        base_url=base, api_key="x", model="local-probe", timeout=300
    )

    calls: list[float] = []
    flag = threading.Event()
    started = time.monotonic()
    _Handler.t0 = started
    _Handler.first_delay = args.sleep
    _Handler.events = []

    def should_stop() -> bool:
        calls.append(time.monotonic() - started)
        return flag.is_set()

    cancel_at: list[float] = []

    def fire() -> None:
        cancel_at.append(time.monotonic() - started)
        flag.set()
        print(f"  [{CANCEL_AFTER:.2f}s] 用户点了取消")

    threading.Timer(CANCEL_AFTER, fire).start()

    cues = parse_srt(_srt(ITEMS))
    outcome = ""
    try:
        engine.translate_cues(
            cues, source_lang="ja", target_lang="zh-CN",
            batch_size=ITEMS, should_stop=should_stop,
        )
        outcome = "跑完了（取消没生效）"
    except TranslationCancelled:
        outcome = "已取消"
    except TranslationError as exc:
        outcome = f"失败：{exc}"
    elapsed = time.monotonic() - started

    print("=" * 84)
    print(f"服务端先睡 {args.sleep:.1f}s，客户端在 {CANCEL_AFTER:.2f}s 点取消")
    print("=" * 84)
    print(f"  结果：{outcome}，总耗时 {elapsed:.2f}s")
    print(f"  should_stop 一共被调用 {len(calls)} 次，时刻：")
    print("   " + " ".join(f"{t:.2f}" for t in calls))
    if calls:
        gaps = [calls[i + 1] - calls[i] for i in range(len(calls) - 1)]
        if gaps:
            print(f"  相邻间隔最大 {max(gaps):.2f}s")
    print("  服务端时间线：")
    for name, at in _Handler.events:
        print(f"    {at:6.2f}s  {name}")
    if cancel_at:
        print(f"  取消时刻：{cancel_at[0]:.2f}s　→ 延迟 {elapsed - cancel_at[0]:.2f}s")
    httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
