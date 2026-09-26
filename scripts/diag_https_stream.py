# -*- coding: utf-8 -*-
"""诊断：把「调短 socket 读超时」用在真实 HTTPS 中继上时，到底哪里坏了。

已经确定的对照结果（``scripts/probe_live_stream.py``）：同一个中继、同一份请求，

  不动 socket 超时      → 4.65s 正常拿到 ``["你好","再见","谢谢"]``
  调短读超时 + 重置标记  → 3.82s 报「声明了事件流却没有任何内容」

而路线 C 在**本地明文 HTTP** 上是好的（``scripts/probe_stream_read.py``）。
所以分水岭是 TLS。本脚本把失败那一刻的内部状态一次打清楚：
``resp.fp`` 还在不在、``resp.closed``、``resp.length``、异常类型，
以及「超时后还能不能接着读」。

用法（会真的发 1 个请求）::

    python scripts/diag_https_stream.py
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.core.engines.openai_compat as oc  # noqa: E402
from app.config import load_config  # noqa: E402

BODY = {
    "model": "",
    "messages": [
        {
            "role": "user",
            "content": (
                "把下面这个 JSON 数组里的每一条翻译成简体中文，"
                "只输出 JSON 数组：" '["こんにちは","さようなら"]'
            ),
        }
    ],
    "temperature": 0,
    "stream": True,
    "max_tokens": 256,
}
POLL = 0.2


def main() -> int:
    cfg = load_config().translation
    if not cfg.model or not cfg.base_url:
        print("配置里没有 model / base_url，跳过。")
        return 0
    engine = oc.OpenAICompatTranslator.from_config(cfg)
    BODY["model"] = engine.model

    req = engine._build_request("/chat/completions", BODY)
    url = f"{engine.base_url}/chat/completions"
    print("=" * 88)
    print(f"真实中继：{url}  model={engine.model}")
    print("=" * 88)

    t_open = time.monotonic()
    resp = urllib.request.urlopen(req, timeout=engine.timeout)
    print(f"  urlopen 返回耗时  {time.monotonic() - t_open:.2f}s")
    fp0 = resp.fp
    io = getattr(getattr(resp, "fp", None), "raw", None)
    sock = getattr(io, "_sock", None)
    print(f"  resp            = {type(resp).__module__}.{type(resp).__name__}")
    print(f"  resp.fp         = {type(fp0).__module__}.{type(fp0).__name__}")
    print(f"  resp.fp.raw     = {type(io).__module__}.{type(io).__name__}")
    print(f"  resp.fp.raw._sock = {type(sock).__module__}.{type(sock).__name__}")
    print(f"  Content-Type    = {resp.headers.get('Content-Type')!r}")
    print(f"  chunked/length  = {resp.chunked!r} / {resp.length!r}")
    if sock is None:
        print("=> 取不到底层 socket，引擎会退回老行为。")
        return 1

    previous = sock.gettimeout()
    sock.settimeout(POLL)
    started = time.monotonic()
    timeouts = 0
    lines = 0
    try:
        for _ in range(2000):
            try:
                raw = resp.readline(1 << 20)
            except TimeoutError:
                timeouts += 1
                # 路线 C 的全部内容
                if io is not None and hasattr(io, "_timeout_occurred"):
                    io._timeout_occurred = False
                if timeouts <= 3 or timeouts % 20 == 0:
                    print(
                        f"  t={time.monotonic() - started:5.2f}s 读超时 #{timeouts}"
                        f"  resp.fp={'None' if resp.fp is None else type(resp.fp).__name__}"
                        f"  closed={resp.closed}"
                    )
                continue
            except OSError as exc:
                print(
                    f"  t={time.monotonic() - started:5.2f}s "
                    f"抛 {type(exc).__name__}: {exc}"
                )
                print(
                    f"     此刻 resp.fp={'None' if resp.fp is None else type(resp.fp).__name__}"
                    f"  closed={resp.closed}"
                )
                break
            if not raw:
                print(f"  t={time.monotonic() - started:5.2f}s 读到 EOF（空）")
                break
            lines += 1
            text = raw.decode("utf-8", "replace").strip()
            if text:
                print(f"  t={time.monotonic() - started:5.2f}s 收到: {text[:70]}")
    finally:
        try:
            sock.settimeout(previous)
        except OSError:
            pass
        try:
            resp.close()
        except OSError:
            pass

    print(
        f"\n  结果：超时 {timeouts} 次，读到 {lines} 行（含空行）。"
        f"  resp.fp={'None' if resp.fp is None else '还在'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
