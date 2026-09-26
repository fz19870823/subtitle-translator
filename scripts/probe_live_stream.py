# -*- coding: utf-8 -*-
"""对真实中继做 A/B：SSE 读取放在后台线程，和放在主线程，结果一样吗。

背景：让「点取消」在模型思考期间也能生效，靠的是把读取挪到后台线程、调用方只在
队列上等轮询间隔（``_StreamReader``）。改动后本地 ``http.server`` 全部通过，但真实
中继报「声明了事件流却没有任何内容」。同一份请求在改动前（主线程同步读）是好的，
所以嫌疑落在「在线程里读 ``HTTPResponse``」这件事本身。

本脚本把两种读法并排跑一遍，各发一个请求：

  A  后台线程 + 队列（当前实现，走 ``iter_sse_deltas``）
  B  主线程 ``readline``（改动前的读法）

判定：B 有内容而 A 没有 → 线程读这条路线在真实中继上不成立，得换唤醒手段；
      A、B 都有内容 → 线上那次是别的原因（中继本轮抽风 / 回抄）。

用法（会真的发 2 个请求）::

    python scripts/probe_live_stream.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.core.engines.openai_compat as oc  # noqa: E402
from app.config import load_config  # noqa: E402

PROMPT = (
    "把下面这个 JSON 数组里的每一条翻译成简体中文，"
    "保持数组形状与条数不变，只输出 JSON 数组："
    '["こんにちは","さようなら","ありがとう"]'
)


def _body(engine) -> dict:
    return {
        "model": engine.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "stream": True,
        "max_tokens": 512,
    }


def _open(engine):
    req = engine._build_request("/chat/completions", _body(engine))
    return urllib.request.urlopen(req, timeout=int(engine.timeout))


def route_a(engine) -> None:
    """当前实现：后台线程读 + 队列。"""
    print("== A：后台线程读（iter_sse_deltas）==")
    resp = _open(engine)
    started = time.monotonic()
    try:
        deltas = list(
            oc.iter_sse_deltas(resp, stop=lambda: False, poll=0.2, stall_timeout=60.0)
        )
    except Exception as exc:  # noqa: BLE001 - 探针：什么错都要看见
        print(f"   FAIL {time.monotonic() - started:.2f}s  {type(exc).__name__}: {exc}")
        return
    text = "".join(deltas)
    print(
        f"   {time.monotonic() - started:.2f}s  分片 {len(deltas)} 个  "
        f"内容 {text[:90]!r}"
    )


def route_b(engine) -> None:
    """改动前的读法：主线程 readline。"""
    print("== B：主线程 readline（改动前的读法）==")
    resp = _open(engine)
    started = time.monotonic()
    lines = 0
    parts: list[str] = []
    try:
        while True:
            raw = resp.readline(1 << 20)
            if not raw:
                break
            lines += 1
            text = raw.decode("utf-8", "replace").strip()
            if text.startswith("data:"):
                data = text[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    parts.extend(oc._deltas_from(json.loads(data)))
                except json.JSONDecodeError:
                    pass
    except Exception as exc:  # noqa: BLE001
        print(f"   FAIL {time.monotonic() - started:.2f}s  {type(exc).__name__}: {exc}")
        return
    print(
        f"   {time.monotonic() - started:.2f}s  原始行 {lines} 行  "
        f"内容 {''.join(parts)[:90]!r}"
    )


def main() -> int:
    cfg = load_config().translation
    print("=" * 88)
    print(f"真实中继 A/B：model = {cfg.model}，stream = {getattr(cfg, 'stream', '?')}")
    print("=" * 88)
    if not cfg.model or not cfg.base_url:
        print("  配置里没有 model / base_url，跳过。")
        return 0
    engine = oc.OpenAICompatTranslator.from_config(cfg)
    if not engine.stream:
        print("  配置里 stream 是关的，先把 A/B 都按流式跑：")
    engine.stream = True

    route_a(engine)
    route_b(engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
