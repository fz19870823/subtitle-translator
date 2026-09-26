"""探针：流式读取时，「读阻塞期间查不到取消」这件事能不能解决。

**背景（真实中继实测）**：模型可能先「想」 2.5–5.7 秒才吐第一个分片，
占整段耗时的 45%。而取消检查原本写在 ``readline`` 之前 —— 读阻塞期间根本
轮不到它，于是「点取消」要干等到模型开口（实测 2.0s，整段则是 12.5s）。

**本脚本记录四条路线的实测结论**（A/B/C 在本机 3.13.12 / 3.14.7 结果一致）：

===  ==========================================  ====================================
路线  做法                                          结果
===  ==========================================  ====================================
A     只把 socket 读超时调短，超时后重试                ✗ 一次超时后 socket 对象作废：
                                                    ``cannot read from timed out object``
B     看门线程 ``sock.shutdown(SHUT_RDWR)``          ✗ 本机不唤醒阻塞的 recv
                                                    （要等对端下一次写入撞上 RST，实测 6.0s）
C     A + 重置 ``SocketIO._timeout_occurred``        ✗ **只在明文 HTTP 上成立**。真实中继走
                                                    TLS：读超时 12 次后于 t=2.55s 提前收到
                                                    EOF，整段内容一个字都收不到；同一份请求
                                                    完全不碰超时则 4.65s 正常返回
D     读取整行放进后台线程，调用方只在队列上等            ✓ HTTP / HTTPS 一致，取消延迟 = 轮询
                                                    间隔。**最终采用**
===  ==========================================  ====================================

引擎走的是 D（``_StreamReader``）。C 的失败证据见 ``probe_live_stream.py``（同一个中继
上的 A/B 对照）与 ``diag_https_stream.py``（失败那一刻的内部状态）；本脚本保留 A/C 的
复现，好让「为什么不能用『调短超时』这条路」一直可复验。

⚠️ 另外两个坑，都在 ``diag_cancel_latency.py`` 里量过（本地服务端只睡 2.0s）：

1. **关连接不能同步做** —— 读线程阻塞在 ``readline`` 时握着 ``BufferedReader`` 的内部锁，
   ``close()`` 要拿同一把锁，于是取消会卡在收尾上直到读回数据（实测延迟 1.68s，
   期间取消回调一次都没再被调用）。关闭丢给后台线程。
2. **打开连接也不能同步做** —— 中继会把响应头憋到第一个分片就绪（实测 ``urlopen``
   阻塞 3.55s），这几秒里同步的 ``urlopen`` 让取消检查根本没机会跑（线上实测 0.5s 点取消、
   3.5s 才返回）。打开动作同样挪进线程。

两条都挪走之后，本地 0.03–0.14s、真实中继 0.62s 返回（点取消那一刻起算 0.12s）。

用法::

    python scripts/probe_stream_read.py            # 跑路线 C
    python scripts/probe_stream_read.py --route a  # 复现路线 A 的失败
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

#: 「模型先想这么久」才吐第一个分片 —— 用真实中继上实测到的量级
FIRST_FRAGMENT_DELAY = 2.0
#: 一次写入几行。多写几行是为了验证「超时重读之后，缓冲区里没消费完的行没丢」
FRAGMENTS = 3


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # noqa: D102 - 静音日志
        pass

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的约定
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(FIRST_FRAGMENT_DELAY)
        blob = b"".join(
            (
                "data: "
                + json.dumps(
                    {"choices": [{"delta": {"content": f"第{index}条"}}]},
                    ensure_ascii=False,
                )
                + "\n\n"
            ).encode("utf-8")
            for index in range(1, FRAGMENTS + 1)
        )
        try:
            self.wfile.write(blob)
            self.wfile.flush()
            time.sleep(0.1)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError as exc:
            print(f"  [服务端] 写入失败 → {type(exc).__name__}（说明客户端提前走了）")
        self.close_connection = True


def _serve() -> tuple[ThreadingHTTPServer, str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions"
    return httpd, url


def _open(url: str):
    req = urllib.request.Request(
        url, data=json.dumps({"stream": True}).encode("utf-8"), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    return urllib.request.urlopen(req, timeout=30)


def route_a() -> int:
    """只调短读超时，超时后直接重试 —— 复现失败。"""
    httpd, url = _serve()
    resp = _open(url)
    sock = resp.fp.raw._sock
    sock.settimeout(0.3)
    started = time.monotonic()
    try:
        while True:
            try:
                line = resp.readline(1 << 20)
            except TimeoutError:
                print(f"  t={time.monotonic() - started:5.2f}s 读超时，直接重试…")
                continue
            if not line:
                break
    except OSError as exc:
        print(f"  t={time.monotonic() - started:5.2f}s 抛 {type(exc).__name__}: {exc}")
        print("=> 路线 A 不可行：超时一次后对象作废，后续读全部失败")
        httpd.shutdown()
        return 1
    print("=> 路线 A 竟然通过了（与实测不符，说明解释器行为变了）")
    httpd.shutdown()
    return 0


def route_c() -> int:
    """调短读超时 + 每次超时后重置 ``_timeout_occurred``。"""
    httpd, url = _serve()
    resp = _open(url)
    io = getattr(getattr(resp, "fp", None), "raw", None)
    sock = getattr(io, "_sock", None)
    print(f"  resp.fp = {type(resp.fp).__name__} | .raw = {type(io).__name__}"
          f" | .raw._sock = {type(sock).__name__ if sock else None}")
    if sock is None or not hasattr(io, "_timeout_occurred"):
        print("=> 取不到底层 socket / 没有 _timeout_occurred，引擎会退回老行为")
        httpd.shutdown()
        return 1

    previous = sock.gettimeout()
    sock.settimeout(0.3)
    started = time.monotonic()
    timeouts = 0
    seen: list[str] = []
    while True:
        try:
            line = resp.readline(1 << 20)
        except TimeoutError:
            timeouts += 1
            io._timeout_occurred = False  # ← 路线 C 的全部内容
            if timeouts <= 3:
                print(f"  t={time.monotonic() - started:5.2f}s 读超时（第 {timeouts} 次）"
                      " → 重置标记后接着读")
            continue
        if not line:
            print(f"  t={time.monotonic() - started:5.2f}s 读到 EOF")
            break
        text = line.decode("utf-8", "replace").strip()
        if text:
            seen.append(text)
            print(f"  t={time.monotonic() - started:5.2f}s 收到: {text[:52]}")

    sock.settimeout(previous)
    httpd.shutdown()
    expected = FRAGMENTS + 1  # 内容行 + [DONE]
    ok = len(seen) == expected
    print(f"\n  超时 {timeouts} 次，完整读到 {len(seen)} 行（期望 {expected}）")
    print("=> 路线 C 可行" if ok else "=> 行数不对，缓冲区在超时重读时丢了数据")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="流式读取的取消响应能力探针")
    parser.add_argument("--route", choices=("a", "c"), default="c", help="要复现哪条路线")
    args = parser.parse_args()

    print("=" * 88)
    print("探针：模型先「想」一会儿时，读阻塞期间能不能查到取消？")
    print(f"      服务端会先睡 {FIRST_FRAGMENT_DELAY}s 才吐第一个分片")
    print("=" * 88)
    code = route_a() if args.route == "a" else route_c()
    print()
    print("结论要写进引擎的注释里：取消的生效延迟 = 一个轮询间隔，而不是「等首个分片」。")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
