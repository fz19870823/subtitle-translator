"""验证「链路临时故障自动重试」：本地起 HTTP 服务，真的返回 502。

为什么不用桩：桩只能证明「我写的分支被走到了」，证明不了 urllib 抛出的
``HTTPError`` 长什么样、连接复不复用、退避到底等了多久。这里起一个真实的
``http.server``，用真实的 ``urllib`` 走完整的 ``translate_batch``，
只在「服务端怎么回答」这一点上做手脚。

四个场景（都在本机，零配额）：

===============  ============================================  ==========================
场景              服务端行为                                     期望
===============  ============================================  ==========================
``flaky``        前 2 次 502，之后正常                          自动重试后成功，共 3 次请求
``always-502``   一直 502                                       重试 3 次后报错，共 4 次请求
``401``          一直 401                                       一次都不重试（确定性失败）
``retry-after``  第一次 503 + ``Retry-After: 1``，之后正常      等满 1 秒再重发，成功
``models``       ``/models`` 前 2 次 502，之后正常              拉模型列表同样能扛过 502
===============  ============================================  ==========================

``--live N`` 另跑一件事：对配置里的真实中继连发 N 个请求，
统计状态码分布 —— 用来判断现实中 502 到底多不多，重试次数该给几。

用法::

    python scripts/verify_http_retry.py
    python scripts/verify_http_retry.py --live 16
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config  # noqa: E402
from app.core.engines.openai_compat import OpenAICompatTranslator, fetch_models  # noqa: E402
from app.core.translator import TranslationError, TranslationRequest  # noqa: E402

#: 假译文：必须是「目标语言的书写族」，否则会被回抄防护当成没翻
#: （源语言是日文假名，译文带汉字，正好是正常的跨语言结果）。
FAKE_TRANSLATION = "这是第 {} 条译文。"

JA_TEXTS = [
    "何度も言ったはずだ。",
    "そんなの無理だよ。",
    "お前、何を考えてるんだ？",
]


class _Scenario:
    """服务端的行为脚本 + 收到的请求记录。"""

    def __init__(self, mode: str, *, fail_times: int = 2, retry_after: str | None = None) -> None:
        self.mode = mode
        self.fail_times = fail_times
        self.retry_after = retry_after
        self.hits = 0
        self.log: list[dict[str, Any]] = []

    def next_status(self, path: str) -> tuple[int, dict[str, str]]:
        self.hits += 1
        headers: dict[str, str] = {}
        if self.mode == "always-502":
            status = 502
        elif self.mode == "401":
            status = 401
        elif self.mode == "flaky":
            status = 502 if self.hits <= self.fail_times else 200
        elif self.mode == "retry-after":
            if self.hits == 1:
                status = 503
                headers["Retry-After"] = self.retry_after or "1"
            else:
                status = 200
        elif self.mode == "models":
            status = 502 if path.endswith("/models") and self.hits <= self.fail_times else 200
        else:  # pragma: no cover - 参数由下面的场景表固定
            raise AssertionError(f"未知场景 {self.mode}")
        return status, headers


def _make_handler(scenario: _Scenario):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # noqa: D102 - 静音默认的 stderr 日志
            pass

        def _respond(self, status: int, payload: dict, headers: dict[str, str]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _record(self, path: str, status: int) -> None:
            scenario.log.append({"path": path, "status": status, "at": time.monotonic()})

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的约定
            status, extra = scenario.next_status(self.path)
            self._record(self.path, status)
            if status == 200:
                self._respond(200, {"data": [{"id": "fake-model"}]}, extra)
            else:
                self._respond(status, {"error": {"code": "gateway", "message": "upstream down"}}, extra)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)  # 必须读完，否则下一个请求会读到脏数据
            status, extra = scenario.next_status(self.path)
            self._record(self.path, status)
            if status != 200:
                self._respond(
                    status,
                    {"error": {"code": "gateway", "message": "upstream down"}},
                    extra,
                )
                return
            texts = _extract_texts(raw)
            translated = [
                FAKE_TRANSLATION.format(index + 1) for index in range(len(texts))
            ]
            body = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(translated, ensure_ascii=False)
                        }
                    }
                ]
            }
            self._respond(200, body, extra)

    return Handler


def _extract_texts(raw: bytes) -> list[str]:
    """从请求体里取出待翻译的数组（user 消息的内容）。"""
    try:
        payload = json.loads(raw.decode("utf-8"))
        content = payload["messages"][-1]["content"]
        if content.startswith("源语言: "):
            content = content.split("\n", 1)[1]
        value = json.loads(content)
        return value if isinstance(value, list) else []
    except Exception:  # noqa: BLE001 - 只是假服务端，取不出来就按 0 条算
        return []


class _Server:
    """上下文管理器：起服务、给出 base_url、退出时关掉。"""

    def __init__(self, scenario: _Scenario) -> None:
        self.scenario = scenario
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(scenario))
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


def _engine_for(base_url: str, *, http_retries: int | None = None) -> OpenAICompatTranslator:
    kwargs: dict[str, Any] = dict(
        base_url=base_url,
        model="fake-model",
        api_key="local-test-key",
        timeout=10,
        batch_size=len(JA_TEXTS),
        echo_downgrade=False,  # 这里只验传输层，别让回抄救援混进来
    )
    if http_retries is not None:
        kwargs["http_retries"] = http_retries
    return OpenAICompatTranslator(**kwargs)


def _requests() -> list[TranslationRequest]:
    return [
        TranslationRequest(text=t, source_lang="ja", target_lang="zh-CN")
        for t in JA_TEXTS
    ]


def _scenarios() -> list[dict[str, Any]]:
    return [
        {
            "name": "flaky",
            "mode": "flaky",
            "expect": "自动重试后成功",
            "retries": 2,
        },
        {
            "name": "always-502",
            "mode": "always-502",
            "expect": "重试耗尽后报错",
            "retries": 3,
        },
        {
            "name": "401",
            "mode": "401",
            "expect": "确定性失败，一次都不重试",
            "retries": None,  # 用默认值
        },
        {
            "name": "retry-after",
            "mode": "retry-after",
            "expect": "等满 Retry-After 再重发",
            "retries": None,
        },
    ]


def run_local(out_lines: list[str]) -> int:
    """四个翻译场景 + 一个拉模型列表场景，全部走真实 HTTP。"""
    failures = 0
    for spec in _scenarios():
        scenario = _Scenario(spec["mode"], retry_after="1")
        with _Server(scenario) as server:
            engine = _engine_for(server.base_url, http_retries=spec["retries"])
            started = time.monotonic()
            error: str | None = None
            delivered: list[str] | None = None
            try:
                delivered = engine.translate_batch(_requests())
            except TranslationError as exc:
                error = str(exc)
            elapsed = time.monotonic() - started

        statuses = [row["status"] for row in scenario.log]
        expected_requests = {
            "flaky": 3,
            "always-502": 4,  # 1 次原始 + 3 次重试
            "401": 1,
            "retry-after": 2,
        }[spec["name"]]
        checks: list[tuple[str, bool, str]] = []

        if spec["mode"] == "flaky":
            ok = delivered is not None and len(delivered) == len(JA_TEXTS)
            checks.append(("交付了完整译文", ok, f"delivered={delivered}"))
        elif spec["mode"] == "always-502":
            ok = error is not None and "502" in error and "已自动重试" in error
            checks.append(("报错而不是静默少翻", ok, error or "(没有报错，反而返回了结果)"))
        elif spec["mode"] == "401":
            ok = error is not None and "401" in error and "已自动重试" not in error
            checks.append(("报错且未重试", ok, error or "(没有报错)"))

        ok = len(scenario.log) == expected_requests
        checks.append((f"请求次数 == {expected_requests}", ok, f"实际 {len(scenario.log)} 次：{statuses}"))

        ok = engine.http_retry_count == expected_requests - 1
        checks.append(("重试计数正确", ok, f"http_retry_count={engine.http_retry_count}"))

        if spec["mode"] == "retry-after":
            ok = elapsed >= 1.0
            checks.append(("确实等满了 Retry-After=1s", ok, f"耗时 {elapsed:.2f}s"))
            ok = delivered is not None
            checks.append(("等待后重发成功", ok, str(delivered)))

        out_lines.append(f"\n### 场景 {spec['name']}　（期望：{spec['expect']}）")
        out_lines.append(
            f"请求序列：{' → '.join(f'#{i + 1} {s}' for i, s in enumerate(statuses))}"
            f"　|　耗时 {elapsed:.2f}s　|　http_retries={engine.http_retries}"
            f"　|　重试 {engine.http_retry_count} 次 {engine.http_retry_reasons}"
        )
        if error:
            out_lines.append(f"错误信息：{error}")
        for label, passed, detail in checks:
            flag = "PASS" if passed else "FAIL"
            out_lines.append(f"  [{flag}] {label}　— {detail}")
            if not passed:
                failures += 1

    # 拉模型列表也要能扛过 502
    scenario = _Scenario("models")
    with _Server(scenario) as server:
        try:
            models = fetch_models(server.base_url, "local-test-key", timeout=10)
            error = None
        except TranslationError as exc:
            models, error = [], str(exc)
    out_lines.append("\n### 场景 models　（期望：拉列表同样重试后成功）")
    out_lines.append(
        f"请求序列：{' → '.join(f'#{i + 1} {r["status"]}' for i, r in enumerate(scenario.log))}"
    )
    passed = models == ["fake-model"] and len(scenario.log) == 3
    out_lines.append(
        f"  [{'PASS' if passed else 'FAIL'}] 两次 502 后拿到模型列表　— "
        f"models={models} 请求数={len(scenario.log)} 错误={error}"
    )
    if not passed:
        failures += 1
    return failures


def run_live(rounds: int, out_lines: list[str]) -> int:
    """对真实中继采样，回答两个问题：链路故障多不多、回抄多不多。

    刻意把 http_retries 设成 0：要看的是**原始** HTTP 结果，
    开着重试就把故障藏起来了。同时数一遍 ``_post`` 的调用次数 ——
    一轮可能发多个请求（回抄重发），"16 轮"不等于"16 个请求"。
    """
    cfg = load_config().translation
    engine = OpenAICompatTranslator.from_config(cfg)
    engine.http_retries = 0      # 暴露原始状态码
    engine.echo_downgrade = False  # 回抄判定保留，但别让逐条救援掩盖频率

    posted = {"n": 0}
    original_post = engine._post

    def counting_post(path, body):
        posted["n"] += 1
        return original_post(path, body)

    engine._post = counting_post  # type: ignore[method-assign]

    out_lines.append("\n" + "=" * 96)
    out_lines.append(
        f"真实中继采样：{rounds} 轮（每轮 1 条字幕，model = {engine.model}，http_retries=0）"
    )
    out_lines.append(
        "每轮可能发出多个请求（回抄会触发重发），所以「轮次」与「请求数」要分开看。"
    )
    out_lines.append("=" * 96)

    status_counts: dict[str, int] = {}
    for index in range(rounds):
        req = [TranslationRequest(text=JA_TEXTS[0], source_lang="ja", target_lang="zh-CN")]
        started = time.monotonic()
        detail = ""
        try:
            engine.translate_batch(req)
            label = "交付正常译文"
        except TranslationError as exc:
            message = str(exc)
            if message.startswith("HTTP "):
                label = "HTTP " + message.split()[1]
            elif "超时" in message:
                label = "超时"
            elif message.startswith("连接 "):
                label = "连接失败"
            elif "完全相同" in message:
                # 这是回抄（请求被处理了但没翻译），**不是**链路故障。
                # 早期版本把它笼统算成"连接错误"，得出了 87.5% 链路故障的假结论。
                label = "回抄（未翻译）"
            else:
                label = "响应异常"
            detail = message.replace("\n", " ")[:90]
        elapsed = time.monotonic() - started
        status_counts[label] = status_counts.get(label, 0) + 1
        out_lines.append(f"  #{index + 1:<3} {label:<14} {elapsed:>5.2f}s  {detail}")

    out_lines.append("\n分布（按轮次）：")
    for label, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        out_lines.append(f"  {label:<16} {count:>3} / {rounds}　（{count / rounds * 100:.1f}%）")

    link_labels = ("连接失败", "超时")
    link_faults = sum(
        count
        for label, count in status_counts.items()
        if label.startswith("HTTP ") or label in link_labels
    )
    echoed = status_counts.get("回抄（未翻译）", 0)
    out_lines.append(f"\n共发出 {posted['n']} 个 HTTP 请求。")
    out_lines.append(
        f"链路故障（HTTP 状态码 / 连接失败 / 超时）：{link_faults} / {rounds} 轮"
        f"（{link_faults / rounds * 100:.1f}%）"
    )
    out_lines.append(
        f"回抄（请求被处理但没翻译）：{echoed} / {rounds} 轮（{echoed / rounds * 100:.1f}%）"
        "—— 这是另一个问题，由 _recover_echoes 处理，不靠重发同一条请求。"
    )
    if link_faults == 0:
        out_lines.append(
            "\n本轮没撞上链路故障：说明 502 这类故障不是稳定复现的，"
            "http_retries 的取值只能按常规默认给（见引擎里的注释），"
            "不能声称是从实测失败率推出来的。"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="链路临时故障（502 等）自动重试的验证")
    parser.add_argument("--live", type=int, default=0, help="对真实中继采样 N 个请求（消耗配额）")
    parser.add_argument("--out", default="output/verify_http_retry.txt")
    args = parser.parse_args()

    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("链路临时故障自动重试 —— 本地真实 HTTP 验证")
    lines.append("=" * 96)
    lines.append("服务端是本机起的 http.server，真的按状态码回答；客户端是真的 urllib。")
    lines.append("只验证传输层，所以 echo_downgrade=False，排除回抄救援的干扰。")

    failures = run_local(lines)

    lines.append("\n" + "=" * 96)
    lines.append(f"本地场景：{'全部通过' if failures == 0 else f'{failures} 项不符合预期'}")
    lines.append("=" * 96)

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
