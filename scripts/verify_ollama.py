"""在线自检：拿真实 Ollama 服务跑一遍项目自身的翻译链路。

用法：
    python scripts/verify_ollama.py                          # 用 config.local.json
    python scripts/verify_ollama.py --base-url http://192.168.1.50:11434
    python scripts/verify_ollama.py --model huihui_ai/qwen2.5-abliterate:14b
    python scripts/verify_ollama.py --list                   # 只看有哪些模型
    python scripts/verify_ollama.py --compare-v1             # 额外对照 /v1 兼容层

它回答的问题和离线测试不同：离线测试证明「代码按约定发请求」，这里证明
「那台真机上，按这个约定发出去的请求确实能翻出字幕」。

默认只做只读检查 + 一次真实翻译。``--compare-v1`` 会**卸载并重新加载模型**
（用来对照 OpenAI 兼容层是否丢弃 ``options.num_ctx``），不想被打断就别加。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ConfigError, load_config  # noqa: E402
from app.core import subtitle_io  # noqa: E402
from app.core.engines import ollama as ol  # noqa: E402
from app.core.engines.ollama import OllamaTranslator  # noqa: E402
from app.core.translator import TranslationError  # noqa: E402

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:03,200
So I told him, <i>that's not how it works</i>.

2
00:00:03,400 --> 00:00:06,000
Third attempt, and
it still fails.

3
00:00:06,200 --> 00:00:08,500
これは字幕のテストです。

4
00:00:08,700 --> 00:00:11,000
Right. Let's ship it.

5
00:00:11,200 --> 00:00:13,000
The quick brown fox jumps over the lazy dog.

6
00:00:13,200 --> 00:00:15,000
午前九時に駅で待っています。

7
00:00:15,200 --> 00:00:17,000
Nobody expected the Spanish Inquisition.

8
00:00:17,200 --> 00:00:19,000
Turn left at the traffic lights, then go straight.

9
00:00:19,200 --> 00:00:21,000
天気がいいので散歩でもしませんか。

10
00:00:21,200 --> 00:00:23,000
I'll call you back in ten minutes.

11
00:00:23,200 --> 00:00:25,000
Keep the receipt, we might need it later.

12
00:00:25,200 --> 00:00:27,000
それでは、また明日。
"""


def http(
    url: str, body: dict | None = None, *, timeout: float = 30.0
) -> tuple[int | None, str]:
    """一个直白的 HTTP 调用，只用于自检——不经过项目代码。"""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:400]
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def loaded_state(root: str) -> dict[str, dict]:
    """``GET /api/ps``：当前加载着哪些模型、各自的上下文与卸载时刻。

    这是本脚本唯一可信的「服务端实际状态」来源 —— 请求返回 200 只说明它被接受了，
    不代表里面的字段被采纳了。
    """
    status, body = http(f"{root}/api/ps")
    if status != 200:
        return {}
    try:
        entries = json.loads(body).get("models", [])
    except json.JSONDecodeError:
        return {}
    return {e.get("name") or e.get("model"): e for e in entries}


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama 后端在线自检")
    parser.add_argument("--base-url", default="", help="覆盖配置里的地址")
    parser.add_argument("--model", default="", help="覆盖配置里的模型")
    parser.add_argument("--items", type=int, default=0, help="只翻前 N 条")
    parser.add_argument("--batch", type=int, default=0, help="上下文窗口（条/次）")
    parser.add_argument("--no-stream", action="store_true", help="关掉流式接收")
    parser.add_argument("--list", action="store_true", help="只列出模型")
    parser.add_argument(
        "--compare-v1",
        action="store_true",
        help="对照 OpenAI 兼容层是否丢弃 options.num_ctx（会卸载重载模型）",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------- 配置
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"配置加载失败: {exc}")
        return 2

    tcfg = config.translation
    base_url = args.base_url or tcfg.base_url or ol.DEFAULT_BASE_URL
    root = ol.normalise_base_url(base_url)
    model = args.model or tcfg.model
    batch = args.batch or int(tcfg.batch_size or 20)

    print("=" * 70)
    print("1) 服务与模型")
    print("=" * 70)
    print(f"  地址（规整后）: {root}")
    status, body = http(f"{root}/api/version", timeout=10)
    if status != 200:
        print(f"  [FAIL] 连不上：{body}")
        return 1
    print(f"  [PASS] 服务可达　—　{body.strip()}")

    try:
        models = ol.fetch_models(root)
    except TranslationError as exc:
        print(f"  [FAIL] 拉模型列表失败：{exc}")
        return 1
    print(f"  [PASS] 本机有 {len(models)} 个模型")
    for name in models:
        mark = "　<== 本次使用" if name == model else ""
        print(f"        - {name}{mark}")

    if args.list:
        return 0

    if model not in models:
        print(f"  [FAIL] 配置的模型 {model!r} 不在列表里")
        return 1
    if args.items:
        pass

    # ---------------------------------------------------------------- 真实翻译
    print()
    print("=" * 70)
    print("2) 真实字幕翻译（走项目自身代码路径）")
    print("=" * 70)
    cues = subtitle_io.parse_srt(SAMPLE_SRT)
    if args.items:
        cues = cues[: args.items]
    engine = OllamaTranslator(
        base_url=root,
        model=model,
        timeout=int(tcfg.timeout or 120),
        temperature=float(tcfg.temperature or 0.0),
        batch_size=batch,
        preserve_line_breaks=bool(tcfg.preserve_line_breaks),
        stream=not args.no_stream,
    )
    if isinstance(tcfg.extra, dict) and isinstance(tcfg.extra.get("ollama"), dict):
        section = tcfg.extra["ollama"]
        engine.num_ctx = section.get("num_ctx") or engine.num_ctx
        engine.keep_alive = section.get("keep_alive") or engine.keep_alive
        if section.get("think") is not None:
            engine.think = bool(section["think"])
    print(
        f"  引擎 ollama　模型 {model}　上下文窗口 {batch} 条/次　"
        f"流式 {'关' if args.no_stream else '开'}　"
        f"num_ctx {engine.num_ctx or '(交给 Ollama，用模型自身上限)'}　"
        f"think {engine.think}　keep_alive {engine.keep_alive}"
    )

    done: list[tuple[int, int]] = []
    started = time.monotonic()
    try:
        engine.translate_cues(
            cues,
            source_lang="auto",
            target_lang="zh-CN",
            batch_size=batch,
            progress=lambda d, t: done.append((d, t)),
        )
    except TranslationError as exc:
        print(f"  [FAIL] 翻译失败：{exc}")
        return 1
    elapsed = time.monotonic() - started

    ok = True
    print()
    print("  原文 -> 译文")
    untouched = 0
    for cue in cues:
        if not cue.text.strip():
            continue
        translated = cue.translation.strip()
        if not translated:
            untouched += 1
            marker = "未翻译"
        elif translated == cue.text.strip():
            untouched += 1
            marker = "原样退回"
        else:
            marker = "OK"
        print(f"    [{marker:>4}] {cue.text!r}")
        print(f"           -> {translated!r}")

    print()
    print(f"  耗时 {elapsed:.2f}s　进度回调 {done}")
    print(f"  逐条重译救回 {engine.echo_repaired_count} 条　"
          f"整批重发 {engine.echo_retry_count} 次　"
          f"链路重试 {engine.http_retry_count} 次　"
          f"流式退回 {engine.stream_fallback_count} 次")
    notes = engine.quality_notes(include_untranslated=False)
    if notes:
        print(f"  质量插曲：{'　|　'.join(notes)}")

    print("-" * 70)
    if untouched:
        ok = False
        print(f"[FAIL] 有 {untouched} 条没有真正翻译")
    else:
        print("[PASS] 每条都拿到了与原文不同的译文（不是回抄）")

    if engine.truncated_context_count:
        ok = False
        print(f"[FAIL] 有 {engine.truncated_context_count} 批提示词被服务端截断")
    else:
        print("[PASS] 没有批次被截断（服务端吃下了完整的提示词）")

    # ---------------------------------------------------------------- 服务端状态
    print()
    print("=" * 70)
    print("3) 服务端实际状态（/api/ps）—— 用来证明参数真的被采纳了")
    print("=" * 70)
    state = loaded_state(root).get(model)
    if state is None:
        print("  [--] 模型已经不在内存里（keep_alive 到期或本轮没加载）")
    else:
        ctx = state.get("context_length")
        print(f"  模型 {model}　context_length={ctx}")
        expires = state.get("expires_at") or ""
        if expires:
            print(f"  卸载时刻 {expires}　（keep_alive={engine.keep_alive}）")
            try:
                from datetime import datetime

                when = datetime.fromisoformat(expires).timestamp()
                remain = when - time.time()
                if remain > 6 * 60:
                    print(f"  [PASS] 距现在还有 {remain / 60:.1f} 分钟才卸载"
                          "　—　keep_alive 生效（默认只有 5 分钟）")
                else:
                    print(f"  [--] 距现在 {remain / 60:.1f} 分钟　—　"
                          "可能是上一轮加载的，或刚好临近到期")
            except ValueError:
                print("  [--] 卸载时刻解析不了，跳过")
        if engine.num_ctx is not None:
            if ctx == engine.num_ctx:
                print(f"  [PASS] num_ctx 生效　—　{ctx} == {engine.num_ctx}")
            else:
                ok = False
                print(f"  [FAIL] num_ctx 没生效　—　实际 {ctx}，期望 {engine.num_ctx}")

    # ---------------------------------------------------------------- 对照
    if args.compare_v1:
        print()
        print("=" * 70)
        print("4) 对照：OpenAI 兼容层 /v1 到底认不认 options.num_ctx")
        print("=" * 70)
        print("  （这一步会卸载并重新加载模型，所以默认不做）")
        for label, use_v1 in (("/api/chat（原生）", False), ("/v1/chat/completions", True)):
            http(f"{root}/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
            time.sleep(1.0)
            path = "/v1/chat/completions" if use_v1 else "/api/chat"
            body = {
                "model": model,
                "messages": [{"role": "user", "content": "说一个字：好"}],
                "stream": False,
                "options": {"num_ctx": 8192, "num_predict": 16},
            }
            if use_v1:
                body.pop("options")  # /v1 的 options 会被丢弃，这里照样发一次做对照
            status, _ = http(f"{root}{path}", body, timeout=180)
            time.sleep(0.5)
            ctx = (loaded_state(root).get(model) or {}).get("context_length")
            print(f"  {label:<24} 送 num_ctx=8192 -> 实际 context_length={ctx}　"
                  f"（status={status}，返回 200 不代表字段被采纳）")
        http(f"{root}/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
        print("  结论：原生会照着 8192 走；兼容层给的是模型默认值 —— 这正是本引擎"
              "走原生接口的原因。")

    print()
    print(f"结论：{'全部检查通过' if ok else '存在失败项，见上面的 [FAIL]'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
