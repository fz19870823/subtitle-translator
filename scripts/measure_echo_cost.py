"""度量回抄防护的**配额开销**与**最终残留**。

免费档（`grok-chat-fast`）的失败是请求级的随机事件，防护靠「多发几次」换质量，
所以真正要回答两个问题：

1. 每批平均要多发多少请求？（配额够不够用）
2. 用尽所有手段后，最后还剩几条没翻译？（成品能不能直接用）

做法：包一层 ``_post`` 数请求次数，每轮独立重置计数，跑完整 ``translate_batch``，
再用 ``locate_untranslated`` 数最终交付里还有多少条是原文。

用法::

    python scripts/measure_echo_cost.py --rounds 10
    python scripts/measure_echo_cost.py --rounds 10 --echo-retries 2 --item-retries 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config  # noqa: E402
from app.core.engines.openai_compat import OpenAICompatTranslator  # noqa: E402
from app.core.translator import (  # noqa: E402
    TranslationRequest,
    locate_untranslated,
)

JA_CORPUS = [
    "何度も言ったはずだ。",
    "そんなの無理だよ。",
    "お前、何を考えてるんだ？",
    "ごめん、遅れた。",
    "電車が止まってて。",
    "静かにして。",
    "ここから先は立入禁止だ。",
    "分かった。任せるよ。",
    "本当にいいのか？",
    "後悔するぞ。",
    "もう決めたことだ。",
    "あ、そうだ。",
    "田中さんに伝えといて。",
    "彼女、明日来るの？",
    "たぶん来ないと思う。",
    "ふざけるな！",
    "行こう。",
    "待って、鍵かけた？",
    "おわり。",
    "それでいいのかい？",
]


class _Counter:
    """把引擎的每一次底层 HTTP 请求记下来。"""

    def __init__(self) -> None:
        self.n = 0

    def reset(self) -> None:
        self.n = 0


def patch_post(engine: OpenAICompatTranslator, counter: _Counter) -> None:
    original = engine._post

    def counting_post(path, body):
        counter.n += 1
        return original(path, body)

    engine._post = counting_post  # type: ignore[method-assign]


def main() -> int:
    parser = argparse.ArgumentParser(description="回抄防护的配额开销与残留度量")
    parser.add_argument("--model", default=None)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--echo-retries", type=int, default=None, help="默认用引擎当前值")
    parser.add_argument("--item-retries", type=int, default=None, help="默认用引擎当前值")
    parser.add_argument("--source", default="auto")
    parser.add_argument("--out", default="output/echo_cost.txt")
    args = parser.parse_args()

    cfg = load_config().translation
    if args.model:
        cfg.model = args.model
    engine = OpenAICompatTranslator.from_config(cfg)
    if args.echo_retries is not None:
        engine.echo_retries = max(0, args.echo_retries)
    if args.item_retries is not None:
        engine.echo_item_retries = max(0, args.item_retries)

    counter = _Counter()
    patch_post(engine, counter)

    target_lang = "zh-CN"
    texts = [engine._encode(t) for t in JA_CORPUS[: args.batch]]
    reqs = [
        TranslationRequest(text=t, source_lang=args.source, target_lang=target_lang)
        for t in texts
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def flush() -> None:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines.append("=" * 112)
    lines.append(
        f"model = {engine.model}   源语言 = {args.source}   目标 = {target_lang}   "
        f"批次条目 = {len(texts)}"
    )
    lines.append(
        f"echo_retries = {engine.echo_retries}（整批重发上限）   "
        f"echo_item_retries = {engine.echo_item_retries}（单条重试上限）"
    )
    lines.append("=" * 112)
    lines.append(
        f"{'轮次':<6}{'请求数':<8}{'耗时':<9}{'整批重发':<10}{'逐条条目':<10}"
        f"{'逐条请求':<10}{'救回':<7}{'最终未翻译':<12}{'结果'}"
    )
    lines.append("-" * 112)
    flush()

    total_reqs = total_before = 0
    rows: list[dict] = []
    for r in range(args.rounds):
        counter.reset()
        before_retry = engine.echo_retry_count
        before_item = engine.echo_item_count
        before_attempts = engine.echo_item_attempts
        before_repaired = engine.echo_repaired_count
        before_untrans = engine.untranslated_count

        started = time.time()
        try:
            delivered = engine.translate_batch(reqs)
        except Exception as exc:  # noqa: BLE001 - 度量脚本，异常也要落盘
            elapsed = time.time() - started
            lines.append(f"R{r + 1:<5}{counter.n:<8}{elapsed:>5.2f}s  异常：{type(exc).__name__}: {str(exc)[:60]}")
            flush()
            continue
        elapsed = time.time() - started

        residual = len(locate_untranslated(texts, delivered, target_lang))
        used = counter.n
        total_reqs += used
        total_before += residual
        rows.append({"reqs": used, "residual": residual, "elapsed": elapsed})

        lines.append(
            f"R{r + 1:<5}{used:<8}{elapsed:>5.2f}s  "
            f"{engine.echo_retry_count - before_retry:<10}"
            f"{engine.echo_item_count - before_item:<10}"
            f"{engine.echo_item_attempts - before_attempts:<10}"
            f"{engine.echo_repaired_count - before_repaired:<7}"
            f"{residual:>3}/{len(texts):<8}"
            f"{'干净' if residual == 0 else '有残留'}"
        )
        flush()

    if not rows:
        lines.append("没有任何一轮成功完成。")
        flush()
        print(f"written: {out}")
        return 1

    n = len(rows)
    avg_reqs = total_reqs / n
    lines.append("-" * 112)
    lines.append(
        f"平均每批发出 {avg_reqs:.2f} 个请求（理想情况 1.00，"
        f"开销 = {(avg_reqs - 1) * 100:.0f}%）"
    )
    lines.append(
        f"平均耗时 {sum(x['elapsed'] for x in rows) / n:.2f}s/批，"
        f"单条字幕折算 {sum(x['elapsed'] for x in rows) / (n * len(texts)):.3f}s"
    )
    lines.append(
        f"最终未翻译 {total_before} 条 / 共 {n * len(texts)} 条"
        f"（{total_before / (n * len(texts)) * 100:.2f}%）；"
        f"折算到 1000 条约 {total_before / (n * len(texts)) * 1000:.1f} 条"
    )
    lines.append(
        f"累计：整批重发 {engine.echo_retry_count} 次，逐条条目 {engine.echo_item_count} 条 / "
        f"实际请求 {engine.echo_item_attempts} 次，救回 {engine.echo_repaired_count} 条"
    )
    flush()
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
