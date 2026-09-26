"""验证修复：源语言写 auto 时，整批回抄能不能被拦住。

对照两组：

- ``--echo-retries 2``（产品默认）：整批重发 → 仍不行的逐条重译；
- ``--echo-retries 0``：完全靠逐条重译救援，用来单独评估这一层的价值。

每轮都记录「修复前的样子」——也就是**只发一次请求**拿到的原始输出里有多少条
与原文一字不差（那正是修复前会被静默交付的内容），以及修复后实际交付了几条。

用法::

    python scripts/verify_echo_fix.py --rounds 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402
from app.core.engines.openai_compat import OpenAICompatTranslator  # noqa: E402
from app.core.translator import (  # noqa: E402
    TranslationRequest,
    guess_text_family,
    locate_untranslated,
    should_check_echo,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--echo-retries", type=int, default=2)
    parser.add_argument("--out", default="output/verify_echo_fix.txt")
    args = parser.parse_args()

    cfg = load_config().translation
    if args.model:
        cfg.model = args.model
    engine = OpenAICompatTranslator.from_config(cfg)
    if args.echo_retries != engine.echo_retries:
        engine.echo_retries = args.echo_retries

    source_lang, target_lang = "auto", "zh-CN"
    texts = [engine._encode(t) for t in JA_CORPUS]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def flush() -> None:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines.append("=" * 104)
    lines.append(
        f"model = {engine.model}   源语言 = {source_lang}（界面默认值）   "
        f"目标 = {target_lang}   echo_retries = {engine.echo_retries}   条目 = {len(texts)}"
    )
    lines.append("-" * 104)
    lines.append(f"推断出的源语言族 = {guess_text_family(texts)}   守卫生效 = {should_check_echo(source_lang, target_lang, samples=texts)}")
    lines.append("=" * 104)
    lines.append(
        f"{'轮次':<6}{'耗时':<8}{'修复前未翻译':<14}{'整批重发':<10}{'逐条重译':<10}"
        f"{'逐条救回':<10}{'交付未翻译':<12}{'结果'}"
    )
    lines.append("-" * 104)
    flush()

    before_total = after_total = 0
    for r in range(args.rounds):
        started = time.time()
        # 修复前的样子：只发一次请求（不重发、不降级）。它会走 _request_batch 的
        # 内建逐条回退，但不做任何回抄处置 —— 等价于旧版本 auto 模式的交付内容。
        raw = engine._request_batch(texts, source_lang, target_lang)
        before = len(locate_untranslated(texts, raw, target_lang))

        retries_before = engine.echo_retry_count
        items_before = engine.echo_item_count
        repaired_before = engine.echo_repaired_count
        unstat_before = engine.untranslated_count

        reqs = [
            TranslationRequest(text=t, source_lang=source_lang, target_lang=target_lang)
            for t in texts
        ]
        delivered = engine.translate_batch(reqs)
        after = len(locate_untranslated(texts, delivered, target_lang))

        elapsed = time.time() - started
        before_total += before
        after_total += after
        verdict = "拦住" if before and not after else ("正常" if not before else "仍有残留")
        lines.append(
            f"R{r+1:<5}{elapsed:>5.2f}s  {before:>3}/{len(texts):<10}"
            f"{engine.echo_retry_count - retries_before:<10}"
            f"{engine.echo_item_count - items_before:<10}"
            f"{engine.echo_repaired_count - repaired_before:<10}"
            f"{after:>3}/{len(texts):<8}{verdict}"
        )
        flush()

    lines.append("-" * 104)
    lines.append(
        f"合计：修复前会交付 {before_total} 条未翻译（{before_total/(args.rounds*len(texts))*100:.1f}%），"
        f"修复后为 {after_total} 条"
    )
    lines.append(f"累计：整批重发 {engine.echo_retry_count} 次，逐条重译 {engine.echo_item_count} 条，"
                 f"逐条救回 {engine.echo_repaired_count} 条，最终未翻译 {engine.untranslated_count} 条")
    flush()
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
