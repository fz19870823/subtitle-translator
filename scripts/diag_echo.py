"""诊断脚本：量化「原样回抄」的真实形态。

直接调用引擎的 ``_request_batch``（绕开内部重试），观测**第一次请求的原始输出**，
分别统计：

- 批次级回抄：整批是否被判定为回抄（现有 ``looks_like_verbatim_echo``）
- 条目级回抄：原始输出里与原文完全相同的条目数与占比
- 守卫是否开启：``should_check_echo(source, target)``

用法::

    python scripts/diag_echo.py [--model MODEL] [--rounds N] [--batch N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402
from app.core.engines.openai_compat import OpenAICompatTranslator  # noqa: E402
from app.core.translator import (  # noqa: E402
    is_translatable,
    looks_like_verbatim_echo,
    should_check_echo,
)

# 典型日文字幕：假名多、汉字少，翻译成中文后与原文必然不同 —— 相同即为回抄。
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

# 汉字占比高的语料：这些条目译成中文后可能**合法地**与原文相同（同形汉字词），
# 用来观察条目级判定会不会误报。
JA_KANJI_CORPUS = [
    "東京駅で待っている。",
    "学校に行く。",
    "今日は休みだ。",
    "明日は晴れる。",
    "電話があった。",
    "時間がない。",
    "仕事が終わった。",
    "人数を確認する。",
    "世界は広い。",
    "自分で決める。",
]


def measure(
    engine: OpenAICompatTranslator,
    corpus: list[str],
    source_lang: str,
    target_lang: str,
    *,
    rounds: int,
    batch_size: int,
    on_row=None,
) -> dict:
    """跑若干轮，统计原始输出（未经重试）的回抄形态。"""
    guard_on = should_check_echo(source_lang, target_lang)
    rows: list[dict] = []
    for r in range(rounds):
        window = corpus[:batch_size]
        texts = [engine._encode(t) for t in window]
        started = time.time()
        try:
            raw = engine._request_batch(texts, source_lang, target_lang)
        except Exception as exc:  # noqa: BLE001 - 诊断脚本，任何异常都要记下来
            row = {"round": r + 1, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            if on_row:
                on_row(row)
            continue
        elapsed = time.time() - started

        pairs = [(s, o) for s, o in zip(texts, raw) if is_translatable(s)]
        same = [s for s, o in pairs if s.strip() == o.strip()]
        batch_echo = looks_like_verbatim_echo(texts, raw, source_lang, target_lang)
        row = {
            "round": r + 1,
            "elapsed": round(elapsed, 2),
            "returned": len(raw),
            "items": len(pairs),
            "same": len(same),
            "ratio": round(len(same) / len(pairs), 3) if pairs else 0.0,
            "batch_echo": batch_echo,
            "same_samples": [s[:24] for s in same[:3]],
        }
        rows.append(row)
        if on_row:
            on_row(row)
    return {"source": source_lang, "target": target_lang, "guard": guard_on, "rows": rows}


def summarize(block: dict) -> dict:
    rows = [r for r in block["rows"] if "error" not in r]
    errors = [r for r in block["rows"] if "error" in r]
    total_items = sum(r["items"] for r in rows)
    total_same = sum(r["same"] for r in rows)
    return {
        "direction": f"{block['source']} -> {block['target']}",
        "guard": block["guard"],
        "rounds": len(rows),
        "errors": len(errors),
        "batch_echo_rounds": sum(1 for r in rows if r["batch_echo"]),
        "item_level_same": total_same,
        "item_level_total": total_items,
        "item_level_ratio": round(total_same / total_items, 4) if total_items else 0.0,
        "partial_echo_rounds": sum(1 for r in rows if 0 < r["same"] and not r["batch_echo"]),
    }


def format_row(r: dict, guard: bool) -> str:
    if "error" in r:
        return f"  R{r['round']:<2} ERROR  {r['error'][:90]}"
    return (
        f"  R{r['round']:<2} {r['elapsed']:>5.2f}s  "
        f"返回 {r['returned']:>2}  可比 {r['items']:>2}  "
        f"与原文相同 {r['same']:>2} ({r['ratio']*100:>5.1f}%)  "
        f"批次判定 {'回抄' if r['batch_echo'] else '正常'}  "
        f"守卫 {'开' if guard else '关'}  {r['same_samples']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="覆盖配置里的模型")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--sources", default="ja,auto", help="逗号分隔的源语言取值")
    parser.add_argument("--corpus", default="kana", choices=("kana", "kanji", "both"))
    parser.add_argument("--out", default="output/diag_echo.txt")
    args = parser.parse_args()

    cfg = load_config()
    tcfg = cfg.translation
    if args.model:
        tcfg.model = args.model
    engine = OpenAICompatTranslator.from_config(tcfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    summaries: list[dict] = []

    def flush() -> None:
        # 每轮都落盘：单次运行可能被外部超时掐断，不能让已跑出来的数据跟着丢。
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines.append("=" * 100)
    lines.append(f"model = {engine.model}   batch = {args.batch}   rounds = {args.rounds}")
    lines.append(f"base_url = {engine.base_url}")
    lines.append("=" * 100)
    flush()

    corpora = []
    if args.corpus in ("kana", "both"):
        corpora.append(("常用语料（假名多）", JA_CORPUS))
    if args.corpus in ("kanji", "both"):
        corpora.append(("汉字语料（同形词多）", JA_KANJI_CORPUS))

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for label, corpus in corpora:
        for source in sources:
            block = {"source": source, "target": "zh-CN", "rows": []}

            def on_row(row: dict, block=block) -> None:
                block["rows"].append(row)
                lines.append(format_row(row, block["guard"]))
                flush()

            block["guard"] = should_check_echo(source, "zh-CN")
            lines.append(f"\n### {label}   [{source} -> zh-CN]  守卫生效 = {block['guard']}")
            flush()
            measure(
                engine,
                corpus,
                source,
                "zh-CN",
                rounds=args.rounds,
                batch_size=args.batch,
                on_row=on_row,
            )
            summaries.append(summarize(block))

    lines.append("\n" + "=" * 100)
    lines.append("汇总")
    lines.append("=" * 100)
    lines.append(
        f"{'方向':<14}{'守卫':<6}{'轮数':<6}{'报错':<6}"
        f"{'整批回抄轮':<11}{'条目级相同':<12}{'条目级占比':<11}{'局部污染轮':<10}"
    )
    lines.append("-" * 100)
    for s in summaries:
        lines.append(
            f"{s['direction']:<14}{'开' if s['guard'] else '关':<6}{s['rounds']:<6}{s['errors']:<6}"
            f"{s['batch_echo_rounds']:<11}{s['item_level_same']:<12}"
            f"{s['item_level_ratio']*100:>8.1f}%   {s['partial_echo_rounds']:<10}"
        )
    flush()
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
