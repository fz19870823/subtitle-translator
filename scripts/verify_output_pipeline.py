"""验证「边翻译边落盘」在**真实链路**上确实成立。

TranslateWorker 干的事就是把 :class:`OutputSink` 挂在进度回调上；这里用**同一组合**
（引擎 + progress 回调 + OutputSink）复现，一边翻一边读那个输出文件，看它是不是
在翻译**进行中**就出现了、并随着进度长大。

为什么非要单独验一次：单元测试里的假引擎是毫秒级跑完的，「边翻边写」在那里根本不
成立 —— 等你去看文件时它早写完了，断言恒真。只有真实中继那种一条几秒、整份几分钟
的链路，才分得清「边翻边写」和「翻完再写」。

用法：
  python scripts/verify_output_pipeline.py
  python scripts/verify_output_pipeline.py --source 某字幕.srt --items 20
  python scripts/verify_output_pipeline.py --interval 1.0 --batch 5

退出码：0 = 全部符合预期；1 = 有断言不成立。
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import OUTPUT_DIR, load_config  # noqa: E402
from app.core.subtitle_io import (  # noqa: E402
    UNTRANSLATED_MARK,
    Cue,
    OutputSink,
    SubtitleFormatError,
    parse_file,
    parse_srt,
)
from app.core.translator import create_engine_for  # noqa: E402

#: 内置素材：日文短句。长度适中 —— 太短一批就跑完了（看不出增量落盘），
#: 太长又会让模型自由发挥、把结论搅浑。
SAMPLE_LINES = [
    "おはようございます",
    "今日はいい天気ですね",
    "ちょっと待ってください",
    "ありがとうございました",
    "そうですね、たしかに",
    "もう一度お願いします",
    "ここで少し休みましょう",
    "それは知りませんでした",
    "電車が遅れているらしい",
    "あとで連絡します",
    "無理しないでくださいね",
    "また明日会いましょう",
]


def build_cues(lines: list[str], step: float = 2.0) -> list[Cue]:
    return [
        Cue(
            index=index,
            start=(index - 1) * step,
            end=(index - 1) * step + step * 0.9,
            text=text,
        )
        for index, text in enumerate(lines, start=1)
    ]


def probe(text: str, sources: list[str]) -> tuple[int, int]:
    """数一数这份**半成品**里已经翻好了几条、标了几条未翻译。

    为什么不能数 ``-->`` 的个数：``OutputSink`` 每次写盘都是**整份重写**
    （``write_file`` 把当前所有条目一起渲染出来），所以文件从第一笔起就是满条数的
    —— 条数恒定，压根看不出增量。上一版脚本就是栽在这里：把恒等于素材条数的
    ``-->`` 计数当成了进度。

    真正随时间长大的是「已经有译文」的条数：还没轮到的条目回落到原文，翻好了的
    才是中文。把每条正文与素材逐条比对，就得到「翻到第几条了」。
    """
    try:
        parsed = parse_srt(text)
    except SubtitleFormatError:
        return (0, 0)  # 写了一半 / 空文件：当它还没翻好
    translated = 0
    for cue, source in zip(parsed, sources):
        body = cue.text.strip()
        if not body or body == UNTRANSLATED_MARK:
            continue
        if body != source.strip():
            translated += 1
    return (translated, text.count(UNTRANSLATED_MARK))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证边翻译边落盘")
    parser.add_argument(
        "--source",
        default="",
        help="从这份字幕里取前 --items 条作为素材（默认用内置的日文短句）",
    )
    parser.add_argument("--items", type=int, default=12, help="翻译多少条")
    parser.add_argument("--interval", type=float, default=2.0, help="落盘节流间隔（秒）")
    parser.add_argument("--batch", type=int, default=0, help="上下文窗口（0 = 用配置值）")
    parser.add_argument("--output", default="", help="输出文件路径")
    parser.add_argument("--timeout", type=float, default=600.0, help="整体超时（秒）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    tcfg = cfg.translation

    engine_name = tcfg.engine or "echo"
    batch = args.batch if args.batch > 0 else (tcfg.batch_size or 20)
    print(f"引擎 {engine_name}　模型 {tcfg.model or '(无)'}　端点 {tcfg.base_url or '(无)'}")
    if engine_name != "openai":
        print("提示：配置里的引擎不是 openai，这一跑不会联网，结论对真实链路无效。")

    if args.source:
        pool = parse_file(args.source)
        lines = [cue.text for cue in pool]
        if not lines:
            print(f"[FAIL] {args.source} 里没有可翻译的条目")
            return 1
    else:
        lines = SAMPLE_LINES
    lines = (lines * (args.items // len(lines) + 1))[: args.items]

    cues = build_cues(lines)
    out = Path(args.output) if args.output else Path(OUTPUT_DIR) / "verify_output_pipeline.srt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        os.remove(out)  # 上一轮的遗留会让「文件何时出现」测不准
    sink = OutputSink(out, interval=args.interval)

    engine = create_engine_for(engine_name, cfg)
    state: dict = {"done": 0, "total": len(cues), "error": None}
    finished = threading.Event()

    def on_progress(done: int, total: int) -> None:
        """与 TranslateWorker._emit_progress 同一件事：顺手把成品刷到磁盘。"""
        state["done"] = done
        sink.maybe(cues)

    def run() -> None:
        try:
            engine.translate_cues(
                cues,
                source_lang="ja",
                target_lang="zh-CN",
                batch_size=batch,
                progress=on_progress,
            )
        except Exception as exc:  # noqa: BLE001 - 报告里要说清是什么炸了
            state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            sink.flush(cues)  # 收尾强制写一次（与 worker 成功路径一致）
            finished.set()

    print(f"素材 {len(cues)} 条　上下文窗口 {batch} 条/次　节流 {args.interval}s")
    print(f"落盘目标 {out}")
    print("-" * 68)

    started = time.monotonic()
    threading.Thread(target=run, daemon=True, name="verify-translate").start()

    snapshots: list[tuple[float, int, int]] = []
    while not finished.is_set():
        elapsed = time.monotonic() - started
        if elapsed > args.timeout:
            print(f"[FAIL] {args.timeout:.0f}s 还没跑完")
            return 1
        if out.exists():
            translated, marked = probe(out.read_text(encoding="utf-8"), lines)
            snapshots.append((elapsed, translated, marked))
        time.sleep(0.3)

    elapsed = time.monotonic() - started
    finished.wait(5)

    if state["error"]:
        print(f"[FAIL] 翻译出错：{state['error']}")
        return 1

    # ---------------------------------------------------------------- 报告
    distinct = sorted({done for _, done, _ in snapshots})
    print()
    print("翻译过程中的落盘时间线（每 0.3s 采一次，只列「已翻好条数」变化的那些）：")
    last = None
    for moment, done, marked in snapshots:
        if done == last:
            continue
        last = done
        print(f"  {moment:6.2f}s　已翻好 {done:3d}/{len(cues)} 条　未翻译标记 {marked} 条")
    if not snapshots:
        print("  （整个过程里文件都没出现过）")

    content = out.read_text(encoding="utf-8") if out.exists() else ""
    final_count = content.count("-->")
    marked = content.count(UNTRANSLATED_MARK)
    # 空条目也算进 cues（时间轴照样写），所以成品条数应当与素材条数相等
    print()
    print(f"总耗时 {elapsed:.2f}s　成品 {final_count}/{len(cues)} 条　未翻译标记 {marked} 条")

    print("-" * 68)
    ok = True

    if snapshots:
        first_at = snapshots[0][0]
        print(
            f"[PASS] 输出文件在翻译结束前就出现了"
            f"　— 首次可见于 {first_at:.2f}s，总耗时 {elapsed:.2f}s"
        )
        halfway = [s for s in snapshots if 0 < s[1] < len(cues)]
        if halfway:
            print(
                f"[PASS] 边翻边长大　— 抓到 {len(halfway)} 次半成品快照，"
                f"最多时已有 {max(s[1] for s in halfway)} 条译文"
            )
        if len(distinct) >= 2:
            print(
                f"[PASS] 内容是随进度长出来的　— 观察到 {len(distinct)} 个不同的已翻条数"
                f"：{distinct}"
            )
        else:
            ok = False
            print(
                "[FAIL] 整段只见到一个已翻条数状态 —— 文件要么一次写完、要么根本没增量写"
            )
    else:
        ok = False
        print("[FAIL] 翻译跑完了，输出文件却始终没出现过")

    if final_count == len(cues):
        print(f"[PASS] 成品完整　— {final_count} 条，与素材一致")
    else:
        ok = False
        print(f"[FAIL] 成品只有 {final_count} 条，素材是 {len(cues)} 条")

    notes = engine.quality_notes(include_untranslated=False)
    if marked:
        print(f"[--] 未翻译标记 {marked} 条 —— 重试三轮后仍未救回，成品里如实标出")
    else:
        print("[--] 这一轮没撞上回抄，成品里没有未翻译标记（属正常）")
    if notes:
        print("     引擎插曲：" + "；".join(notes))

    print()
    print("结论：" + ("全部符合预期" if ok else "有不成立项，见上面的 [FAIL]"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
