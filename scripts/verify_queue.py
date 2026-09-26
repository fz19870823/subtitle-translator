"""验证：一次排入多个文件，能不能依次翻完、各自落盘、失败不拖住后面。

全程离线 —— 用项目自带的 ``echo`` 引擎（给每行加个语言前缀），不需要密钥、
不发网络请求，所以可以随便重复跑。

    python scripts/verify_queue.py
    python scripts/verify_queue.py --files 4 --batch 5 --fail-at 2 --keep

按生产里的那套编排走一遍（core 层的队列状态机 + 逐份翻译 + 自动导出）：

1. 造几份**内容互不相同**的字幕，外加一个坏文件和一个同名文件 ——
   坏文件该在入队时就被剔掉，同名的两份该拿到两个不同的译文文件；
2. 依次翻译，中间故意让某一份失败，看队列是不是跳过它继续往下走；
3. 检查每一份的译文是否真的落到输出目录，以及失败的那份有没有留下半成品。

不需要 PySide6：这里验证的是编排语义（谁先谁后、失败怎么办、产物在哪），
界面那一层的行为由 ``tests/test_ui_smoke.py`` 覆盖。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import queue as queue_store  # noqa: E402
from app.core import subtitle_io  # noqa: E402
from app.core.subtitle_io import SubtitleFormatError  # noqa: E402
from app.core.translator import (  # noqa: E402
    EchoTranslator,
    TranslationError,
    locate_untranslated,
)

#: 语料必须**逐份不同**：重复的文本会让「这一份到底翻了没有」的比对恒为真。
LINES = [
    "何度も言ったはずだ。",
    "学校の前に集まってください。",
    "それは私の責任です。",
    "昨日の夜、雨が降りました。",
    "この問題は難しいですね。",
    "彼女は三年生です。",
    "時間があるなら一緒に行きましょう。",
    "静かにしてください。",
    "約束は守らなければならない。",
    "駅まで歩いて十分かかります。",
]


def build_srt(tag: str, lines: int) -> str:
    blocks = []
    for number in range(1, lines + 1):
        text = f"{LINES[(number - 1) % len(LINES)]}<{tag}-{number}>"
        blocks.append(
            f"{number}\n"
            f"{subtitle_io.format_timestamp(number * 2)} --> "
            f"{subtitle_io.format_timestamp(number * 2 + 1.5)}\n{text}"
        )
    return "\n\n".join(blocks) + "\n"


class FlakyEcho(EchoTranslator):
    """碰到「毒文本」就整条链路失联，用来验证「失败跳过、继续下一个」。

    这正是 502 重试耗尽的样子：这一份翻不动了，但后面的文件不该跟着遭殃。
    """

    name = "echo"

    def __init__(self, poison: set) -> None:
        self._poison = set(poison)
        self.batches = 0

    def translate_batch(self, requests):
        if any(request.text in self._poison for request in requests):
            raise TranslationError("模拟链路中断：HTTP 502（已自动重试 3 次）")
        self.batches += 1
        return super().translate_batch(requests)


def lay_out_files(workdir: Path, files: int, lines: int) -> tuple:
    """造出一批字幕：一个坏文件、一个与第一份同名的文件（在子目录里）。

    返回 ``(坏文件路径, 队列入参)``。
    """
    source = workdir / "src"
    source.mkdir(parents=True, exist_ok=True)
    paths = []
    for number in range(1, files + 1):
        path = source / f"{number:02d}.srt"
        path.write_text(build_srt(f"f{number}", lines), encoding="utf-8")
        paths.append(path)

    # 与第 1 份同名但来自另一个目录 —— 两者的译文不能落到同一个文件上。
    twin = source / "season2" / "01.srt"
    twin.parent.mkdir(parents=True, exist_ok=True)
    twin.write_text(build_srt("twin", lines), encoding="utf-8")
    paths.append(twin)

    broken = source / "broken.srt"
    broken.write_text("这不是字幕", encoding="utf-8")
    return broken, [*paths, broken]


def main() -> int:
    parser = argparse.ArgumentParser(description="验证批量队列的编排行为")
    parser.add_argument("--files", type=int, default=3, help="字幕份数（不含同名那份）")
    parser.add_argument("--lines", type=int, default=6, help="每份多少条")
    parser.add_argument("--batch", type=int, default=5, help="上下文窗口（条/次）")
    parser.add_argument(
        "--fail-at", type=int, default=2, help="让第几份失败（从 1 数；0 表示都不失败）"
    )
    parser.add_argument("--keep", action="store_true", help="保留临时工作目录")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="verify-queue-"))
    broken, candidates = lay_out_files(workdir, args.files, args.lines)

    print(f"工作目录  {workdir}")
    print()

    # ------------------------------------------------------------ 入队
    batch = queue_store.TranslationQueue(workdir / "output")
    added, notes = batch.add(candidates)
    # 试读一遍再定稿 —— 与界面里的 ``_enqueue`` 是同一套做法：读不了的文件
    # 必须在**入队时**剔掉，而不是等轮到它才发现。
    for item in list(added):
        try:
            item.cue_count = len(subtitle_io.parse_file(item.path))
        except (SubtitleFormatError, OSError) as exc:
            batch.remove(item)
            notes.append(f"{item.path.name}：{exc}")

    print("入队")
    for item in batch:
        plain = item.path.stem + ".translated" + item.path.suffix
        tail = "" if item.output_path.name == plain else "（与前面的同名，另起一个译文文件）"
        print(f"  {item.name:<10} {item.cue_count} 条 → {item.output_path.name}{tail}")
    for note in notes:
        print(f"  跳过：{note}")
    print()

    assert any(broken.name in note for note in notes), "坏文件本该在入队时就被剔除"

    # ------------------------------------------------------------ 依次翻译
    poison = set()
    if args.fail_at:
        victim = batch[args.fail_at - 1]
        poison = {
            cue.text for cue in subtitle_io.parse_file(victim.path)
        }

    print("依次翻译")
    total = len(batch)
    while True:
        index = batch.next_pending()
        if index is None:
            break
        item = batch[index]
        batch.mark_running(index)

        cues = subtitle_io.parse_file(item.path)
        engine = FlakyEcho(poison)
        try:
            engine.translate_cues(
                cues,
                source_lang="ja",
                target_lang="zh-CN",
                batch_size=args.batch,
            )
        except TranslationError as exc:
            batch.fail(index, str(exc))
            print(f"  [{index + 1}/{total}] {item.name:<10} 失败：{exc}")
            print("            → 跳过，继续下一个")
            continue

        item.untranslated = len(
            locate_untranslated(
                [cue.text for cue in cues],
                [cue.translation or "" for cue in cues],
                "zh-CN",
            )
        )
        try:
            written = subtitle_io.write_file(item.output_path, cues)
        except OSError as exc:
            batch.fail(index, f"导出失败：{exc}")
            print(f"  [{index + 1}/{total}] {item.name:<10} 导出失败：{exc}")
            continue

        batch.mark_done(index)
        print(
            f"  [{index + 1}/{total}] {item.name:<10} 完成（{len(cues)} 条）"
            f" → {written.relative_to(workdir)}"
        )
    print()

    # ------------------------------------------------------------ 汇总
    counts = batch.summary()
    print("汇总")
    print(f"  入队        {counts['total']} 份（另有 1 个坏文件被剔除）")
    print(f"  完成        {counts['done']} 份")
    print(f"  失败        {counts['failed']} 份")
    print(f"  未翻译      {counts['untranslated']} 条")
    print()

    # ------------------------------------------------------------ 校验
    print("校验")
    done_items = [item for item in batch if item.status == queue_store.DONE]
    failed_items = [item for item in batch if item.status == queue_store.FAILED]

    missing = [item.name for item in done_items if not item.output_path.exists()]
    not_translated = [
        item.name
        for item in done_items
        if "[zh-CN]" not in item.output_path.read_text(encoding="utf-8")
    ]
    leaked = [
        item.name
        for item in failed_items
        if item.output_path.exists()
        and "[zh-CN]" in item.output_path.read_text(encoding="utf-8")
    ]
    outputs = [item.output_path.name for item in batch]
    duplicated = len(outputs) != len(set(outputs))
    expected_done = counts["total"] - (1 if args.fail_at else 0)

    print(f"  坏文件被剔除            {'否（有问题）' if any(broken.name in i.name for i in batch) else '是'}")
    print(f"  完成份数                {counts['done']}（期望 {expected_done}）")
    print(f"  产物缺失                {'有：' + '、'.join(missing) if missing else '无'}")
    print(f"  产物里没有译文          {'有：' + '、'.join(not_translated) if not_translated else '无'}")
    print(f"  失败的那份留下了半成品  {'有：' + '、'.join(leaked) if leaked else '无'}")
    print(f"  译文文件名互相撞车      {'是（有问题）' if duplicated else '否'}")
    for item in batch:
        if item.status == queue_store.DONE:
            print(f"    {item.path.name:<10} → {item.output_path.name}")
    print()

    ok = (
        counts["done"] == expected_done
        and not missing
        and not not_translated
        and not leaked
        and not duplicated
    )
    print("结论：" + ("批量队列按预期工作" if ok else "有不符合预期的地方，见上面各栏"))

    if args.keep:
        print(f"（工作目录已保留：{workdir}）")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
