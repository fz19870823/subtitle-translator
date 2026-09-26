"""验证：翻译中断后能不能接着翻，以及到底省下多少请求。

全程离线 —— 用项目自带的 ``echo`` 引擎（给每行加个语言前缀），不需要密钥、
不发网络请求，所以可以随便重复跑。

    python scripts/verify_resume.py
    python scripts/verify_resume.py --count 200 --batch 20 --stop-after 60

流程分两轮，中间那一下就是用户点「取消」或直接关窗：

1. 第一轮翻到 ``--stop-after`` 条时中止（走生产里的 ``should_stop`` 回调），
   收尾时把进度写进断点文件；
2. 第二轮先 ``inspect`` 断点，把已翻好的译文填回字幕，再用 ``skip_translated=True``
   只翻剩下的。

脚本统计两轮各自发出的条目数，对着看就知道省了多少 —— 这正是断点存在的意义：
一部上千条的字幕跑几分钟，中断一次就从头再来，代价是分钟级的时间和额度。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import OUTPUT_DIR, ensure_runtime_dirs  # noqa: E402
from app.core import checkpoint, subtitle_io  # noqa: E402
from app.core.translator import EchoTranslator, TranslationCancelled  # noqa: E402

JA_LINES = [
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


class RecordingEcho(EchoTranslator):
    """记账版 echo：记下每一批真正发出去的文本。

    断言「翻过的条目没有被再翻一遍」不能只看结果译文（覆盖写上去的还是同一句话），
    必须看引擎**收到了什么**。
    """

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.batches = 0

    def translate_batch(self, requests):
        self.batches += 1
        self.sent.extend(request.text for request in requests)
        return super().translate_batch(requests)


def build_srt(count: int) -> str:
    """造一份字幕。每句都带序号后缀，保证**文本唯一** —— 校验「翻过的条目有没有
    被再翻一遍」靠的是文本比对，而真实字幕里重复台词很常见（这批语料只有 10 句），
    不唯一就永远能比中，判定会失效。
    """
    blocks = []
    for index in range(count):
        start = index * 2
        text = f"{JA_LINES[index % len(JA_LINES)]}（第 {index + 1} 句）"
        blocks.append(
            f"{index + 1}\n"
            f"{subtitle_io.format_timestamp(start)} --> "
            f"{subtitle_io.format_timestamp(start + 1.5)}\n{text}"
        )
    return "\n\n".join(blocks) + "\n"


def translate(cues, *, batch: int, skip: bool, stop_after: int | None = None):
    engine = RecordingEcho()
    if stop_after is None:
        engine.translate_cues(
            cues, source_lang="ja", target_lang="zh-CN",
            batch_size=batch, skip_translated=skip,
        )
        return engine
    try:
        engine.translate_cues(
            cues,
            source_lang="ja",
            target_lang="zh-CN",
            batch_size=batch,
            skip_translated=skip,
            # 取消在批与批之间生效 —— 与生产路径完全一致
            should_stop=lambda: len(engine.sent) >= stop_after,
        )
    except TranslationCancelled:
        pass
    return engine


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="验证翻译断点续传（离线，可重复）")
    parser.add_argument("--count", type=int, default=40, help="字幕条数")
    parser.add_argument("--batch", type=int, default=5, help="每批条数")
    parser.add_argument("--stop-after", type=int, default=12, help="翻到多少条时中断")
    args = parser.parse_args(argv)

    ensure_runtime_dirs()
    source = OUTPUT_DIR / "resume_demo.srt"
    source.write_text(build_srt(args.count), encoding="utf-8")
    cues = subtitle_io.parse_file(source)

    directory = checkpoint.checkpoints_dir()
    path = checkpoint.checkpoint_path(source, directory=directory)
    checkpoint.clear(path)

    print(f"字幕 {len(cues)} 条 ｜ 每批 {args.batch} 条 ｜ 打算翻到第 {args.stop_after} 条时中断")
    print(f"字幕文件 {source}")
    print(f"断点目录 {directory}")
    print()

    # ---------------------------------------------------------- 第一轮：中断
    print("第一轮")
    first = translate(cues, batch=args.batch, skip=False, stop_after=args.stop_after)
    # 等价于 worker 在线程收尾时做的那次强制落盘
    checkpoint.save(
        path, cues, source_path=source,
        engine="echo", model="", source_lang="ja", target_lang="zh-CN",
    )
    done = len(first.sent)
    print(f"  实际翻好        {done} 条（{first.batches} 批）"
          "　← 取消在批与批之间生效，所以会翻到批边界，比预期多几条")
    print(f"  断点文件        {path}")
    print(f"  记录条数        {len(checkpoint.load(path)['entries'])}")
    print()

    # ---------------------------------------------------------- 第二轮：续传
    print("第二轮（续传）")
    plan = checkpoint.inspect(
        source, cues, engine="echo", model="",
        source_lang="ja", target_lang="zh-CN", directory=directory,
    )
    for index, text in plan.usable.items():
        cues[index].translation = text
    rest = translate(cues, batch=args.batch, skip=True)
    # 界面在整份翻完之后会把记录删掉，脚本这里手工做同一件事
    checkpoint.clear(path)
    print(f"  可复用          {plan.count} 条")
    print(f"  仍需翻译        {len(rest.sent)} 条（{rest.batches} 批）")
    print()

    # ---------------------------------------------------------- 对照与校验
    whole = translate(subtitle_io.parse_file(source), batch=args.batch, skip=False)
    saved = len(whole.sent) - len(rest.sent)
    print("对照（不续传 vs 续传）")
    print(f"  不续传需要发的条目  {len(whole.sent)} 条（{whole.batches} 批）")
    print(f"  续传实际发出        {len(rest.sent)} 条（{rest.batches} 批）")
    print(f"  省下                {saved} 条"
          f"（{saved / max(1, len(whole.sent)):.0%}，折合 {done} 条已完成的工作）")
    print()

    print("校验")
    first_sent = set(first.sent)
    repeated = [text for text in rest.sent if text in first_sent]
    untranslated = [cue for cue in cues if not cue.is_translated]
    leftover = path.exists()
    print(f"  已翻条目被重复翻译  {'是（有问题）' if repeated else '否'}")
    print(f"  最终未翻译条目      {len(untranslated)}")
    print(f"  断点文件是否清理    {'否（有问题）' if leftover else '是（整份翻完）'}")

    ok = not repeated and not untranslated and not leftover and saved == done
    checkpoint.clear(path)
    print()
    print("结论：" + ("断点续传按预期工作" if ok else "有不符合预期的地方，见上面各栏"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
