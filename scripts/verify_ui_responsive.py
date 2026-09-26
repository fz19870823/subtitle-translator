"""验证：翻译期间界面线程有没有被阻塞。

背景：早期实现直接在界面线程里调 ``translate_cues()``，一部上千条的字幕要发
几十次请求、跑好几分钟。那段时间主线程被网络等待完全占住 —— 进度条不重绘、
按钮点不动、连「取消」都点不了，用户看到的就是程序卡死。现在翻译跑在
:class:`~app.ui.translate_worker.TranslateWorker` 里，主线程只做信号投递。

**度量方式**：起一个 20ms 周期的 QTimer 打点，记录事件循环相邻两次打点的间隔。
间隔涨到「一批的耗时」（几秒）就说明主线程被占住了；几十毫秒才是正常。

    python scripts/verify_ui_responsive.py            # 后台线程（当前实现）
    python scripts/verify_ui_responsive.py --sync     # 复现修复前的做法，做对照
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 必须在导入 PySide6 之前设好离屏，否则无头环境下起不来
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core import subtitle_io  # noqa: E402
from app.core.translator import create_engine_for  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

#: 打点周期。远小于单批耗时（秒级），足够分辨「阻塞」与「正常」。
TICK_MS = 20

#: 一遍语料循环用的日文句子，够看出翻译是否真的在跑。
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


def build_srt(count: int) -> str:
    """造一份 count 条的日文字幕。"""
    blocks = []
    for index in range(count):
        start = index * 2
        blocks.append(
            f"{index + 1}\n"
            f"{_stamp(start)} --> {_stamp(start + 1)}\n"
            f"{JA_LINES[index % len(JA_LINES)]}\n"
        )
    return "\n".join(blocks)


def _stamp(seconds: int) -> str:
    return f"00:{seconds // 60:02d}:{seconds % 60:02d},000"


def measure_sync(window, cues, source_lang: str, target_lang: str) -> tuple[object, str]:
    """修复前的做法：直接在界面线程里跑完整翻译。"""
    engine = create_engine_for(window.engine_combo.currentText().strip(), window._config)
    try:
        engine.translate_cues(cues, source_lang=source_lang, target_lang=target_lang)
    except Exception as exc:  # noqa: BLE001 - 度量脚本，异常也要记下来
        return engine, f"{type(exc).__name__}: {exc}"
    return engine, ""


def measure_async(window, app) -> tuple[object, str]:
    """当前实现：交给后台线程，主线程只跑事件循环。"""
    window._on_translate()
    worker = window._worker
    if worker is None:
        raise RuntimeError("翻译线程没有启动（配置或语言选择有问题？）")

    loop = QEventLoop()
    worker.succeeded.connect(loop.quit)
    worker.cancelled.connect(lambda *_: loop.quit())
    worker.failed.connect(lambda *_: loop.quit())
    loop.exec()
    app.processEvents()  # 让收尾槽跑完
    return worker.engine, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="字幕条数")
    parser.add_argument("--sync", action="store_true", help="复现修复前的同步调用")
    parser.add_argument("--out", default="output/ui_responsive.txt")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    cues = subtitle_io.parse_srt(build_srt(args.count))
    window._cues = cues
    window._source_path = Path("probe.srt")
    window.translate_button.setEnabled(True)
    window.export_button.setEnabled(True)

    source_lang = window.source_combo.currentText().strip()
    target_lang = window.target_combo.currentText().strip()
    model = window.model_selector.current_model() or "(未设置)"
    if window.engine_combo.currentText().strip() != "echo":
        window._config.translation.model = model

    ticks: list[float] = []
    timer = QTimer()
    timer.setInterval(TICK_MS)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()

    started = time.monotonic()
    if args.sync:
        engine, error = measure_sync(window, cues, source_lang, target_lang)
    else:
        engine, error = measure_async(window, app)
    elapsed = time.monotonic() - started
    timer.stop()

    # 只看翻译期间的打点：间隔涨到秒级就是主线程被占住了。
    span = [t for t in ticks if started <= t <= elapsed + started]
    if len(span) >= 2:
        gaps = [b - a for a, b in zip(span, span[1:])]
        max_gap = max(gaps)
        mid_gap = sorted(gaps)[len(gaps) // 2]
        tick_count = len(span)
    else:
        # 一次都没打上点 —— 主线程整段时间没有任何事件处理，这就是最严重的阻塞。
        max_gap = elapsed
        mid_gap = elapsed
        tick_count = 0

    mode = "sync（修复前）" if args.sync else "async（当前）"
    batch = getattr(engine, "batch_size", 0)
    batches = (len(cues) + batch - 1) // batch if batch else 0
    notes = engine.quality_notes() if engine is not None else []

    lines = [
        "=" * 96,
        "翻译期间界面线程响应性",
        "=" * 96,
        f"模式      : {mode}",
        f"引擎/模型 : {window.engine_combo.currentText().strip()} / {model}",
        f"方向      : {source_lang} -> {target_lang}",
        f"字幕条数  : {len(cues)}   批次大小: {batch}   批数: {batches}",
        f"总耗时    : {elapsed:.2f}s",
        "",
        f"主线程最长一次无响应 : {max_gap * 1000:.0f} ms",
        f"打点间隔中位数       : {mid_gap * 1000:.0f} ms   （周期 {TICK_MS} ms）",
        f"翻译期间打点次数     : {tick_count}",
        "",
        f"翻译质量插曲 : {'；'.join(notes) if notes else '（无）'}",
    ]
    if error:
        lines.append(f"异常         : {error}")

    lines.append("")
    lines.append("=" * 96)
    if max_gap >= 1.0:
        lines.append(f"结论：主线程被独占 {max_gap:.1f}s —— 界面在这段时间里完全冻结。")
    else:
        lines.append(
            f"结论：主线程最长只停了 {max_gap * 1000:.0f} ms，界面保持响应"
            f"（对比单批请求约 {elapsed / max(batches, 1):.1f}s）。"
        )
    lines.append("=" * 96)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
