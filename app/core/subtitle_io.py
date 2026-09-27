"""字幕文件的解析与序列化。

只处理 SRT 与 WebVTT：这两种格式在「读入 -> 翻译 -> 导出」链路上可以做到
无损往返。ASS/SSA 携带排版与样式，纯文本模型会破坏它们，因此暂不纳入。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence


class SubtitleFormatError(ValueError):
    """字幕文件无法解析。"""


#: 「该翻译、但用尽重试仍没翻出来」的条目在导出时的内容。
#:
#: 刻意**不写原文**：把原文填回去，「模型没翻」和「本来就不需要翻」就再也分不出来了
#: —— 用户拿到一份看起来完整的字幕，实际有一半根本没翻译，而且毫无迹象。
#: 写出一个显眼的标记，是把「没翻」摆到台面上，而不是把它藏起来。
UNTRANSLATED_MARK = "[未翻译]"


_TIMECODE_RE = re.compile(
    r"^(?P<h>\d{1,2}):(?P<m>\d{1,2}):(?P<s>\d{1,2})[,.](?P<ms>\d{1,3})$"
)

# 尝试顺序：带/不带 BOM 的 UTF-8，再退到中文环境常见的 GB18030。
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


@dataclass
class Cue:
    """单条字幕。start / end 单位为秒。"""

    index: int
    start: float
    end: float
    text: str
    translation: str = ""
    #: 这条「该翻译、但重试用尽也没翻出来」。导出时写成 :data:`UNTRANSLATED_MARK`，
    #: 续传与重试时会被重新送去翻译。
    #:
    #: 必须是**显式标记**、不能靠「译文为空」推断：纯符号行（``♪``、``123``）
    #: 本来就不送模型，译文同样是空的，可两者该导出成完全不同的东西。
    failed: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def is_translated(self) -> bool:
        """是否已有非空译文。

        断点续传靠它挑出「还欠一次翻译」的条目。空白译文按未翻译算 ——
        模型偶尔会吐回空串，那不是可交付的结果。
        """
        return bool(self.translation.strip())

    def render_text(self) -> str:
        """导出时使用的内容。

        三种情况必须分开，不能图省事回落到原文：

        - 有译文 → 用译文；
        - 该翻译却翻失败了（``failed``）→ 写 :data:`UNTRANSLATED_MARK`。
          留空等于把这几条从字幕里抹掉，观看时只会觉得「这儿怎么没字幕」；
        - 其余（纯符号行、数字行、日期之类本来就不需要翻译的）→ 保留原文。
        """
        if self.translation.strip():
            return self.translation.strip()
        if self.failed:
            return UNTRANSLATED_MARK
        return self.text.strip()


def parse_timestamp(value: str) -> float:
    """把 ``00:01:02,500`` / ``0:1:2.5`` 解析为秒。"""
    match = _TIMECODE_RE.match(value.strip())
    if not match:
        raise SubtitleFormatError(f"无法解析时间码: {value!r}")
    parts = match.groupdict()
    milliseconds = int(parts["ms"].ljust(3, "0"))
    return (
        int(parts["h"]) * 3600
        + int(parts["m"]) * 60
        + int(parts["s"])
        + milliseconds / 1000.0
    )


def format_timestamp(seconds: float, separator: str = ",") -> str:
    """把秒格式化为 ``HH:MM:SS<sep>mmm``。SRT 用逗号，VTT 用句点。"""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def _blocks(content: str) -> List[str]:
    """按空行切分字幕块，统一换行符并剥掉 BOM。"""
    text = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    return [block for block in re.split(r"\n{2,}", text.strip()) if block.strip()]


def _split_timing(line: str) -> tuple:
    """从 ``start --> end`` 行解析出 (start, end)。"""
    raw_start, arrow, raw_end = line.partition("-->")
    if not arrow:
        raise SubtitleFormatError(f"缺少时间轴箭头: {line!r}")
    # VTT 允许在结束时间后追加排版参数，如 "00:00:02.000 align:start"
    end_token = raw_end.strip().split(" ")[0]
    return parse_timestamp(raw_start), parse_timestamp(end_token)


def parse_srt(content: str) -> List[Cue]:
    cues: List[Cue] = []
    for block in _blocks(content):
        lines = block.split("\n")
        index = len(cues) + 1
        cursor = 0
        if lines[0].strip().isdigit():
            index = int(lines[0].strip())
            cursor = 1
        if cursor >= len(lines):
            raise SubtitleFormatError(f"第 {index} 条字幕缺少时间轴")
        start, end = _split_timing(lines[cursor])
        cues.append(
            Cue(
                index=index,
                start=start,
                end=end,
                text="\n".join(lines[cursor + 1:]).strip(),
            )
        )
    if not cues:
        raise SubtitleFormatError("未解析到任何字幕条目")
    return cues


def parse_vtt(content: str) -> List[Cue]:
    cues: List[Cue] = []
    for block in _blocks(content):
        lines = block.split("\n")
        head = lines[0].strip().upper()
        # WEBVTT 文件头，以及 NOTE / STYLE / REGION 元信息块，都不是字幕。
        if head.startswith("WEBVTT") or head.split(" ")[0] in {"NOTE", "STYLE", "REGION"}:
            continue
        # 时间轴行可能被一个 cue 标识行顶到第二行。
        cursor = 0 if "-->" in lines[0] else 1
        if cursor >= len(lines) or "-->" not in lines[cursor]:
            continue
        start, end = _split_timing(lines[cursor])
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=start,
                end=end,
                text="\n".join(lines[cursor + 1:]).strip(),
            )
        )
    if not cues:
        raise SubtitleFormatError("未解析到任何字幕条目")
    return cues


def read_text(path: str | Path) -> str:
    """读取字幕文本，依次尝试常见编码。"""
    target = Path(path)
    for encoding in _ENCODINGS:
        try:
            return target.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise SubtitleFormatError(f"无法用 {_ENCODINGS} 解码: {target}")


def parse_file(path: str | Path) -> List[Cue]:
    """按扩展名选择解析器。"""
    target = Path(path)
    content = read_text(target)
    suffix = target.suffix.lower()
    if suffix == ".srt":
        return parse_srt(content)
    if suffix == ".vtt":
        return parse_vtt(content)
    raise SubtitleFormatError(
        f"不支持的扩展名: {target.suffix!r}（当前支持 .srt / .vtt）"
    )


def to_srt(cues: Iterable[Cue]) -> str:
    blocks = []
    for position, cue in enumerate(cues, start=1):
        timing = f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}"
        blocks.append(f"{position}\n{timing}\n{cue.render_text()}")
    return "\n\n".join(blocks) + "\n"


def to_vtt(cues: Iterable[Cue]) -> str:
    blocks = ["WEBVTT"]
    for cue in cues:
        timing = (
            f"{format_timestamp(cue.start, '.')} --> {format_timestamp(cue.end, '.')}"
        )
        blocks.append(f"{timing}\n{cue.render_text()}")
    return "\n\n".join(blocks) + "\n"


def write_file(
    path: str | Path,
    cues: Iterable[Cue],
    *,
    as_vtt: bool | None = None,
) -> Path:
    """写出字幕。``as_vtt`` 为 None 时按目标文件扩展名决定格式。

    **原子写**：先落 ``.tmp`` 再 ``os.replace``。翻译就是靠这个函数**边翻边写**的，
    用户随时可能打开那个文件；直接覆写的话，撞上写了一半的瞬间会看到一个被截断的
    字幕，而播放器只会闷声不响地少放半集。
    """
    target = Path(path)
    if as_vtt is None:
        as_vtt = target.suffix.lower() == ".vtt"
    content = to_vtt(cues) if as_vtt else to_srt(cues)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8", newline="\n")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return target


@dataclass
class OutputSink:
    """在**后台线程**里把当前译文按批写进目标字幕文件。

    为什么与断点文件分开：断点（``data/checkpoints``）解决的是「下次还能接着翻」，
    它躺在内部目录、格式也不是字幕，用户拿不到手。这里写的是**成品本身** ——
    翻到哪，打开那个 .srt 就能看到哪；中途关窗、掉电、崩溃，已经翻好的部分都在。

    节流沿用 ``CheckpointWriter`` 的思路：一批 3–5 秒、上千条要几十上百批，每批都
    写盘纯属浪费（Windows Defender 会跟着扫一遍）。默认两秒一次，真丢了也只是
    重翻最后两秒的那一批。

    ``error`` 记下最近一次写失败的原因（磁盘满、目录只读）：用户以为译文在正常
    落盘、实际没有，是最该趁早说出来的一件事。
    """

    path: Path
    #: 两次写盘的最小间隔（秒）；0 表示每批都写
    interval: float = 2.0
    clock: Callable[[], float] = time.monotonic
    error: str = ""

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        # 0 而不是 clock()：进程刚起来时 clock() 也可能很小，用 0 保证**第一批必写**。
        # 否则前两秒内崩掉，磁盘上连一个字节都没有。
        self._last = 0.0

    def maybe(self, cues: Sequence[Cue]) -> bool:
        """到点了才写。返回本次是否真的落了盘。"""
        if self.interval > 0 and (self.clock() - self._last) < self.interval:
            return False
        return self.flush(cues)

    def flush(self, cues: Sequence[Cue]) -> bool:
        """立刻写一次（取消、失败、收尾时用，不受节流限制）。"""
        try:
            write_file(self.path, cues)
        except OSError as exc:
            self.error = str(exc)
            return False
        self._last = self.clock()
        self.error = ""
        return True
