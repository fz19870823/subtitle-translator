"""字幕文件的解析与序列化。

只处理 SRT 与 WebVTT：这两种格式在「读入 -> 翻译 -> 导出」链路上可以做到
无损往返。ASS/SSA 携带排版与样式，纯文本模型会破坏它们，因此暂不纳入。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


class SubtitleFormatError(ValueError):
    """字幕文件无法解析。"""


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

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def render_text(self) -> str:
        """导出时使用的内容：有译文就用译文，否则回落到原文。"""
        return self.translation.strip() or self.text.strip()


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
    """写出字幕。``as_vtt`` 为 None 时按目标文件扩展名决定格式。"""
    target = Path(path)
    if as_vtt is None:
        as_vtt = target.suffix.lower() == ".vtt"
    content = to_vtt(cues) if as_vtt else to_srt(cues)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target
