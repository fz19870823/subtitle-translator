"""翻译断点：把已经翻好的条目落盘，中断之后还能接着翻。

一部上千条的字幕要发几十次请求、跑好几分钟。中途「取消」、关掉程序、
中继抽风、机器休眠 —— 任何一种都会让这一轮白跑。用户重新点「翻译」时若从
第 1 条重来，前面几分钟的时间和额度就都丢了。所以每翻完一批就把进度写下来，
下次接着走。

## 落盘位置

``data/checkpoints/<源文件stem>-<路径摘要>.json``（``data/`` 已在 .gitignore）。
路径里带着源文件路径的摘要，两份同名文件（不同目录）不会互相踩。

## 什么时候能复用

必须**同时**满足：

- 源文件路径一致；
- 引擎、模型、源语言、目标语言一致；
- 逐条**原文一字不差**。

前两条是「同一件事」的定义。模型换了不续用，是因为不同模型的行文风格不同，
接在一起会出现前后语气断裂，而且用户换模型通常就是想整份重来。
第三条比看上去重要：只对条数是不够的 —— 字幕文件被改过而条数没变时，
按序号硬套会把 A 条的译文写到 B 条上，交付出去几乎不可能被发现。

任一条件不满足就返回空计划（**不删记录**）：用户把模型改回去还能接着用。
真正的删除只发生在整份翻完之后。

## 写入方式

先写 ``.tmp`` 再 ``os.replace`` 原子替换（与 ``save_config`` 同一套路），
断电或被杀进程都不会留下半截 JSON。读取时遇到坏文件一律当作「没有断点」，
绝不因为一个缓存文件把翻译挡住。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence

from app.core.subtitle_io import Cue

#: 记录格式版本。字段含义变化时 +1，旧记录会被当成「不可用」而忽略。
CHECKPOINT_VERSION = 1

#: 文件名里不允许出现的字符（Windows 限制更严，按它的来）
_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
#: Windows 保留设备名，撞上就没法建文件
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _norm(value: object) -> str:
    """指纹比较用的归一化：去空白 + 忽略大小写。

    路径在 Windows 上不分大小写，模型名也常被手输成大写的另一种写法，
    不归一化就会把「同一个任务」判成两个。
    """
    return str(value or "").strip().casefold()


def _digest(path: Path) -> str:
    return hashlib.sha1(_norm(path).encode("utf-8")).hexdigest()[:10]


def _safe_stem(name: str) -> str:
    cleaned = _UNSAFE_RE.sub("_", name).strip(" .") or "subtitle"
    if cleaned.casefold().split(".")[0] in _RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned[:48]


def checkpoints_dir() -> Path:
    """断点目录（顺带创建）。测试里可替换 ``app.config.CHECKPOINT_DIR`` 隔离。"""
    import app.config as app_config

    directory = Path(app_config.CHECKPOINT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def checkpoint_path(source_path: str | Path, *, directory: str | Path | None = None) -> Path:
    """某份字幕对应的断点文件路径。同一份字幕永远映射到同一个文件。"""
    resolved = Path(source_path).expanduser().resolve()
    root = Path(directory) if directory is not None else checkpoints_dir()
    return root / f"{_safe_stem(resolved.stem)}-{_digest(resolved)}.json"


@dataclass
class ResumePlan:
    """「能不能接着上次翻」的结论。

    ``usable`` 是**可以直接用**的 ``{条目下标: 译文}``；``rejected`` 非空时
    表示发现了记录但不能用（并说明原因），界面照此提示用户，别让它静默失效。
    """

    path: Path | None = None
    exists: bool = False
    usable: Dict[int, str] = field(default_factory=dict)
    total: int = 0
    saved: int = 0
    updated_at: str = ""
    rejected: str = ""

    @property
    def count(self) -> int:
        """能复用的条数。"""
        return len(self.usable)

    @property
    def stale(self) -> bool:
        """有记录但一条都用不上。"""
        return self.exists and not self.usable


def snapshot(cues: Sequence[Cue]) -> Dict[int, Dict[str, str]]:
    """挑出已有译文的条目，作为要写盘的内容。"""
    return {
        index: {"s": cue.text, "t": cue.translation}
        for index, cue in enumerate(cues)
        if cue.is_translated
    }


def save(
    path: str | Path,
    cues: Sequence[Cue],
    *,
    source_path: str | Path,
    engine: str = "",
    model: str = "",
    source_lang: str = "",
    target_lang: str = "",
) -> Path | None:
    """原子写入断点。一条都没翻时返回 None（不建空文件）。"""
    entries = snapshot(cues)
    if not entries:
        return None

    payload = {
        "version": CHECKPOINT_VERSION,
        "source_file": str(Path(source_path).expanduser().resolve()),
        "total": len(cues),
        "engine": engine,
        "model": model,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "updated_at": _now(),
        "entries": {str(index): item for index, item in sorted(entries.items())},
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return target


def load(path: str | Path) -> dict | None:
    """读断点记录。文件不存在 / 坏掉 / 版本不符都返回 None。"""
    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != CHECKPOINT_VERSION:
        return None
    if not isinstance(payload.get("entries"), dict):
        return None
    return payload


def clear(path: str | Path | None) -> bool:
    """删除断点记录。返回是否真的删掉了一个文件。"""
    if path is None:
        return False
    target = Path(path)
    try:
        target.unlink()
    except (FileNotFoundError, OSError):
        return False
    return True


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class CheckpointWriter:
    """在**后台线程**里按批把进度写进断点文件。

    为什么要有节流：一批 3–5 秒、上千条字幕上百批，每批都写盘就是上百次
    几 KB 的写入。虽然开销很小，但 Windows Defender 会跟着扫，纯属浪费。
    默认两秒一次足够 —— 真丢了也只是重翻最后两秒的那一批。

    ``error`` 记下最近一次写失败的原因（磁盘满、目录只读）。界面要把它说出来：
    「断点没存上」是用户会误判的事，以为能续传结果不能，比不提供断点更糟。
    """

    path: Path
    source_path: Path | str
    engine: str = ""
    model: str = ""
    source_lang: str = ""
    target_lang: str = ""
    #: 两次写盘的最小间隔（秒）；0 表示每批都写
    interval: float = 2.0
    clock: Callable[[], float] = time.monotonic
    error: str = ""

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        # 0 而不是 clock()：进程刚起来时 clock() 也可能很小，用 0 保证**第一批必写**。
        # 否则前两秒内崩掉就一点记录都没留下。
        self._last = 0.0

    def maybe(self, cues: Sequence[Cue]) -> bool:
        """到点了才写。返回本次是否真的落盘。"""
        if self.interval > 0 and (self.clock() - self._last) < self.interval:
            return False
        return self.flush(cues)

    def flush(self, cues: Sequence[Cue]) -> bool:
        """立刻写盘（取消、失败、整批收尾时用，不受节流限制）。"""
        try:
            written = save(
                self.path,
                cues,
                source_path=self.source_path,
                engine=self.engine,
                model=self.model,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )
        except OSError as exc:
            self.error = str(exc)
            return False
        self._last = self.clock()
        self.error = ""
        return written is not None


def _mismatch(
    payload: Mapping,
    *,
    source_path: str | Path,
    engine: str,
    model: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """检查任务指纹，返回不可用的原因；全部一致返回空串。"""
    recorded_file = payload.get("source_file")
    if recorded_file and _norm(Path(str(recorded_file)).expanduser().resolve()) != _norm(
        Path(source_path).expanduser().resolve()
    ):
        return "记录属于另一份字幕文件"
    if _norm(payload.get("engine")) != _norm(engine):
        return f"上次用的引擎是 {payload.get('engine') or '(空)'}，这次是 {engine or '(空)'}"
    if _norm(payload.get("model")) != _norm(model):
        return (
            f"上次用的模型是 {payload.get('model') or '(空)'}，"
            f"这次是 {model or '(空)'}"
        )
    if _norm(payload.get("source_lang")) != _norm(source_lang):
        return (
            f"上次的源语言是 {payload.get('source_lang') or '(空)'}，"
            f"这次是 {source_lang or '(空)'}"
        )
    if _norm(payload.get("target_lang")) != _norm(target_lang):
        return (
            f"上次的目标语言是 {payload.get('target_lang') or '(空)'}，"
            f"这次是 {target_lang or '(空)'}"
        )
    return ""


def inspect(
    source_path: str | Path,
    cues: Sequence[Cue],
    *,
    engine: str = "",
    model: str = "",
    source_lang: str = "",
    target_lang: str = "",
    directory: str | Path | None = None,
) -> ResumePlan:
    """判断这份字幕上次的翻译记录现在还能不能接着用。

    这是界面唯一需要的入口：拿它返回的 ``usable`` 填回 ``cue.translation``，
    再让引擎只翻剩下的条目。
    """
    path = checkpoint_path(source_path, directory=directory)
    payload = load(path)
    if payload is None:
        return ResumePlan(path=path)

    entries = payload["entries"]
    plan = ResumePlan(
        path=path,
        exists=True,
        total=int(payload.get("total") or 0),
        saved=len(entries),
        updated_at=str(payload.get("updated_at") or ""),
    )

    reason = _mismatch(
        payload,
        source_path=source_path,
        engine=engine,
        model=model,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    if reason:
        plan.rejected = reason
        return plan

    for key, item in entries.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(cues) or not isinstance(item, dict):
            continue
        source = item.get("s")
        translation = item.get("t")
        if not isinstance(source, str) or not isinstance(translation, str):
            continue
        # 原文对不上就不敢用：条数没变但内容被改过时，序号会整体错位。
        if cues[index].text != source:
            continue
        if translation.strip():
            plan.usable[index] = translation

    if not plan.usable:
        plan.rejected = "记录里的条目与当前字幕对不上（原文被改过了？）"
    return plan
