"""批量翻译队列：一次排入多个字幕文件，依次翻完并自动导出。

为什么要有这一层：单份字幕是「打开 → 翻译 → 导出」，人守着就行；十几集的剧集
按这个流程走一遍，光是重复点按钮就够烦，而且中途一走开就停在原地。队列把
「下一个是谁、翻完了没、结果存到哪」这些记账从界面里抽出来，界面只管显示和点按钮。

刻意**不含**任何 Qt 依赖、也不做网络与磁盘 I/O（读写字幕由调用方负责，因为它
还需要 ``cues``）。这样整套推进逻辑能在只装了 pytest 的那套解释器里跑测试 ——
本机 PySide6 与 pytest 装在两套不同的 Python 里（见 README 的环境说明）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from app.config import SUPPORTED_EXTENSIONS

#: 队列条目的状态。用字符串而不是 Enum：它要显示在界面上、写进日志，
#: 字符串更好读，也让 core 层不必为了一个枚举多引一层依赖。
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

#: 「状态 → 中文标签」由界面负责（core 不管显示），这里只声明有哪些状态。
STATUSES = (PENDING, RUNNING, DONE, FAILED)


def _norm(path: Path) -> str:
    """去重用的归一化路径。Windows 上不分大小写，还可能有 ``./`` 这类写法。"""
    expanded = path.expanduser()
    try:
        resolved = expanded.resolve()
    except OSError:  # 路径离谱到 resolve 都报错时退回绝对路径，别让入队整个失败
        resolved = expanded.absolute()
    return str(resolved).casefold()


@dataclass
class QueueItem:
    """队列里的一份字幕。

    ``path`` 是源文件，``output_path`` 是译文的去处。两者在入队时**一起**定下来：
    译文路径要提前显示给用户看，等翻完再决定放哪就太晚了 —— 同一批里有两个
    ``01.srt``（来自不同目录）时，用户有权在开跑前就知道第二个会被改名，
    而不是事后发现自己的成果被覆盖了。
    """

    path: Path
    output_path: Path
    status: str = PENDING
    #: 字幕条数（入队时解析一次填上）。界面用它显示规模，也用来预估总工作量。
    cue_count: int = 0
    #: 翻完后仍有几条没翻出来（对应界面上的「重试未翻译」数量）。
    untranslated: int = 0
    #: 这一份复用了多少条上次的进度（0 表示从头翻）。
    resumed: int = 0
    #: 失败原因（解析失败 / 引擎构造失败 / 翻译失败 / 导出失败）。
    error: str = ""
    #: 这一轮只重翻「上次没翻出来」的那几条，不是整份重来。
    #: 由队列收尾后的「重试未翻译」设置（见 :meth:`TranslationQueue.mark_retry_round`）。
    retry_only: bool = False

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_done(self) -> bool:
        return self.status == DONE

    def one_line_error(self, limit: int = 80) -> str:
        """把失败原因压成一行 —— 它要显示在列表里，换行会把行高撑开。"""
        text = " ".join((self.error or "").split())
        return text if len(text) <= limit else text[: limit - 1] + "…"


class TranslationQueue:
    """队列的状态机。

    只管「有哪些文件、各是什么状态、下一个该翻谁」，不碰界面、不做 I/O。
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._items: List[QueueItem] = []
        self._running: int | None = None

    # ---------------------------------------------------------------- 读取

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index: int) -> QueueItem:
        return self._items[index]

    @property
    def items(self) -> Tuple[QueueItem, ...]:
        """当前内容的只读快照。"""
        return tuple(self._items)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def running_index(self) -> int | None:
        """正在翻的那一项的下标；None 表示没有在跑的。"""
        return self._running

    def index_of(self, path: str | Path | None) -> int | None:
        """这一份在队列里的位置；不在队列里（或 ``path`` 是 None）返回 ``None``。

        界面靠它让「编辑器里显示的那份」和「列表里选中的那行」始终一致。
        两者一旦对不上，用户在译文区看到的内容和他以为在看的那一份就不是同一个
        文件 —— 而且界面上没有任何地方能看出这点，正是最难发现的那类错。
        """
        if path is None:
            return None
        key = _norm(Path(path))
        for index, item in enumerate(self._items):
            if _norm(item.path) == key:
                return index
        return None

    def contains(self, path: str | Path) -> bool:
        return self.index_of(path) is not None

    # ---------------------------------------------------------------- 增删

    def add(self, paths: Iterable[str | Path]) -> Tuple[List[QueueItem], List[str]]:
        """把若干文件排进队列。

        返回 ``(真正入队的条目, 被跳过的原因)``。第二个返回值不能省：用户一次选了
        20 个文件，只进来 18 个却什么都不说，他会以为全在里面 —— 等队列跑完
        才发现少了两集。
        """
        added: List[QueueItem] = []
        notes: List[str] = []
        taken = {_norm(item.path) for item in self._items}
        outputs = {item.output_path for item in self._items}

        for raw in paths:
            path = Path(raw)
            key = _norm(path)
            if key in taken:
                notes.append(f"{path.name}：已在队列中")
                continue
            if not path.is_file():
                notes.append(f"{path.name}：不是可读的文件")
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                notes.append(f"{path.name}：不支持 {path.suffix or '（无扩展名）'}")
                continue
            item = QueueItem(path=path, output_path=self._unique_output(path, outputs))
            outputs.add(item.output_path)
            taken.add(key)
            self._items.append(item)
            added.append(item)
        return added, notes

    def remove(self, item: QueueItem) -> bool:
        """移出一项。**正在翻的那一项不能移** —— 后台线程还在往它的 cues 里写。"""
        if item.status == RUNNING:
            return False
        try:
            self._items.remove(item)
        except ValueError:
            return False
        # 索引全变了，旧的下标不再指向任何东西，留着只会误导调用方。
        self._running = None
        return True

    def clear(self) -> None:
        """清空队列。正在翻的时候不允许 —— 同样的理由。"""
        if self._running is not None:
            return
        self._items.clear()

    def move(self, index: int, delta: int) -> int:
        """把第 ``index`` 项上移/下移 ``delta`` 位，返回它的新位置。"""
        target = index + delta
        if not 0 <= index < len(self._items) or not 0 <= target < len(self._items):
            return index
        self._items[index], self._items[target] = self._items[target], self._items[index]
        return target

    # ---------------------------------------------------------------- 状态

    def next_pending(self) -> int | None:
        """下一个待翻项的下标。

        只看 ``PENDING`` ——「失败」的**不**当场自动重来。否则一个注定失败的文件
        （比如整条链路都挂了）会被反复重发，队列永远跑不完。
        让用户处理完问题再点一次「依次翻译」，那时它会被重置回待翻译。
        """
        for index, item in enumerate(self._items):
            if item.status == PENDING:
                return index
        return None

    def mark_running(self, index: int) -> None:
        if 0 <= index < len(self._items):
            self._items[index].status = RUNNING
            self._running = index

    def mark_done(self, index: int) -> None:
        """标记为完成。失败原因顺手清掉 —— 重跑成功后不该还挂着上次的报错。"""
        if 0 <= index < len(self._items):
            item = self._items[index]
            item.status = DONE
            item.error = ""
        self._running = None

    def fail(self, index: int, error: str) -> None:
        if 0 <= index < len(self._items):
            item = self._items[index]
            item.status = FAILED
            item.error = error
        self._running = None

    def reset(self, index: int) -> None:
        """把一项打回「待翻译」，清掉上一轮的结论。"""
        if 0 <= index < len(self._items):
            item = self._items[index]
            item.status = PENDING
            item.untranslated = 0
            item.resumed = 0
            item.error = ""
            item.retry_only = False
        self._running = None

    def reset_unfinished(self) -> int:
        """把所有没成功的项打回「待翻译」，返回重置了几项。

        队列跑过一轮之后再点「依次翻译」，走的就是这条路：上一轮失败（或没轮到，
        比如用户中途取消）的可以重来，已经翻好的不会白跑第二遍。
        """
        count = 0
        for index, item in enumerate(self._items):
            if item.status != DONE:
                self.reset(index)
                count += 1
        return count

    def reset_all(self) -> int:
        """全部打回「待翻译」，返回重置了几项。用于「重新翻译全部」。"""
        for index in range(len(self._items)):
            self.reset(index)
        return len(self._items)

    def mark_retry_round(self) -> int:
        """把「翻完了、但还剩未翻译条目」的项打回待翻译，返回有几项要重试。

        这是队列跑完后的第二手。那些条目在引擎里已经自动重试过三次，按同样的
        前提再跑一遍不会改善 —— 真正能改变结局的是**换个前提**（换个模型、
        或把上下文窗口调小），所以单独开一条路：只碰这几份，且每份只重翻
        ``failed`` 的那几条，已经翻好的一个字不动。

        返回 0 表示没有任何文件还剩未翻译条目，调用方据此拒绝启动空跑。
        """
        count = 0
        for index, item in enumerate(self._items):
            if item.status == DONE and item.untranslated:
                self.reset(index)
                self._items[index].retry_only = True
                count += 1
        return count

    def summary(self) -> dict:
        """给界面和汇总报告用的计数。"""
        done = failed = pending = 0
        untranslated = resumed = cue_count = 0
        stuck_files = 0
        for item in self._items:
            if item.status == DONE:
                done += 1
            elif item.status == FAILED:
                failed += 1
            else:
                pending += 1
            untranslated += item.untranslated
            if item.untranslated:
                stuck_files += 1
            resumed += item.resumed
            cue_count += item.cue_count
        return {
            "total": len(self._items),
            "done": done,
            "failed": failed,
            "pending": pending,
            #: 还需要跑的文件数（待翻译 + 失败）——「依次翻译」按钮上的数字
            "unfinished": pending + failed,
            "untranslated": untranslated,
            #: 还剩未翻译条目的文件数 ——「重试未翻译」按钮据此决定要不要出现
            "stuck_files": stuck_files,
            "resumed": resumed,
            "cue_count": cue_count,
        }

    # ---------------------------------------------------------------- 落点

    def _unique_output(self, path: Path, taken: set) -> Path:
        """算出这份字幕的译文路径。

        同一批里出现两个同名文件（不同目录的 ``01.srt``）时，后来者加序号 ——
        否则后翻完的会把先翻完的覆盖掉，而界面上两份都显示「已完成」。

        磁盘上已存在的同名译文**不加序号**：重跑一遍本来就是要覆盖上一轮的结果，
        加序号只会让输出目录里堆起一堆 ``-2`` ``-3``，谁是最新的反而看不出来。
        """
        stem = path.stem
        suffix = path.suffix.lower()
        candidate = self._output_dir / f"{stem}.translated{suffix}"
        serial = 1
        while candidate in taken:
            serial += 1
            candidate = self._output_dir / f"{stem}.translated-{serial}{suffix}"
        return candidate
