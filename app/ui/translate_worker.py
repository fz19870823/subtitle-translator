"""翻译任务的后台线程。

**为什么必须有这一层**：一部上千条的字幕要分批发几十次网络请求，每批 3–5 秒，
整个过程是**分钟级**的。若直接在界面线程里调用 ``translate_cues()``，
主线程会被网络等待完全占住 —— 进度条不重绘、按钮点不动、窗口拖动无响应，
用户看到的就是「程序卡死了」，而且连「取消」都点不了。

所以翻译一律在 :class:`TranslateWorker` 里跑，进度与结果通过信号回到主线程。
信号跨线程是排队投递的，槽函数仍在主线程执行，可以安全地碰界面对象。

**断点也在这里写**：进度落盘是磁盘 I/O，天然属于后台线程；顺手还能保证
「每翻完一批就有机会存一次」，取消或崩溃时最多丢一批。界面线程只负责
在开始前读记录、结束后删记录，不参与高频写入。
"""
from __future__ import annotations

from threading import Event
from typing import Sequence

from PySide6.QtCore import QThread, Signal

from app.core.checkpoint import CheckpointWriter
from app.core.subtitle_io import Cue
from app.core.translator import TranslationCancelled, Translator


class TranslateWorker(QThread):
    """在后台线程里跑一次完整翻译。

    ``engine`` 由调用方在**主线程**构造好再传进来：构造只是读配置、不发请求，
    抛配置错误时能同步弹窗；工作线程里只负责真正耗时的网络往返。
    """

    #: (已完成条数, 总条数)——在主线程里更新进度条
    progressed = Signal(int, int)
    #: 正常跑完
    succeeded = Signal()
    #: 用户取消：(已翻译条数, 总条数)
    cancelled = Signal(int, int)
    #: 失败：(异常类名, 消息)
    failed = Signal(str, str)

    def __init__(
        self,
        engine: Translator,
        cues: Sequence[Cue],
        *,
        source_lang: str,
        target_lang: str,
        batch_size: int | None = None,
        writer: CheckpointWriter | None = None,
        skip_translated: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        # 浅拷贝列表，但 Cue 对象是同一批 —— 引擎就地写 cue.translation，
        # 主线程持有的那份列表会同步看到结果。这正是我们要的。
        self._cues = list(cues)
        self._source_lang = source_lang
        self._target_lang = target_lang
        #: 界面上的「上下文窗口」。显式传给引擎，而不是靠在引擎上设属性：
        #: 引擎的 batch_size 是构造时从配置读的，界面改一个数字不该牵扯到重建引擎。
        self._batch_size = None if batch_size is None else int(batch_size)
        self._writer = writer
        self._skip_translated = bool(skip_translated)
        self._stop = Event()
        self._done = 0
        self._total = len(self._cues)

    # ------------------------------------------------------------ 主线程侧

    @property
    def engine(self) -> Translator:
        """跑这次翻译的引擎实例（主线程完成后读它的质量统计）。"""
        return self._engine

    @property
    def done(self) -> int:
        """已翻译条数，取消时用来告诉用户保住了多少。"""
        return self._done

    @property
    def total(self) -> int:
        return self._total

    @property
    def checkpoint_path(self):
        """这次翻译的断点文件路径；None 表示没启用断点。"""
        return self._writer.path if self._writer is not None else None

    @property
    def checkpoint_error(self) -> str:
        """最近一次写断点失败的原因（空串表示一路正常）。"""
        return self._writer.error if self._writer is not None else ""

    def cancel(self) -> None:
        """请求中止。当前批次会跑完（HTTP 请求已在路上），之后不再发新请求。"""
        self._stop.set()

    def is_cancelled(self) -> bool:
        return self._stop.is_set()

    # ------------------------------------------------------------ 工作线程侧

    def _emit_progress(self, done: int, total: int) -> None:
        """在**工作线程**里被引擎回调 —— 只发信号，绝不碰界面对象。"""
        self._done = done
        self.progressed.emit(done, total)
        if self._writer is not None:
            self._writer.maybe(self._cues)

    def _flush_checkpoint(self) -> None:
        """收尾时强制写一次，不受节流限制。"""
        if self._writer is not None:
            self._writer.flush(self._cues)

    def run(self) -> None:  # noqa: D102 - QThread 入口
        try:
            self._engine.translate_cues(
                self._cues,
                source_lang=self._source_lang,
                target_lang=self._target_lang,
                batch_size=self._batch_size,
                progress=self._emit_progress,
                should_stop=self._stop.is_set,
                skip_translated=self._skip_translated,
            )
        except TranslationCancelled:
            self._flush_checkpoint()  # 取消要留住进度，下次接着翻
            self.cancelled.emit(self._done, self._total)
            return
        except Exception as exc:  # 第三方后端什么异常都可能抛，绝不能掀掉线程
            # 失败同样要保住已翻好的部分：502 重试耗尽、中继突然挂掉，
            # 前面十几分钟的成绩不该跟着一起丢。
            self._flush_checkpoint()
            self.failed.emit(type(exc).__name__, str(exc))
            return
        # 成功路径不写断点：整份都翻完了，记录由主线程删掉。
        self.succeeded.emit()
