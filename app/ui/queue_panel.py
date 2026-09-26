"""批量翻译队列的面板：显示列表与编辑（增删、排序）。

为什么不把队列逻辑整个塞进主窗口：主窗口已经背着单文件翻译那一整套（断点续传、
未翻译重试、忙碌锁定），再堆一个队列就成了什么都往里放的杂物间。这里只回答
「队列长什么样、用户点了什么」，动作通过信号交出去；真正的翻译编排留在主窗口 ——
只有它知道引擎、断点、导出目录这些上下文。

增删移这些**纯数据结构操作**由面板直接落到队列上（队列本身没有 I/O），
只有「添加文件」（要弹文件对话框、还要解析试读）和「依次翻译」才交给主窗口。
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.queue import DONE, FAILED, PENDING, RUNNING, TranslationQueue

#: 状态 → 列表里显示的中文标签
STATUS_LABELS = {
    PENDING: "待翻译",
    RUNNING: "翻译中",
    DONE: "完成",
    FAILED: "失败",
}

#: 失败项的红色。浅色主题下醒目；深色主题下虽然偏暗，但「失败」两个字仍然在。
FAILURE_COLOR = QColor("#c0392b")

HINT = "尚未添加文件 —— 可一次选择多个字幕，依次翻译并自动保存到输出目录"


class QueuePanel(QWidget):
    """队列的列表与按钮。"""

    #: 用户点了「添加文件…」（主窗口负责弹对话框、解析试读、入队）
    add_requested = Signal()
    #: 用户点了「依次翻译」
    start_requested = Signal()
    #: 队列内容变了（增/删/移），主窗口据此刷新它自己那部分状态
    queue_changed = Signal()

    def __init__(self, queue: TranslationQueue, parent=None) -> None:
        super().__init__(parent)
        self._queue = queue
        self._busy = False
        self._build_ui()
        self.refresh()

    # ---------------------------------------------------------------- 界面

    def _build_ui(self) -> None:
        self.hint_label = QLabel(HINT)
        self.hint_label.setStyleSheet("color: palette(mid);")

        self.list = QListWidget()
        self.list.setMaximumHeight(132)
        # 多选是为了「一次移除好几个」；排序仍只认单选（见 _on_move）。
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self._sync_buttons)

        self.add_button = QPushButton("添加文件…")
        self.add_button.setToolTip("可一次选择多个字幕文件；不支持或读不了的文件会被当场剔除")
        self.add_button.clicked.connect(self.add_requested.emit)

        self.remove_button = QPushButton("移除")
        self.remove_button.clicked.connect(self._on_remove)
        self.up_button = QPushButton("上移")
        self.up_button.setToolTip("调整翻译顺序（只对选中的一项生效）")
        self.up_button.clicked.connect(lambda: self._on_move(-1))
        self.down_button = QPushButton("下移")
        self.down_button.setToolTip("调整翻译顺序（只对选中的一项生效）")
        self.down_button.clicked.connect(lambda: self._on_move(1))
        self.clear_button = QPushButton("清空")
        self.clear_button.clicked.connect(self._on_clear)

        self.start_button = QPushButton("依次翻译")
        self.start_button.clicked.connect(self.start_requested.emit)

        buttons = QHBoxLayout()
        buttons.addWidget(QLabel("队列"))
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.up_button)
        buttons.addWidget(self.down_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        buttons.addWidget(self.start_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.list)
        layout.addLayout(buttons)

    # ---------------------------------------------------------------- 显示

    def refresh(self) -> None:
        """按队列的当前状态重绘列表与按钮。

        整个重建而不是增量改：队列最多几十项，重建的开销可以忽略，而增量修改
        很容易漏掉某一项的状态变化 —— 那种 bug 表现为「列表说完成、实际没翻」，
        比多画几行严重得多。
        """
        selected = {entry.row() for entry in self.list.selectedIndexes()}
        self.list.clear()
        for index, item in enumerate(self._queue):
            entry = QListWidgetItem(self._describe(item))
            entry.setToolTip(f"{item.path}\n译文保存到 {item.output_path}")
            if item.status == FAILED:
                entry.setForeground(FAILURE_COLOR)
            self.list.addItem(entry)
            if index in selected:
                entry.setSelected(True)
        self._sync_buttons()

    def _describe(self, item) -> str:
        text = f"{STATUS_LABELS.get(item.status, item.status)}　{item.name}"
        if item.cue_count:
            text += f"（{item.cue_count} 条）"
        if item.status == FAILED:
            text += f"：{item.one_line_error()}"
        elif item.status == DONE:
            if item.untranslated:
                text += f"（{item.untranslated} 条未翻译）"
            if item.resumed:
                text += f"（沿用上次 {item.resumed} 条）"
            text += f" → {item.output_path.name}"
        return text

    def _sync_buttons(self) -> None:
        counts = self._queue.summary()
        has_items = counts["total"] > 0

        # 队列空的时候整块收起来，只留一句提示和「添加文件…」 —— 用不到的按钮
        # 长期灰着，只会让人去猜它什么时候能用。
        self.hint_label.setVisible(not has_items)
        self.list.setVisible(has_items)
        for button in (
            self.remove_button,
            self.up_button,
            self.down_button,
            self.clear_button,
            self.start_button,
        ):
            button.setVisible(has_items)

        editable = not self._busy
        self.add_button.setEnabled(editable)
        self.remove_button.setEnabled(editable and bool(self.list.selectedIndexes()))
        self.up_button.setEnabled(editable)
        self.down_button.setEnabled(editable)
        self.clear_button.setEnabled(editable)

        if self._busy:
            self.start_button.setEnabled(False)
            self.start_button.setText("队列翻译中…")
        elif counts["total"] == 0:
            self.start_button.setEnabled(False)
            self.start_button.setText("依次翻译")
        elif counts["unfinished"] == 0:
            # 全都成功过了：再点只能是「整份重来」，文案要说清楚。
            self.start_button.setEnabled(True)
            self.start_button.setText("重新翻译全部")
        else:
            self.start_button.setEnabled(True)
            label = "继续队列" if counts["done"] else "依次翻译"
            self.start_button.setText(f"{label}（{counts['unfinished']}）")

        self.start_button.setToolTip(
            f"共 {counts['total']} 个文件、{counts['cue_count']} 条字幕；"
            "译文会自动保存到输出目录。\n"
            "已经翻到一半的文件会接着上次继续，不必重头再来。"
            if has_items
            else ""
        )

    def set_busy(self, busy: bool) -> None:
        """翻译进行中：锁住会改变队列内容的按钮。

        中途改队列会让「正在翻第几项」对不上号，而列表上也看不出任何异常。
        """
        self._busy = bool(busy)
        self._sync_buttons()

    # ---------------------------------------------------------------- 编辑

    def _on_remove(self) -> None:
        rows = sorted(
            {entry.row() for entry in self.list.selectedIndexes()}, reverse=True
        )
        removed = 0
        for row in rows:
            if 0 <= row < len(self._queue) and self._queue.remove(self._queue[row]):
                removed += 1
        if removed:
            self.refresh()
            self.queue_changed.emit()

    def _on_clear(self) -> None:
        # 不做确认弹窗：队列里存的只是文件路径，源文件都还在，重新选一次就行。
        if not len(self._queue):
            return
        self._queue.clear()
        self.refresh()
        self.queue_changed.emit()

    def _on_move(self, delta: int) -> None:
        rows = [entry.row() for entry in self.list.selectedIndexes()]
        # 多选时「上移」的语义不清（按什么顺序、移完谁在前？），索性不响应 ——
        # 猜一个做法比不动更糟，用户会以为顺序已经按他想的排好了。
        if len(rows) != 1:
            return
        new_row = self._queue.move(rows[0], delta)
        if new_row == rows[0]:
            return
        self.refresh()
        entry = self.list.item(new_row)
        if entry is not None:
            self.list.setCurrentItem(entry)
        self.queue_changed.emit()
