"""主窗口。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config import (
    APP_NAME,
    DEFAULT_SOURCE_LANG,
    DEFAULT_TARGET_LANG,
    ensure_runtime_dirs,
)
from app.core import subtitle_io
from app.core.subtitle_io import Cue, SubtitleFormatError
from app.core.translator import ENGINES, TranslationError, create_engine


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_runtime_dirs()
        self._cues: List[Cue] = []
        self._source_path: Path | None = None
        self.setWindowTitle(f"{APP_NAME} — 字幕翻译")
        self.resize(1000, 660)
        self._build_ui()

    # ---------- 界面搭建 ----------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        self.open_button = QPushButton("打开字幕…")
        self.open_button.clicked.connect(self._on_open)
        self.path_label = QLabel("未选择文件")

        file_row = QHBoxLayout()
        file_row.addWidget(self.open_button)
        file_row.addWidget(self.path_label, 1)

        self.source_combo = self._language_combo(
            ("auto", "en", "ja", "ko", "zh-CN"), DEFAULT_SOURCE_LANG
        )
        self.target_combo = self._language_combo(
            ("zh-CN", "en", "ja", "ko"), DEFAULT_TARGET_LANG
        )

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(sorted(ENGINES))

        self.translate_button = QPushButton("翻译")
        self.translate_button.clicked.connect(self._on_translate)
        self.translate_button.setEnabled(False)

        self.export_button = QPushButton("导出…")
        self.export_button.clicked.connect(self._on_export)
        self.export_button.setEnabled(False)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("源语言"))
        control_row.addWidget(self.source_combo)
        control_row.addWidget(QLabel("目标语言"))
        control_row.addWidget(self.target_combo)
        control_row.addWidget(QLabel("引擎"))
        control_row.addWidget(self.engine_combo)
        control_row.addStretch(1)
        control_row.addWidget(self.translate_button)
        control_row.addWidget(self.export_button)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("打开字幕文件后，原文与译文会显示在这里…")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        layout = QVBoxLayout(central)
        layout.addLayout(file_row)
        layout.addLayout(control_row)
        layout.addWidget(self.editor, 1)
        layout.addWidget(self.progress)

    @staticmethod
    def _language_combo(values: Tuple[str, ...], current: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)  # 允许手填未列出的语言码
        combo.addItems(values)
        combo.setCurrentText(current)
        return combo

    # ---------- 交互 ----------

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开字幕文件", "", "字幕文件 (*.srt *.vtt);;所有文件 (*)"
        )
        if not path:
            return
        try:
            cues = subtitle_io.parse_file(path)
        except (SubtitleFormatError, OSError) as exc:
            QMessageBox.critical(self, "解析失败", str(exc))
            return

        self._cues = cues
        self._source_path = Path(path)
        self.path_label.setText(f"{path}    （{len(cues)} 条）")
        self.editor.setPlainText(subtitle_io.to_srt(cues))
        self.translate_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.progress.setValue(0)
        self.statusBar().showMessage(f"已载入 {len(cues)} 条字幕")

    def _on_translate(self) -> None:
        if not self._cues:
            return
        engine_name = self.engine_combo.currentText().strip()
        source_lang = self.source_combo.currentText().strip()
        target_lang = self.target_combo.currentText().strip()

        try:
            engine = create_engine(engine_name)
            engine.translate_cues(
                self._cues,
                source_lang=source_lang,
                target_lang=target_lang,
                progress=self._on_progress,
            )
        except TranslationError as exc:
            QMessageBox.critical(self, "翻译失败", str(exc))
            self.statusBar().showMessage("翻译失败")
            return
        except Exception as exc:  # 第三方后端可能抛出任意异常，不能让它掀掉界面
            QMessageBox.critical(self, "翻译失败", f"{type(exc).__name__}: {exc}")
            self.statusBar().showMessage("翻译失败")
            return

        self.editor.setPlainText(subtitle_io.to_srt(self._cues))
        self.statusBar().showMessage(
            f"{engine_name} 翻译完成：{source_lang} → {target_lang}"
        )

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setValue(0 if not total else int(done / total * 100))

    def _on_export(self) -> None:
        if not self._cues:
            return
        stem = self._source_path.stem if self._source_path else "translated"
        suffix = self._source_path.suffix if self._source_path else ".srt"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出字幕",
            f"{stem}.translated{suffix}",
            "SRT (*.srt);;WebVTT (*.vtt)",
        )
        if not path:
            return
        try:
            written = subtitle_io.write_file(path, self._cues)
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.statusBar().showMessage(f"已导出: {written}")
