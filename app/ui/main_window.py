"""主窗口。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
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
    AppConfig,
    ConfigError,
    ensure_runtime_dirs,
    load_config,
    save_config,
)
from app.core import subtitle_io
from app.core.subtitle_io import Cue, SubtitleFormatError
from app.core.translator import ENGINES, TranslationError, create_engine_for
from app.ui.model_selector import MAX_FETCH_TIMEOUT, ModelSelector
from app.ui.settings_dialog import SettingsDialog
from app.ui.translate_worker import TranslateWorker

#: 关窗时最多等后台翻译线程多久收尾（毫秒）。超过就放弃等待，避免窗口卡住。
CLOSE_WAIT_MS = 8000


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_runtime_dirs()
        self._cues: List[Cue] = []
        self._source_path: Path | None = None
        #: 正在跑的后台翻译线程；None 表示当前空闲
        self._worker: TranslateWorker | None = None
        #: 本次翻译的 (引擎名, 源语言, 目标语言)，成功后写进状态栏
        self._translation_meta: Tuple[str, str, str] = ("", "", "")

        # 配置文件缺失不算错误（退回默认值），但 JSON 非法要明确告诉用户。
        self._config_error = ""
        try:
            self._config = load_config()
        except ConfigError as exc:
            self._config = AppConfig()
            self._config_error = str(exc)

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

        self.model_selector = ModelSelector()
        self.model_selector.set_source_provider(self._engine_source)
        self.model_selector.set_current_model(self._config.translation.model)

        self.settings_button = QPushButton("设置…")
        self.settings_button.clicked.connect(self._on_settings)

        self.translate_button = QPushButton("翻译")
        self.translate_button.clicked.connect(self._on_translate)
        self.translate_button.setEnabled(False)

        # 翻译是分钟级的（上千条字幕要发几十次请求），必须给用户一条退路。
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self._on_cancel)
        self.cancel_button.setToolTip("中止翻译（当前这批请求返回后停止）")
        self.cancel_button.setVisible(False)

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
        control_row.addWidget(self.settings_button)
        control_row.addStretch(1)
        control_row.addWidget(self.translate_button)
        control_row.addWidget(self.cancel_button)
        control_row.addWidget(self.export_button)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型"))
        model_row.addWidget(self.model_selector, 1)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("打开字幕文件后，原文与译文会显示在这里…")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.config_label = QLabel(self._config_summary())
        self.config_label.setStyleSheet("color: palette(mid);")

        layout = QVBoxLayout(central)
        layout.addLayout(file_row)
        layout.addLayout(control_row)
        layout.addLayout(model_row)
        layout.addWidget(self.config_label)
        layout.addWidget(self.editor, 1)
        layout.addWidget(self.progress)

        # 连接放在最后：引擎一变化就要去改 model_selector，得先保证它已经建好。
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        configured_engine = self._config.translation.engine
        if configured_engine in ENGINES:
            self.engine_combo.setCurrentText(configured_engine)
        # 必须显式刷一次：setCurrentText 设成和当前相同的值**不会发信号**，
        # 光靠信号会让初始状态错掉（例如配置是 echo，模型控件却是可用的）。
        self._on_engine_changed(self.engine_combo.currentText())

    def _config_summary(self) -> str:
        """配置摘要。只报告密钥来源，绝不显示密钥内容。"""
        if self._config_error:
            return f"配置有误：{self._config_error}"
        info = self._config.describe()
        return (
            f"配置 {info['config_file']}　|　引擎 {info['engine']}　|　"
            f"模型 {info['model'] or '(未设置)'}　|　"
            f"地址 {info['base_url'] or '(未设置)'}　|　密钥 {info['key_origin']}"
        )

    @staticmethod
    def _language_combo(values: Tuple[str, ...], current: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)  # 允许手填未列出的语言码
        combo.addItems(values)
        combo.setCurrentText(current)
        return combo

    # ---------- 连接参数 ----------

    def _engine_source(self) -> Tuple[str, str, int]:
        """给模型选择器用的 (base_url, api_key, timeout)。"""
        t = self._config.translation
        try:
            key = t.resolve_api_key()
        except ConfigError:
            key = ""
        return t.base_url, key, min(int(t.timeout or 30), MAX_FETCH_TIMEOUT)

    def _refresh_config_label(self) -> None:
        self.config_label.setText(self._config_summary())

    def _on_engine_changed(self, engine_name: str) -> None:
        """按引擎能力决定模型控件是否可用（echo 用不到地址和模型）。"""
        engine = ENGINES.get(engine_name)
        self.model_selector.set_active(bool(getattr(engine, "requires_api", False)))
        self._refresh_config_label()

    def _on_settings(self) -> None:
        engine_name = self.engine_combo.currentText().strip()
        dialog = SettingsDialog(
            self._config,
            requires_api=bool(getattr(ENGINES.get(engine_name), "requires_api", False)),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        updated = dialog.result_config()
        # 引擎是主界面上的选择，对话框不管它
        updated.translation.engine = engine_name
        try:
            saved = save_config(updated)
        except OSError as exc:
            QMessageBox.critical(self, "保存配置失败", str(exc))
            return

        updated.source = saved
        self._config = updated
        self._config_error = ""
        self.model_selector.set_current_model(self._config.translation.model)
        self._refresh_config_label()
        self.statusBar().showMessage(f"配置已保存：{saved}")

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
        """启动翻译。**立即返回** —— 耗时的部分在后台线程里跑。"""
        if not self._cues or self._worker is not None:
            return
        engine_name = self.engine_combo.currentText().strip()
        source_lang = self.source_combo.currentText().strip()
        target_lang = self.target_combo.currentText().strip()

        # 界面上临时选的模型要即时生效，否则会悄悄沿用配置文件里的旧值。
        # 想持久化就走「设置…」。
        if getattr(ENGINES.get(engine_name), "requires_api", False):
            self._config.translation.model = self.model_selector.current_model()

        # 引擎构造留在主线程：它只读配置、不发网络请求，出错时能同步弹窗。
        try:
            engine = create_engine_for(engine_name, self._config)
        except (TranslationError, ConfigError) as exc:
            QMessageBox.critical(self, "翻译失败", str(exc))
            self.statusBar().showMessage("翻译失败")
            return
        except Exception as exc:  # 第三方后端可能抛出任意异常，不能让它掀掉界面
            QMessageBox.critical(self, "翻译失败", f"{type(exc).__name__}: {exc}")
            self.statusBar().showMessage("翻译失败")
            return

        # 真正耗时的部分必须进后台线程：上千条字幕要发几十次请求、跑好几分钟，
        # 放在主线程里窗口会整个冻住（进度条不动、取消都点不了）。
        self._translation_meta = (engine_name, source_lang, target_lang)
        worker = TranslateWorker(
            engine,
            self._cues,
            source_lang=source_lang,
            target_lang=target_lang,
            parent=self,
        )
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(self._on_translate_succeeded)
        worker.cancelled.connect(self._on_translate_cancelled)
        worker.failed.connect(self._on_translate_failed)
        self._worker = worker  # 保留引用：线程被回收会连带丢掉信号连接
        self._set_busy(True)
        self.progress.setValue(0)
        self.statusBar().showMessage(f"翻译中…（共 {len(self._cues)} 条）")
        worker.start()

    # ---------- 后台线程的回调（信号跨线程排队投递，槽仍在主线程执行） ----------

    def _on_translate_succeeded(self) -> None:
        engine = self._worker.engine if self._worker is not None else None
        self._finish_translation()

        self.editor.setPlainText(subtitle_io.to_srt(self._cues))
        engine_name, source_lang, target_lang = self._translation_meta
        notes = engine.quality_notes() if engine is not None else []
        summary = f"{engine_name} 翻译完成：{source_lang} → {target_lang}"
        if notes:
            summary += "　|　" + "；".join(notes)
        self.statusBar().showMessage(summary)

        # 回抄是**静默**故障：不报告就没人会发现手里那份字幕根本没翻。
        # 救回来了在状态栏记账；没救回来的必须弹出来。
        if engine is not None and engine.untranslated_count:
            QMessageBox.warning(
                self,
                "部分字幕可能没有翻译",
                f"有 {engine.untranslated_count} 条字幕与原文完全相同，"
                "整批重发和逐条重译都没能让模型改写它们。\n\n"
                "请先在译文区核对这几条再导出；换成别的模型通常可以解决。",
            )

    def _on_translate_cancelled(self, done: int, total: int) -> None:
        self._finish_translation()
        # 已翻好的部分留着：用户能核对或导出半成品，不必从头再来。
        self.editor.setPlainText(subtitle_io.to_srt(self._cues))
        self.statusBar().showMessage(f"已取消：{done}/{total} 条已翻译")

    def _on_translate_failed(self, kind: str, message: str) -> None:
        self._finish_translation()
        text = message if kind == "TranslationError" else f"{kind}: {message}"
        QMessageBox.critical(self, "翻译失败", text)
        self.statusBar().showMessage("翻译失败")

    def _on_cancel(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.statusBar().showMessage("正在取消…（等当前这批请求返回）")

    def _finish_translation(self) -> None:
        """收尾：回收线程、恢复控件。成功 / 取消 / 失败三条路径都要走这里。"""
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait(CLOSE_WAIT_MS)  # 信号已到主线程，线程此时基本已退出
            worker.deleteLater()  # worker 是窗口的子对象，不显式回收会随翻译次数累积
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        """翻译期间锁住会改变任务输入或结果的控件。

        不锁的话用户能在翻译跑着时换字幕文件 —— 后台线程还在往旧对象里写译文，
        界面最终显示一份与当前文件对不上的结果，而且看不出哪里错了。
        """
        busy = bool(busy)
        # 取消按钮只在忙碌时出现；每次重新进入忙碌都要把它的禁用状态复位，
        # 否则「取消」过一次之后，下一轮翻译的取消按钮点不动。
        self.cancel_button.setVisible(busy)
        if busy:
            self.cancel_button.setEnabled(True)
        self.open_button.setEnabled(not busy)
        self.settings_button.setEnabled(not busy)
        self.engine_combo.setEnabled(not busy)
        self.source_combo.setEnabled(not busy)
        self.target_combo.setEnabled(not busy)
        self.model_selector.setEnabled(not busy)
        self.translate_button.setEnabled(not busy and bool(self._cues))
        self.export_button.setEnabled(not busy and bool(self._cues))

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setValue(0 if not total else int(done / total * 100))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 的命名约定
        """关窗时必须先送走后台线程。

        窗口对象一旦被回收，还在跑的 QThread 会踩到野指针（Qt 会直接报
        "Destroyed while thread is still running" 并可能崩进程）。
        """
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            if not worker.wait(CLOSE_WAIT_MS):
                # 当前这批 HTTP 请求还挂着（中继无响应时可等到超时）。宁可多等一会儿，
                # 也不能让线程在运行中被销毁。
                self.statusBar().showMessage("等待当前请求结束…")
                worker.wait()
        super().closeEvent(event)

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
