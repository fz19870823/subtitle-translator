"""设置对话框：API 连接、模型与翻译参数。

对话框只负责收集输入，不负责落盘 —— 写文件由主窗口调
:func:`app.config.save_config` 统一处理，避免两处都在写配置。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig, ConfigError, TranslationConfig
from app.ui.model_selector import MAX_FETCH_TIMEOUT, ModelSelector

_HINT_STYLE = "color: palette(mid);"


class SettingsDialog(QDialog):
    """编辑 API 地址、密钥、模型与翻译参数。

    通过 :meth:`result_config` 取回改好的配置，调用方决定是否保存。
    """

    def __init__(
        self, config: AppConfig, *, requires_api: bool = True, parent=None
    ) -> None:
        super().__init__(parent)
        self._config = config
        self.setWindowTitle("设置")
        self.setMinimumWidth(660)
        self._build_ui(requires_api)
        self._load_from(config)

    # ------------------------------------------------------------ 界面搭建

    def _build_ui(self, requires_api: bool) -> None:
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://your-endpoint.example.com/v1")

        self.key_file_edit = QLineEdit()
        self.key_file_edit.setPlaceholderText("密钥文件路径（推荐：配置里不留密钥副本）")
        browse_button = QPushButton("浏览…")
        browse_button.clicked.connect(self._on_browse_key_file)
        key_file_row = QHBoxLayout()
        key_file_row.setContentsMargins(0, 0, 0, 0)
        key_file_row.addWidget(self.key_file_edit, 1)
        key_file_row.addWidget(browse_button)
        key_file_widget = QWidget()
        key_file_widget.setLayout(key_file_row)

        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("明文密钥；留空则使用上面的密钥文件")

        self.key_origin_label = QLabel("")
        self.key_origin_label.setWordWrap(True)
        self.key_origin_label.setStyleSheet(_HINT_STYLE)

        self.model_selector = ModelSelector()
        self.model_selector.set_source_provider(self._probe_source)
        self.model_selector.set_active(requires_api)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 200)
        self.batch_spin.setToolTip("单次请求携带多少条字幕")

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 600)
        self.timeout_spin.setSuffix(" 秒")

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(2)

        self.preserve_check = QCheckBox("保留字幕行内换行（多行条目不被并成一行）")

        self.style_hint_edit = QLineEdit()
        self.style_hint_edit.setPlaceholderText("风格/术语约束，例如「人名保留原文」")

        form = QFormLayout()
        form.addRow("API 地址", self.base_url_edit)
        form.addRow("密钥文件", key_file_widget)
        form.addRow("明文密钥", self.key_edit)
        form.addRow("生效密钥", self.key_origin_label)
        form.addRow("模型", self.model_selector)
        form.addRow("批量大小", self.batch_spin)
        form.addRow("超时", self.timeout_spin)
        form.addRow("温度", self.temperature_spin)
        form.addRow("", self.preserve_check)
        form.addRow("风格提示", self.style_hint_edit)

        hint = QLabel(
            "密钥优先级：环境变量 &gt; 明文密钥 &gt; 密钥文件。"
            "明文密钥会以明文写进配置文件，能落成文件的话推荐用密钥文件。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(_HINT_STYLE)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)

        # 密钥来源随输入实时变化，让用户随时知道运行时到底用哪一个
        self.key_edit.textChanged.connect(self._refresh_key_origin)
        self.key_file_edit.textChanged.connect(self._refresh_key_origin)

    def _load_from(self, config: AppConfig) -> None:
        t = config.translation
        self.base_url_edit.setText(t.base_url)
        self.key_file_edit.setText(t.api_key_file)
        self.key_edit.setText(t.api_key)
        self.model_selector.set_current_model(t.model)
        self.batch_spin.setValue(min(200, max(1, int(t.batch_size or 1))))
        self.timeout_spin.setValue(min(600, max(5, int(t.timeout or 120))))
        self.temperature_spin.setValue(float(t.temperature or 0.0))
        self.preserve_check.setChecked(bool(t.preserve_line_breaks))
        self.style_hint_edit.setText(t.style_hint)
        self._refresh_key_origin()

    # ------------------------------------------------------------ 取值

    def _draft_translation(self) -> TranslationConfig:
        """把界面上的当前输入组装成一份配置（不落盘）。"""
        return replace(
            self._config.translation,
            base_url=self.base_url_edit.text().strip().rstrip("/"),
            model=self.model_selector.current_model(),
            api_key=self.key_edit.text().strip(),
            api_key_file=self.key_file_edit.text().strip(),
            batch_size=int(self.batch_spin.value()),
            timeout=int(self.timeout_spin.value()),
            temperature=float(self.temperature_spin.value()),
            preserve_line_breaks=bool(self.preserve_check.isChecked()),
            style_hint=self.style_hint_edit.text().strip(),
        )

    def result_config(self) -> AppConfig:
        """返回编辑后的配置；``source`` 等未涉及的字段原样保留。"""
        return replace(self._config, translation=self._draft_translation())

    # ------------------------------------------------------------ 交互

    def _probe_source(self):
        """给 ModelSelector 用：探测用的 (base_url, api_key, timeout)。

        密钥走和运行时完全相同的解析逻辑，所以「生效密钥」显示的就是实际要用的。
        """
        draft = self._draft_translation()
        try:
            key = draft.resolve_api_key()
        except ConfigError:
            key = ""
        return draft.base_url, key, min(int(draft.timeout or 30), MAX_FETCH_TIMEOUT)

    def _refresh_key_origin(self) -> None:
        origin = self._draft_translation().key_origin()
        if origin.startswith("env:"):
            text = f"环境变量 {origin[4:]} 正在覆盖下面的设置，运行时实际用的是它"
        elif origin == "(未配置)":
            text = "尚未配置密钥"
        else:
            text = f"当前来自 {origin}"
        self.key_origin_label.setText(text)

    def _on_browse_key_file(self) -> None:
        start = self.key_file_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "选择密钥文件", start, "文本文件 (*.txt);;所有文件 (*)"
        )
        if path:
            self.key_file_edit.setText(path)
