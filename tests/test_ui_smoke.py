"""界面冒烟测试：离屏构建主窗口，验证各处理器。

需要同一解释器里同时有 PySide6 和 pytest。缺任一者自动跳过。
在只有单边的机器上可改用等价脚本手动验证。
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("PySide6", reason="需要 PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppConfig, TranslationConfig  # noqa: E402
from app.core import subtitle_io  # noqa: E402
from app.core.translator import ENGINES  # noqa: E402
from app.ui import main_window as mw  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

SAMPLE = """1
00:00:01,000 --> 00:00:02,000
hello

2
00:00:03,000 --> 00:00:04,000
world
"""


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp, monkeypatch):
    """构造主窗口，配置替换为离线 echo，避免测试联网。"""
    import app.config as config_module

    offline = AppConfig(translation=TranslationConfig(engine="echo"))
    monkeypatch.setattr(mw, "load_config", lambda *a, **k: offline)
    win = MainWindow()
    yield win
    win.close()


def load_sample(win) -> None:
    """模拟打开文件成功后的状态（绕过 QFileDialog）。"""
    from pathlib import Path

    cues = subtitle_io.parse_srt(SAMPLE)
    win._cues = cues
    win._source_path = Path("demo.srt")
    win.translate_button.setEnabled(True)
    win.export_button.setEnabled(True)


def test_buttons_start_disabled(window):
    assert window.translate_button.isEnabled() is False
    assert window.export_button.isEnabled() is False


def test_engine_combo_lists_registered_engines(window):
    items = [window.engine_combo.itemText(i) for i in range(window.engine_combo.count())]
    assert items == sorted(ENGINES)
    assert "echo" in items and "openai" in items


def test_engine_combo_defaults_to_configured_engine(window):
    assert window.engine_combo.currentText() == "echo"


def test_config_label_shows_summary_without_crashing(window):
    text = window.config_label.text()
    assert "echo" in text
    assert "密钥" in text


def test_translate_fills_editor_and_progress(window):
    load_sample(window)
    window._on_translate()
    assert window.progress.value() == 100
    body = window.editor.toPlainText()
    assert "[echo] hello" in body or "[zh-CN] hello" in body


def test_export_round_trip(window, tmp_path):
    load_sample(window)
    window._on_translate()
    dst = subtitle_io.write_file(tmp_path / "out.srt", window._cues)
    reloaded = subtitle_io.parse_file(dst)
    assert len(reloaded) == 2
    assert reloaded[0].text.strip().startswith("[")


def test_unknown_engine_shows_error_dialog(window, monkeypatch):
    load_sample(window)
    captured: list[tuple[str, str]] = []

    class FakeBox:
        @staticmethod
        def critical(parent, title, text, *a, **k):
            captured.append((title, text))

    # main_window 里是 `from ... import QMessageBox`，必须替换它模块命名空间里的名字
    monkeypatch.setattr(mw, "QMessageBox", FakeBox)

    def boom(name, app_config=None):
        from app.core.translator import TranslationError

        raise TranslationError(f"未知的翻译引擎 {name!r}")

    monkeypatch.setattr(mw, "create_engine_for", boom)
    window._on_translate()

    assert captured and captured[0][0] == "翻译失败"
    assert window.statusBar().currentMessage() == "翻译失败"


def test_engine_combo_is_not_editable(window):
    # 不可编辑是有意为之：引擎必须从注册表里选，不能手打
    assert window.engine_combo.isEditable() is False


# ------------------------------------------------------------------ 模型选择


def test_settings_button_exists(window):
    assert window.settings_button.text() == "设置…"


def test_model_selector_follows_engine_capability(window):
    # 配置里是 echo：没有 API 概念，模型控件应整体禁用
    assert window.engine_combo.currentText() == "echo"
    assert window.model_selector.combo.isEnabled() is False

    window.engine_combo.setCurrentText("openai")
    assert window.model_selector.combo.isEnabled() is True

    window.engine_combo.setCurrentText("echo")
    assert window.model_selector.combo.isEnabled() is False


def test_model_selector_is_editable(window):
    # 必须可编辑：拉不到列表或想用没列出的模型时得能手输
    assert window.model_selector.combo.isEditable() is True


def test_engine_source_reports_configured_endpoint(window):
    window._config.translation.base_url = "https://cfg.invalid/v1"
    window._config.translation.api_key = "cfg-key"

    base_url, key, timeout = window._engine_source()
    assert base_url == "https://cfg.invalid/v1"
    assert key == "cfg-key"
    assert 0 < timeout <= 60


def test_translate_uses_model_picked_in_the_ui(window, monkeypatch):
    """界面换模型要即时生效，不能悄悄沿用配置文件里的旧值。"""
    load_sample(window)
    window.engine_combo.setCurrentText("openai")
    window.model_selector.set_current_model("ui-picked-model")

    seen: dict = {}

    class FakeEngine:
        name = "openai"

        def translate_cues(self, cues, **kwargs):
            for cue in cues:
                cue.translation = "[fake] " + cue.text
            return cues

    def fake_create(name, app_config=None):
        seen["model"] = app_config.translation.model
        return FakeEngine()

    monkeypatch.setattr(mw, "create_engine_for", fake_create)
    window._on_translate()

    assert seen["model"] == "ui-picked-model"
    assert "[fake] hello" in window.editor.toPlainText()


# ------------------------------------------------------------------ 设置保存


def test_settings_dialog_result_is_persisted(window, monkeypatch, tmp_path):
    from dataclasses import replace

    target = tmp_path / "config.local.json"
    # 让保存落到临时目录：绝不能在测试里碰仓库里的真配置
    window._config.source = target

    class FakeDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Accepted

        def result_config(self):
            return replace(
                window._config,
                translation=replace(window._config.translation, model="saved-model"),
            )

    monkeypatch.setattr(mw, "SettingsDialog", FakeDialog)
    window._on_settings()

    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["translation"]["model"] == "saved-model"
    assert window._config.translation.model == "saved-model"
    assert window.model_selector.current_model() == "saved-model"
    assert "已保存" in window.statusBar().currentMessage()


def test_cancelled_settings_dialog_writes_nothing(window, monkeypatch, tmp_path):
    target = tmp_path / "config.local.json"
    window._config.source = target

    class CancelDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(mw, "SettingsDialog", CancelDialog)
    window._on_settings()

    assert not target.exists(), "点了取消就不该写文件"
