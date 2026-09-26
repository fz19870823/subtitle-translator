"""模型选择控件与设置对话框的测试。

离屏运行，全程不联网：拉取路径靠替换 ``fetch_models`` 走通。
需要同一解释器里同时有 PySide6 与 pytest，缺任一者整模块跳过。
"""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6", reason="需要 PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import ENV_API_KEY, AppConfig, TranslationConfig  # noqa: E402
from app.core.translator import TranslationError  # noqa: E402
from app.ui import model_selector as ms  # noqa: E402
from app.ui.model_selector import MAX_FETCH_TIMEOUT, ModelSelector  # noqa: E402
from app.ui.settings_dialog import SettingsDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def pump(predicate, timeout: float = 5.0) -> bool:
    """驱动事件循环直到条件成立。

    QThread 的结果是靠信号排进主线程事件队列的，不让事件循环转起来就永远收不到。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def make_config(**overrides) -> AppConfig:
    kwargs = dict(
        base_url="https://old.invalid/v1",
        model="old-model",
        api_key_file="C:/keys/k.txt",
    )
    kwargs.update(overrides)
    return AppConfig(translation=TranslationConfig(**kwargs))


# ------------------------------------------------------------------ 模型选择控件


def test_selector_starts_idle(qapp):
    selector = ModelSelector()
    assert selector.is_fetching() is False
    assert selector.current_model() == ""


def test_selector_keeps_hand_typed_value_across_refresh(qapp):
    selector = ModelSelector()
    selector.set_models(["a", "b"])
    selector.set_current_model("hand-typed")
    assert selector.current_model() == "hand-typed"

    # 重新拉取到另一批模型，用户手输的值不该被悄悄清掉
    selector.set_models(["c", "d"])
    assert selector.current_model() == "hand-typed"
    assert [selector.combo.itemText(i) for i in range(selector.combo.count())] == ["c", "d"]


def test_selector_selects_existing_item(qapp):
    selector = ModelSelector()
    selector.set_models(["a", "b", "c"])
    selector.set_current_model("b")
    assert selector.current_model() == "b"
    assert selector.combo.currentIndex() == 1


def test_selector_fetch_without_provider_warns(qapp):
    selector = ModelSelector()
    selector.fetch()
    assert "未配置" in selector.status.text()


def test_selector_fetch_without_base_url_warns(qapp):
    selector = ModelSelector()
    selector.set_source_provider(lambda: ("", "k", 10))
    selector.fetch()
    assert "API 地址" in selector.status.text()
    assert selector.is_fetching() is False


def test_selector_fetch_without_key_warns(qapp):
    selector = ModelSelector()
    selector.set_source_provider(lambda: ("https://x.invalid/v1", "", 10))
    selector.fetch()
    assert "密钥" in selector.status.text()


def test_selector_populates_from_background_fetch(qapp, monkeypatch):
    selector = ModelSelector()
    selector.set_source_provider(lambda: ("https://x.invalid/v1", "k", 10))
    monkeypatch.setattr(ms, "fetch_models", lambda url, key, timeout=30: ["m1", "m2", "m3"])

    selector.fetch()
    assert pump(lambda: selector.combo.count() == 3), selector.status.text()

    assert [selector.combo.itemText(i) for i in range(3)] == ["m1", "m2", "m3"]
    assert "3" in selector.status.text()
    assert pump(lambda: selector.fetch_button.isEnabled())


def test_selector_reports_fetch_failure_and_recovers_button(qapp, monkeypatch):
    selector = ModelSelector()
    selector.set_source_provider(lambda: ("https://x.invalid/v1", "k", 10))

    def boom(url, key, timeout=30):
        raise TranslationError("拉取模型列表失败: HTTP 401 —— 密钥无效")

    monkeypatch.setattr(ms, "fetch_models", boom)

    selector.fetch()
    assert pump(lambda: "401" in selector.status.text())
    assert pump(lambda: selector.fetch_button.isEnabled()), "失败后按钮必须恢复可点"


def test_selector_set_active_toggles_both_controls(qapp):
    selector = ModelSelector()
    selector.set_active(False)
    assert selector.combo.isEnabled() is False
    assert selector.fetch_button.isEnabled() is False

    selector.set_active(True)
    assert selector.combo.isEnabled() is True
    assert selector.fetch_button.isEnabled() is True


# ------------------------------------------------------------------ 设置对话框


def test_dialog_loads_existing_values(qapp):
    dialog = SettingsDialog(
        make_config(
            batch_size=7,
            timeout=33,
            temperature=0.2,
            style_hint="人名保留原文",
            preserve_line_breaks=False,
        )
    )
    assert dialog.base_url_edit.text() == "https://old.invalid/v1"
    assert dialog.key_file_edit.text() == "C:/keys/k.txt"
    assert dialog.model_selector.current_model() == "old-model"
    assert dialog.batch_spin.value() == 7
    assert dialog.timeout_spin.value() == 33
    assert dialog.temperature_spin.value() == pytest.approx(0.2)
    assert dialog.preserve_check.isChecked() is False
    assert dialog.style_hint_edit.text() == "人名保留原文"


def test_dialog_result_reflects_edits(qapp):
    dialog = SettingsDialog(make_config())
    dialog.base_url_edit.setText("https://new.invalid/v1/")  # 末尾斜杠应被吃掉
    dialog.model_selector.set_current_model("hand-typed-model")
    dialog.batch_spin.setValue(5)
    dialog.key_edit.setText("plain-key")

    cfg = dialog.result_config()
    assert cfg.translation.base_url == "https://new.invalid/v1"
    assert cfg.translation.model == "hand-typed-model"
    assert cfg.translation.batch_size == 5
    assert cfg.translation.api_key == "plain-key"


def test_dialog_result_keeps_unedited_fields(qapp):
    original = make_config(style_hint="保留")
    dialog = SettingsDialog(original)
    dialog.model_selector.set_current_model("changed")
    cfg = dialog.result_config()

    assert cfg.source == original.source
    assert cfg.translation.style_hint == "保留"
    assert cfg.translation.api_key_file == "C:/keys/k.txt"


def test_dialog_key_origin_hint_follows_input(qapp, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    dialog = SettingsDialog(make_config(api_key_file="", api_key=""))
    assert "尚未配置" in dialog.key_origin_label.text()

    dialog.key_edit.setText("plain")
    assert "config:api_key" in dialog.key_origin_label.text()


def test_dialog_flags_env_override(qapp, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "from-env")
    dialog = SettingsDialog(make_config())
    assert "环境变量" in dialog.key_origin_label.text()


def test_dialog_probe_source_uses_typed_key_and_caps_timeout(qapp, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    dialog = SettingsDialog(make_config(timeout=300))
    dialog.key_edit.setText("typed-key")

    base_url, key, timeout = dialog._probe_source()
    assert base_url == "https://old.invalid/v1"
    assert key == "typed-key"
    assert timeout == MAX_FETCH_TIMEOUT, "拉列表的超时要封顶，不能跟翻译超时一样长"


def test_dialog_probe_source_falls_back_to_key_file(qapp, tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-key", encoding="utf-8")

    dialog = SettingsDialog(make_config(api_key_file=str(key_file), api_key=""))
    _, key, _ = dialog._probe_source()
    assert key == "file-key"


def test_dialog_fetch_then_pick_model(qapp, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    dialog = SettingsDialog(make_config(api_key_file="", api_key="typed"))
    monkeypatch.setattr(
        ms, "fetch_models", lambda url, key, timeout=30: ["grok-a", "grok-b"]
    )

    dialog.model_selector.fetch()
    assert pump(lambda: dialog.model_selector.combo.count() == 2)

    dialog.model_selector.set_current_model("grok-b")
    assert dialog.result_config().translation.model == "grok-b"
