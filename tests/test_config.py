"""本地配置加载与密钥解析的测试。不访问网络。"""
from __future__ import annotations

import json

import pytest

from app.config import (
    CONFIG_FILE,
    ENV_API_KEY,
    AppConfig,
    ConfigError,
    TranslationConfig,
    load_config,
    save_config,
)


def write_config(tmp_path, payload: dict):
    path = tmp_path / "config.local.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_missing_file_falls_back_to_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.json")
    assert isinstance(cfg, AppConfig)
    assert cfg.source is None
    assert cfg.translation.engine == "echo"
    assert cfg.translation.batch_size == 20
    assert cfg.translation.preserve_line_breaks is True


def test_nested_and_flat_layouts_both_work(tmp_path):
    nested = load_config(
        write_config(tmp_path, {"translation": {"model": "m-nested"}})
    )
    assert nested.translation.model == "m-nested"

    flat = load_config(write_config(tmp_path, {"model": "m-flat"}))
    assert flat.translation.model == "m-flat"


def test_unknown_keys_land_in_extra(tmp_path):
    cfg = load_config(
        write_config(tmp_path, {"translation": {"model": "m", "future_option": 42}})
    )
    assert cfg.translation.extra == {"future_option": 42}
    assert cfg.translation.model == "m"


def test_default_config_file_is_project_local():
    # 防止有人把 CONFIG_FILE 指向仓库外的路径导致配置漂移
    assert CONFIG_FILE.name == "config.local.json"
    assert CONFIG_FILE.parent.name == "subtitle-translator"


def test_invalid_json_raises_config_error(tmp_path):
    path = tmp_path / "config.local.json"
    path.write_text("{ not json ", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_non_object_toplevel_raises(tmp_path):
    path = tmp_path / "config.local.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_base_url_trailing_slash_is_stripped(tmp_path):
    cfg = load_config(
        write_config(tmp_path, {"translation": {"base_url": "https://x.example/v1/"}})
    )
    assert cfg.translation.base_url == "https://x.example/v1"


# ------------------------------------------------------------------ 密钥解析

def test_resolve_api_key_prefers_plain_key_over_file(tmp_path, monkeypatch):
    key_file = tmp_path / "key.txt"
    key_file.write_text("from-file\n", encoding="utf-8")
    monkeypatch.delenv(ENV_API_KEY, raising=False)

    cfg = TranslationConfig(api_key="from-config", api_key_file=str(key_file))
    assert cfg.resolve_api_key() == "from-config"


def test_resolve_api_key_reads_file_with_bom_and_blank_lines(tmp_path, monkeypatch):
    key_file = tmp_path / "key.txt"
    key_file.write_text("\ufeff\n\n  the-real-key  \n", encoding="utf-8")
    monkeypatch.delenv(ENV_API_KEY, raising=False)

    cfg = TranslationConfig(api_key_file=str(key_file))
    assert cfg.resolve_api_key() == "the-real-key"


def test_env_var_overrides_everything(tmp_path, monkeypatch):
    key_file = tmp_path / "key.txt"
    key_file.write_text("from-file", encoding="utf-8")
    monkeypatch.setenv(ENV_API_KEY, "from-env")

    cfg = TranslationConfig(api_key="from-config", api_key_file=str(key_file))
    assert cfg.resolve_api_key() == "from-env"


def test_missing_key_file_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    cfg = TranslationConfig(api_key_file=str(tmp_path / "absent.txt"))
    with pytest.raises(ConfigError):
        cfg.resolve_api_key()


def test_no_key_configured_raises(monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(ConfigError):
        TranslationConfig().resolve_api_key()


# ------------------------------------------------------------------ 摘要脱敏

def test_describe_never_leaks_the_key(tmp_path, monkeypatch):
    secret = "g2a_SUPERSECRETVALUE1234567890"
    monkeypatch.setenv(ENV_API_KEY, secret)

    cfg = load_config(
        write_config(tmp_path, {"translation": {"api_key": secret, "model": "m"}})
    )
    dump = json.dumps(cfg.describe(), ensure_ascii=False)
    assert secret not in dump
    assert "env:" in dump


def test_describe_reports_key_file_origin_without_content(tmp_path, monkeypatch):
    secret = "g2a_ANOTHERSECRET9988776655443322"
    key_file = tmp_path / "key.txt"
    key_file.write_text(secret, encoding="utf-8")
    monkeypatch.delenv(ENV_API_KEY, raising=False)

    cfg = load_config(write_config(tmp_path, {"translation": {"api_key_file": str(key_file)}}))
    info = cfg.describe()
    assert info["key_origin"].startswith("file:")
    assert secret not in json.dumps(info, ensure_ascii=False)


# ------------------------------------------------------------------ 密钥来源


def test_key_origin_reports_source(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    assert TranslationConfig().key_origin() == "(未配置)"
    assert TranslationConfig(api_key="k").key_origin() == "config:api_key"
    assert TranslationConfig(api_key_file="C:/k/k.txt").key_origin() == "file:C:/k/k.txt"
    assert TranslationConfig(api_key="k", api_key_file="C:/k/k.txt").key_origin() == "config:api_key"

    monkeypatch.setenv(ENV_API_KEY, "env-key")
    assert TranslationConfig(api_key="k").key_origin() == f"env:{ENV_API_KEY}"


# ------------------------------------------------------------------ 保存


def test_to_dict_covers_every_known_field():
    t = TranslationConfig(
        engine="openai",
        base_url="https://x/v1",
        model="m",
        api_key="k",
        api_key_file="f",
        batch_size=3,
        timeout=9,
        temperature=0.5,
        preserve_line_breaks=False,
        stream=False,
        style_hint="人名保留原文",
    )
    dumped = t.to_dict()
    assert set(dumped) == {
        "engine", "base_url", "model", "api_key", "api_key_file",
        "batch_size", "timeout", "temperature", "preserve_line_breaks",
        "stream", "style_hint",
    }
    assert dumped["preserve_line_breaks"] is False
    assert dumped["stream"] is False
    assert dumped["style_hint"] == "人名保留原文"


def test_save_config_round_trip(tmp_path):
    path = tmp_path / "config.local.json"
    cfg = load_config(path)  # 文件不存在 -> 默认值
    assert cfg.source is None

    cfg.translation.base_url = "https://edited.invalid/v1"
    cfg.translation.model = "edited-model"
    cfg.translation.api_key_file = "C:/keys/k.txt"

    written = save_config(cfg, path)
    assert written == path

    again = load_config(path)
    assert again.translation.base_url == "https://edited.invalid/v1"
    assert again.translation.model == "edited-model"
    assert again.translation.api_key_file == "C:/keys/k.txt"


def test_save_config_writes_back_to_recorded_source(tmp_path):
    path = write_config(tmp_path, {"translation": {"model": "before"}})
    cfg = load_config(path)
    cfg.translation.model = "after"
    # 不显式传 path：应当写回 load 时记住的那个文件
    assert save_config(cfg) == path
    assert load_config(path).translation.model == "after"


def test_save_config_backs_up_the_previous_file(tmp_path):
    path = write_config(tmp_path, {"translation": {"model": "old"}})
    cfg = load_config(path)
    cfg.translation.model = "new"
    save_config(cfg, path)

    backup = path.with_name(path.name + ".bak")
    assert backup.exists(), "覆盖前必须留一份备份"
    assert json.loads(backup.read_text(encoding="utf-8"))["translation"]["model"] == "old"
    assert load_config(path).translation.model == "new"


def test_save_config_leaves_no_temp_file(tmp_path):
    path = tmp_path / "config.local.json"
    save_config(load_config(path), path)
    assert not (tmp_path / "config.local.json.tmp").exists()
    assert path.exists()


def test_save_config_preserves_unknown_keys(tmp_path):
    path = write_config(
        tmp_path, {"translation": {"model": "m", "future_option": 42, "nested": {"a": 1}}}
    )
    cfg = load_config(path)
    save_config(cfg, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["translation"]["future_option"] == 42
    assert payload["translation"]["nested"] == {"a": 1}
    assert payload["translation"]["model"] == "m"


def test_save_config_output_is_utf8_without_bom(tmp_path):
    path = tmp_path / "config.local.json"
    cfg = load_config(path)
    cfg.translation.style_hint = "人名保留原文"
    save_config(cfg, path)

    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert "人名保留原文" in raw.decode("utf-8")
