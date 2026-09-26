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
