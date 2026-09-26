"""应用级常量、路径与本地配置加载。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "Subtitle Translator"
APP_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 本地配置：不入版本库（见 .gitignore 的 config.local.json / *.local.json）。
CONFIG_FILE = PROJECT_ROOT / "config.local.json"

# 本工具原生读写、可无损往返的字幕格式。
SUPPORTED_EXTENSIONS = (".srt", ".vtt")

# 首次启动时界面上的默认语言对。
DEFAULT_SOURCE_LANG = "auto"
DEFAULT_TARGET_LANG = "zh-CN"

# 配置项的环境变量覆盖。优先级：环境变量 > config.local.json > 默认值。
ENV_API_KEY = "SUBTITLE_TRANSLATOR_API_KEY"
ENV_BASE_URL = "SUBTITLE_TRANSLATOR_BASE_URL"
ENV_MODEL = "SUBTITLE_TRANSLATOR_MODEL"


class ConfigError(RuntimeError):
    """配置文件存在但无法使用。"""


@dataclass
class TranslationConfig:
    """翻译后端的连接与模型参数。

    ``api_key`` 与 ``api_key_file`` 二选一：
    - ``api_key`` 直接写明文密钥；
    - ``api_key_file`` 指向存放密钥的文件，运行时读取。

    推荐后者：密钥只存在于它原本的位置，配置里不放副本。
    """

    engine: str = "echo"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    api_key_file: str = ""
    batch_size: int = 20
    timeout: int = 120
    temperature: float = 0.0
    #: 强制保留原文行内换行结构（多行字幕不被合并成一行）。
    preserve_line_breaks: bool = True
    #: 追加到 system prompt 的术语/风格约束，例如 "人名保留原文"。
    style_hint: str = ""
    extra: dict = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """取密钥。环境变量优先，其次 api_key，最后读 api_key_file。"""
        from_env = os.environ.get(ENV_API_KEY, "").strip()
        if from_env:
            return from_env
        if self.api_key.strip():
            return self.api_key.strip()
        if self.api_key_file.strip():
            path = Path(self.api_key_file)
            if not path.exists():
                raise ConfigError(f"api_key_file 不存在: {path}")
            content = path.read_text(encoding="utf-8-sig", errors="replace")
            key = content.strip().splitlines()[0].strip() if content.strip() else ""
            if not key:
                raise ConfigError(f"api_key_file 内容为空: {path}")
            return key
        raise ConfigError(
            "未配置密钥：请在 config.local.json 里设置 api_key 或 api_key_file，"
            f"或设置环境变量 {ENV_API_KEY}"
        )


@dataclass
class AppConfig:
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    #: 配置文件路径；未找到文件时为 None（此时用默认值）。
    source: Path | None = None

    def describe(self) -> dict:
        """用于日志/自检的可打印摘要。密钥一律只报告来源，不报告内容。"""
        t = self.translation
        if os.environ.get(ENV_API_KEY, "").strip():
            key_origin = f"env:{ENV_API_KEY}"
        elif t.api_key.strip():
            key_origin = "config:api_key"
        elif t.api_key_file.strip():
            key_origin = f"file:{t.api_key_file}"
        else:
            key_origin = "(未配置)"
        return {
            "config_file": str(self.source) if self.source else "(默认值，无配置文件)",
            "engine": t.engine,
            "base_url": t.base_url,
            "model": t.model,
            "key_origin": key_origin,
            "batch_size": t.batch_size,
            "timeout": t.timeout,
            "temperature": t.temperature,
            "preserve_line_breaks": t.preserve_line_breaks,
            "style_hint": t.style_hint or "(无)",
        }


def _pick(mapping: dict, *names: str, default=None):
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return default


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载本地配置。

    文件不存在不算错误 —— 返回默认配置，让 UI 仍能启动。
    文件存在但 JSON 非法则抛 :class:`ConfigError`，避免静默用错参数。
    """
    target = Path(path) if path is not None else CONFIG_FILE
    if not target.exists():
        return AppConfig()

    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{target} 不是合法 JSON: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"无法读取 {target}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ConfigError(f"{target} 顶层必须是 JSON 对象")

    # 允许 {"translation": {...}} 或直接把字段放在顶层
    section = payload.get("translation")
    if section is None:
        section = payload
    if not isinstance(section, dict):
        raise ConfigError(f"{target} 的 translation 字段必须是 JSON 对象")

    known = {
        "engine", "base_url", "model", "api_key", "api_key_file",
        "batch_size", "timeout", "temperature", "style_hint",
        "preserve_line_breaks",
    }
    kwargs = {k: v for k, v in section.items() if k in known}
    extra = {k: v for k, v in section.items() if k not in known}

    cfg = TranslationConfig(**kwargs)
    if extra:
        cfg.extra = extra

    # 环境变量覆盖 base_url / model
    env_base = os.environ.get(ENV_BASE_URL, "").strip()
    if env_base:
        cfg.base_url = env_base
    env_model = os.environ.get(ENV_MODEL, "").strip()
    if env_model:
        cfg.model = env_model

    cfg.base_url = (cfg.base_url or "").rstrip("/")
    return AppConfig(translation=cfg, source=target)


def ensure_runtime_dirs() -> None:
    """创建应用会写入的目录，可重复调用。"""
    for directory in (DATA_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
