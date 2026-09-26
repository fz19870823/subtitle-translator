"""应用级常量与路径。"""
from __future__ import annotations

from pathlib import Path

APP_NAME = "Subtitle Translator"
APP_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 本工具原生读写、可无损往返的字幕格式。
SUPPORTED_EXTENSIONS = (".srt", ".vtt")

# 首次启动时界面上的默认语言对。
DEFAULT_SOURCE_LANG = "auto"
DEFAULT_TARGET_LANG = "zh-CN"


def ensure_runtime_dirs() -> None:
    """创建应用会写入的目录，可重复调用。"""
    for directory in (DATA_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
