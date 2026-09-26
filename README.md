# Subtitle Translator

字幕翻译工具：读取 SRT / WebVTT 字幕，批量翻译后导出。

PySide6 桌面界面 + 可插拔翻译后端。当前仓库为最小可运行骨架：
整条链路（读取 → 翻译 → 导出）已打通并有测试覆盖，翻译后端只挂了
一个离线占位实现 `echo`，真实翻译引擎（LLM API / 本地模型 / 机翻服务）
按下面的方式接入。

## 目录结构

```
subtitle-translator/
├── main.py                    # 入口，创建 QApplication 并拉起主窗口
├── pyproject.toml             # 依赖、打包与 pytest 配置
├── app/
│   ├── config.py              # 常量与运行时目录
│   ├── core/
│   │   ├── subtitle_io.py     # SRT / VTT 解析与序列化
│   │   └── translator.py      # 翻译后端抽象 + 引擎注册表
│   └── ui/
│       └── main_window.py     # 主窗口
└── tests/
    └── test_subtitle_io.py    # 解析器往返测试（不依赖 Qt）
```

## 环境与安装

PySide6 需要带 tkinter/Qt 支持的解释器。本机可用的是系统 Python：

```
C:\Python314\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

用 uv 的话等价于：

```
uv venv
uv pip install -e ".[dev]"
```

> `requires-python = ">=3.11"`，实际请优先用 `C:\Python314\python.exe` 建 venv，
> 托管解释器（`.workbuddy\binaries\python`）不带 GUI 相关依赖。

## 运行

```
.venv\Scripts\python.exe main.py
```

## 测试

```
.venv\Scripts\python.exe -m pytest
```

## 接入一个新的翻译引擎

在 `app/core/translator.py` 里继承 `Translator` 并注册即可，
界面层不需要任何改动 —— 引擎下拉框是从 `ENGINES` 动态生成的。

```python
from app.core.translator import TranslationRequest, Translator, register


@register
class MyLlmTranslator(Translator):
    name = "my-llm"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def translate_batch(self, requests: list[TranslationRequest]) -> list[str]:
        # 返回长度必须与入参一致，否则 translate_cues 会抛 TranslationError
        ...
```

API key 一律从环境变量或本地 `*.local.json` 读取，不要写进仓库
（`.gitignore` 已屏蔽 `.env` / `*.local.json`）。

## 已实现 / 待办

- [x] SRT 解析与导出（保留条号、多行文本、CRLF、BOM）
- [x] WebVTT 解析与导出（跳过 WEBVTT 头与 NOTE / STYLE / REGION 块）
- [x] 编码回退：utf-8-sig → utf-8 → gb18030
- [x] 翻译引擎注册表与批量/进度回调
- [ ] 真实翻译后端（LLM API）
- [ ] ASS / SSA 支持
- [ ] 双语对照导出（原文 + 译文同时保留）
- [ ] 长任务放到 QThread，避免翻译时界面卡死
