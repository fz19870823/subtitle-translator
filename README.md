# Subtitle Translator

字幕翻译工具：读取 SRT / WebVTT 字幕，批量翻译后导出。

PySide6 桌面界面 + 可插拔翻译后端。整条链路（读取 → 翻译 → 导出）已打通并有测试覆盖。
已内置一个 **OpenAI 兼容**翻译后端，可对接任何暴露
`POST {base_url}/chat/completions` 的服务（OpenAI、各类中转/聚合网关、本地推理服务）。

## 目录结构

```
subtitle-translator/
├── main.py                         # 入口，创建 QApplication 并拉起主窗口
├── pyproject.toml                  # 依赖、打包与 pytest 配置
├── config.example.json             # 配置模板（复制为 config.local.json）
├── config.local.json               # 本地配置，已被 .gitignore 屏蔽
├── scripts/
│   └── check_api.py                # 在线自检：API / 模型配置 + 真实翻译冒烟
├── app/
│   ├── config.py                   # 常量、运行时目录、本地配置加载
│   ├── core/
│   │   ├── subtitle_io.py          # SRT / VTT 解析与序列化
│   │   ├── translator.py           # 翻译后端抽象 + 引擎注册表
│   │   └── engines/
│   │       └── openai_compat.py    # OpenAI 兼容后端
│   └── ui/
│       ├── main_window.py          # 主窗口
│       ├── model_selector.py       # 模型下拉：可拉取列表、可手输
│       └── settings_dialog.py      # 设置对话框：API / 密钥 / 模型 / 参数
└── tests/
    ├── test_subtitle_io.py         # 解析器往返测试
    ├── test_config.py              # 配置加载、保存与密钥脱敏
    ├── test_translator.py          # 注册表、批量与进度回调
    ├── test_openai_compat.py       # 协议解析、换行处理、模型列表（不联网）
    ├── test_ui_smoke.py            # 主窗口冒烟（需要 PySide6，否则跳过）
    └── test_ui_settings.py         # 模型控件与设置对话框（同上）
```

## 环境与安装

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

用 uv 的话等价于：

```
uv venv
uv pip install -e ".[dev]"
```

## 配置

API 地址、密钥、模型**都能在界面里改**，不必手写 JSON。两种方式落到同一个
`config.local.json`，改哪个都行。

### 方式一：界面里配置（推荐）

主窗口点「设置…」：

| 项 | 说明 |
| --- | --- |
| API 地址 | 填到 `/v1` 为止，代码自行拼 `/chat/completions` 与 `/models` |
| 密钥文件 | 填密钥文件的路径（推荐，配置里不留副本），带「浏览…」 |
| 明文密钥 | 直接写密钥；留空则回退到密钥文件 |
| 生效密钥 | 实时显示运行时**实际会用哪个**。环境变量优先级最高，被它覆盖时会明确提示 |
| 模型 | 点「拉取模型」从服务端 `/models` 取回列表下拉选择；**也可以直接手输**列表里没有的 id |
| 其余 | 批量大小 / 超时 / 温度 / 保留换行 / 风格提示 |

几个实现上的取舍：

- **拉取模型在网络线程里跑**，后台 QThread + 信号回主线程。点一下按钮窗口不该假死。
- **拉取超时封顶 60 秒**，不复用翻译的 `timeout`（那个默认 120 秒，等列表等不起）。
- 保存时先把原文件备份成 `config.local.json.bak`，再写临时文件原子替换，避免写坏配置。
- 主窗口上的模型下拉是**临时**切换，当次运行生效；要持久化就进「设置…」保存。

### 方式二：直接编辑配置文件

复制模板后填写：

```
copy config.example.json config.local.json
```

```jsonc
{
  "translation": {
    "engine": "openai",
    "base_url": "https://your-endpoint.example.com/v1",
    "model": "your-model-id",
    // 二选一：api_key_file 指向密钥文件（推荐，配置里不留密钥副本）
    "api_key_file": "",
    "api_key": "",
    "batch_size": 20,
    "timeout": 120,
    "temperature": 0,
    "preserve_line_breaks": true,
    "style_hint": ""
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `engine` | 注册表里的引擎名；`openai` 为 OpenAI 兼容后端，`echo` 为离线占位 |
| `base_url` | 到 `/v1` 为止，代码会自行拼 `/chat/completions` 与 `/models` |
| `model` | 模型 id，必须存在于服务端 `/v1/models` 返回的列表中 |
| `api_key_file` | 密钥文件路径，运行时读取；**推荐**，避免密钥出现副本 |
| `api_key` | 直接写明文密钥（优先级低于环境变量、高于 `api_key_file`） |
| `batch_size` | 单次请求携带多少条字幕 |
| `preserve_line_breaks` | 保留字幕行内换行结构，多行条目不被合并成一行（默认 `true`） |
| `style_hint` | 追加到 system prompt 的风格/术语约束，如 `"人名保留原文"` |

环境变量优先级最高：`SUBTITLE_TRANSLATOR_API_KEY` / `SUBTITLE_TRANSLATOR_BASE_URL` /
`SUBTITLE_TRANSLATOR_MODEL`。`config.local.json` 已被 `.gitignore` 屏蔽。

### 密钥安全

密钥只在运行时从 `api_key_file` 读取，**不复制、不落库、不进日志**：

- `TranslationConfig.resolve_api_key()` 每次现读；
- `AppConfig.describe()` 只报告密钥来源（`env:` / `file:` / `config:`），从不输出内容；
- 后端的 `__repr__` 输出 `api_key=<hidden>`，避免异常回溯带出密钥。

## 运行

```
.venv\Scripts\python.exe main.py
```

## 自检（API 配置 / 模型配置）

```
.venv\Scripts\python.exe scripts\check_api.py
.venv\Scripts\python.exe scripts\check_api.py --model <other-model>
.venv\Scripts\python.exe scripts\check_api.py --no-chat      # 只查模型列表
```

依次做三件事：打印配置摘要（密钥只报来源）、拉 `/v1/models` 并确认配置的模型存在、
用样例字幕跑通项目自身的翻译代码路径并校验行数/标签/换行。密钥不会出现在输出里。

## 测试

```
.venv\Scripts\python.exe -m pytest
```

## 接入一个新的翻译引擎

继承 `Translator` 并注册即可，界面层不改动 —— 引擎下拉框从 `ENGINES` 动态生成。
若引擎需要外部参数，再加一个 `from_config(cfg)` 类方法，界面与 `create_engine_for()`
会自动走它。

需要 API 地址 / 密钥 / 模型的引擎，把类属性 `requires_api` 设为 `True`，主界面据此启用
模型选择控件；默认 `False`（`echo` 这类离线引擎就用不到）。

```python
from app.core.translator import TranslationRequest, Translator, register


@register
class MyLlmTranslator(Translator):
    name = "my-llm"

    def translate_batch(self, requests: list[TranslationRequest]) -> list[str]:
        # 返回长度必须与入参一致，否则 translate_cues 会抛 TranslationError
        ...
```

## 翻译协议说明

批量翻译时把整批字幕作为一个 JSON 数组送出，要求模型返回等长 JSON 数组。
实测有两个坑，代码里都做了处理：

1. **换行会被模型吃掉。** 只让模型「保留换行符」不可靠，多行字幕常被合并成一行。
   现在送出前把换行替换成可见记号 `⏎`（U+23CE），收回后再还原；若模型仍丢记号，
   则对**该条**按行重译（`line_repair_count` 可观测触发次数）。
2. **批量协议可能被破坏。** 模型加了说明或数量不符时，整批退回逐条翻译，宁可慢也不丢内容。

## 已实现 / 待办

- [x] SRT 解析与导出（保留条号、多行文本、CRLF、BOM）
- [x] WebVTT 解析与导出（跳过 WEBVTT 头与 NOTE / STYLE / REGION 块）
- [x] 编码回退：utf-8-sig → utf-8 → gb18030
- [x] 翻译引擎注册表与批量/进度回调
- [x] 本地配置加载 + 密钥脱敏
- [x] 界面内配置 API 地址 / 密钥，模型可拉取列表选择或手输
- [x] 配置保存（覆盖前备份、临时文件原子替换）
- [x] OpenAI 兼容后端（批量 JSON 协议、换行保真、逐条回退）
- [x] 在线自检脚本 `scripts/check_api.py`
- [ ] ASS / SSA 支持
- [ ] 双语对照导出（原文 + 译文同时保留）
- [ ] 翻译本身放到 QThread，避免长字幕翻译时界面卡死
- [ ] 并发请求以提升长字幕的翻译速度
