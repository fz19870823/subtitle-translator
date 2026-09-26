"""在线自检：验证 API 配置与模型配置是否真的可用。

用法：
    python scripts/check_api.py
    python scripts/check_api.py --model grok-4.7
    python scripts/check_api.py --limit 30

做三件事：
1. 打印配置摘要（密钥只报告来源，不显示内容）
2. 拉取 /v1/models，确认配置的 model 在列表中
3. 用一份真的带换行/标签的样例字幕跑通项目自身的翻译代码路径，并校验结果

密钥永远不会出现在输出里。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ConfigError, load_config  # noqa: E402
from app.core import subtitle_io  # noqa: E402
from app.core.engines.openai_compat import OpenAICompatTranslator  # noqa: E402
from app.core.translator import TranslationError, create_engine_for  # noqa: E402

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:03,200
So I told him, <i>that's not how it works</i>.

2
00:00:03,400 --> 00:00:06,000
Third attempt, and
it still fails.

3
00:00:06,200 --> 00:00:08,500


4
00:00:08,700 --> 00:00:11,000
Right. Let's ship it.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="字幕翻译 API/模型自检")
    parser.add_argument("--model", default="", help="覆盖配置里的模型 id")
    parser.add_argument("--limit", type=int, default=0, help="只翻译前 N 条")
    parser.add_argument("--no-chat", action="store_true", help="跳过真实翻译，只查模型列表")
    args = parser.parse_args()

    print("=" * 62)
    print("1) 配置")
    print("=" * 62)
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"配置加载失败: {exc}")
        return 2

    for key, value in config.describe().items():
        print(f"  {key:14}: {value}")

    tcfg = config.translation
    if args.model:
        tcfg.model = args.model
        print(f"  (已用命令行覆盖模型 -> {tcfg.model})")

    try:
        key = tcfg.resolve_api_key()
    except ConfigError as exc:
        print(f"密钥解析失败: {exc}")
        return 2
    print(f"  密钥长度        : {len(key)} 字符（内容不显示）")

    if tcfg.engine != "openai":
        print(f"\n注意：配置里的 engine 是 {tcfg.engine!r}，本次自检固定走 openai 兼容后端。")

    print()
    print("=" * 62)
    print("2) 模型列表")
    print("=" * 62)
    try:
        engine = create_engine_for("openai", config)
    except (TranslationError, ConfigError) as exc:
        print(f"引擎构造失败: {exc}")
        return 2

    assert isinstance(engine, OpenAICompatTranslator)
    try:
        models = engine.list_models()
    except TranslationError as exc:
        print(f"拉取失败: {exc}")
        return 1

    print(f"  服务端返回 {len(models)} 个模型")
    for name in models:
        mark = "  <== 当前配置" if name == tcfg.model else ""
        print(f"    - {name}{mark}")

    if tcfg.model not in models:
        print(f"\n  配置的模型 {tcfg.model!r} 不在列表中！")
        return 1
    print(f"\n  配置的模型 {tcfg.model!r} 在列表中。")

    if args.no_chat:
        print("\n--no-chat：跳过真实翻译。")
        return 0

    print()
    print("=" * 62)
    print("3) 真实字幕翻译（走项目自身代码路径）")
    print("=" * 62)
    cues = subtitle_io.parse_srt(SAMPLE_SRT)
    if args.limit:
        cues = cues[: args.limit]
    print(f"  输入 {len(cues)} 条，其中空文本条目用于验证「空串不送模型」")

    done: list[tuple[int, int]] = []
    try:
        engine.translate_cues(
            cues,
            source_lang="en",
            target_lang="zh-CN",
            progress=lambda d, t: done.append((d, t)),
        )
    except (TranslationError, ConfigError) as exc:
        print(f"  翻译失败: {exc}")
        return 1

    print(f"  进度回调: {done}")
    print(f"  因丢换行被重译的条目数: {engine.line_repair_count}")
    print()
    print("  原文 -> 译文")
    ok = True
    for cue in cues:
        if not cue.text.strip():
            marker = "OK  " if cue.translation == "" else "BAD "
            ok &= cue.translation == ""
            print(f"    [{marker}] (空条目) translation={cue.translation!r}")
            continue
        marker = "OK  " if cue.translation.strip() else "BAD "
        ok &= bool(cue.translation.strip())
        print(f"    [{marker}] {cue.text!r}")
        print(f"           -> {cue.translation!r}")

    # 标签与换行应当保留
    tagged = [c for c in cues if "<i>" in c.text]
    if tagged:
        kept = all("<i>" in c.translation for c in tagged)
        print(f"\n  HTML 标签保留: {kept}  (检查 {len(tagged)} 条)")
        ok &= kept
    multiline = [c for c in cues if "\n" in c.text.strip()]
    if multiline:
        kept_nl = all("\n" in c.translation for c in multiline)
        print(f"  换行保留    : {kept_nl}  (检查 {len(multiline)} 条)")
        ok &= kept_nl

    print()
    print(f"结论: {'全部检查通过' if ok else '存在失败项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
