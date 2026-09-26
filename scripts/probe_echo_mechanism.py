"""判别实验：回抄到底是「模型不会翻译」还是「请求被转给了不干活的后端」。

设计
----
同一模型、同一批 20 条日文、同一份 JSON 数组协议，只改任务或指令位置：

- ``translate``   真实翻译任务（指令走 system 角色，引擎现状）
- ``marker``      机械任务：每个元素原样复制并加 ``[1]`` 前缀。
                 不需要任何语言能力 —— 连它都不做，才说明「指令没被执行」。
- ``userinstr``   翻译任务，但把完整指令搬进 user 消息、**不发 system 角色**
                 （用来验证「弱后端丢掉 system 提示词」这一假设）
- ``b1``          翻译任务，每批只发 1 条（验证是否与「一整个数组」的形状相关）

度量
----
- ``kana``      回复里的假名数。日译中成功 → 假名≈0；原样回抄 → 假名一大堆。
- ``same``      逐条与原文完全相同（仅在 JSON 能解析时统计）
- ``parse_ok``  回复是否为合法 JSON 数组；解析失败单独记，**不**算作回抄
- ``raw_echo``  回复里是否出现了我们拼进去的 ``源语言:`` 前缀 —— 原始请求体被回吐的铁证

用法::

    python scripts/probe_echo_mechanism.py --models grok-chat-fast --tasks translate,marker,userinstr,b1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config  # noqa: E402
from app.core.engines.openai_compat import OpenAICompatTranslator  # noqa: E402
from app.core.translator import is_translatable  # noqa: E402

# 与 diag_echo.py 同一份语料，保证跨实验可比。
JA_CORPUS = [
    "何度も言ったはずだ。",
    "そんなの無理だよ。",
    "お前、何を考えてるんだ？",
    "ごめん、遅れた。",
    "電車が止まってて。",
    "静かにして。",
    "ここから先は立入禁止だ。",
    "分かった。任せるよ。",
    "本当にいいのか？",
    "後悔するぞ。",
    "もう決めたことだ。",
    "あ、そうだ。",
    "田中さんに伝えといて。",
    "彼女、明日来るの？",
    "たぶん来ないと思う。",
    "ふざけるな！",
    "行こう。",
    "待って、鍵かけた？",
    "おわり。",
    "それでいいのかい？",
]

MARKER_PROMPT = (
    "你是一名文本批处理助手。用户会给你一个 JSON 数组。\n"
    "请返回一个同样长度的 JSON 数组：每个元素原样复制，并在最前面加上 [1] 。\n"
    "只输出 JSON 数组，不要输出任何解释、编号或代码块。"
)

#: 平假名 U+3040-309F + 片假名 U+30A0-30FF（含长音符 U+30FC 已在片假名区内）
KANA_RE = re.compile(r"[\u3040-\u30ff]")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")

TARGET_LANG = "zh-CN"


def build_payload(texts: list[str], source_lang: str) -> str:
    """完全复刻引擎 _request_batch 里的拼装方式。"""
    content = json.dumps(list(texts), ensure_ascii=False)
    if source_lang and source_lang != "auto":
        content = f"源语言: {source_lang}\n{content}"
    return content


def one_request(
    engine: OpenAICompatTranslator,
    system: str | None,
    instruction: str,
    texts: list[str],
    source_lang: str,
    *,
    marker: bool = False,
) -> dict:
    """发一次请求，记录原始响应的形态。``system=None`` 表示不发送 system 角色。"""
    payload_text = build_payload(texts, source_lang)
    if system is None:
        # 指令搬进 user，完全不出现 system 角色
        user_content = f"{instruction}\n\n{payload_text}"
    else:
        user_content = payload_text

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})

    body = {
        "model": engine.model,
        "messages": messages,
        "temperature": engine.temperature,
        "stream": False,
        "max_tokens": min(8192, 256 + 120 * len(texts)),
    }

    started = time.time()
    payload = engine._post("/chat/completions", body)
    elapsed = time.time() - started

    choices = payload.get("choices") or []
    content = ((choices[0].get("message") or {}).get("content") if choices else "") or ""

    parsed: list[str] | None = None
    try:
        candidate = json.loads(content.strip())
        if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
            parsed = candidate
    except (json.JSONDecodeError, AttributeError):
        parsed = None

    comparable = [t for t in texts if is_translatable(t)]
    row: dict = {
        "elapsed": round(elapsed, 2),
        "resp_model": payload.get("model"),
        "kana": len(KANA_RE.findall(content)),
        "han": len(HAN_RE.findall(content)),
        "parse_ok": parsed is not None and len(parsed) == len(texts),
        "raw_echo": "源语言" in content,
        "head": content.strip()[:52].replace("\n", "\\n"),
        "sent": len(texts),
        "comparable": len(comparable),
    }

    if row["parse_ok"] and parsed is not None:
        if marker:
            # 机械任务：没加上 [1] 就是没执行
            missed = [o for o in parsed if not o.strip().startswith("[1]")]
            row["same"] = len(missed)
            row["batch_echo"] = len(missed) / len(parsed) > 0.5 if parsed else False
        else:
            same = sum(
                1
                for s, o in zip(texts, parsed)
                if is_translatable(s) and s.strip() == o.strip()
            )
            row["same"] = same
            row["batch_echo"] = row["comparable"] >= 3 and same / row["comparable"] > 0.5
    else:
        # 解析失败：不冒充回抄，单独归类
        row["same"] = None
        row["batch_echo"] = False

    return row


def run_task(
    engine: OpenAICompatTranslator,
    label: str,
    system: str | None,
    instruction: str,
    source_lang: str,
    *,
    batch: int,
    reps: int,
    marker: bool,
    lines: list[str],
    flush,
) -> dict:
    rows: list[dict] = []
    for r in range(reps):
        window = [JA_CORPUS[r % len(JA_CORPUS)]] if batch == 1 else JA_CORPUS[:batch]
        texts = [engine._encode(t) for t in window]
        try:
            row = one_request(
                engine, system, instruction, texts, source_lang, marker=marker
            )
        except Exception as exc:  # noqa: BLE001 - 判别实验，任何异常都记下来
            row = {"error": f"{type(exc).__name__}: {exc}"}
        row["round"] = r + 1
        rows.append(row)

        if "error" in row:
            lines.append(f"  R{r + 1:<2} ERROR  {row['error'][:110]}")
        else:
            if row["parse_ok"]:
                detail = f"同原文 {row['same']:>2}/{row['comparable']}"
                verdict = "整批回抄" if row["batch_echo"] else "正常  "
            else:
                detail = "解析失败(非JSON数组)"
                verdict = "格式坏  "
            lines.append(
                f"  R{r + 1:<2} {row['elapsed']:>5.2f}s  发出 {row['sent']:>2}  "
                f"假名 {row['kana']:>3}  汉字 {row['han']:>3}  {detail:<20} "
                f"{verdict}  原始体回吐={'是' if row['raw_echo'] else '否'}  | {row['head']}"
            )
        flush()

    ok = [r for r in rows if "error" not in r]
    parsed = [r for r in ok if r["parse_ok"]]
    total_cmp = sum(r["comparable"] for r in parsed)
    total_same = sum(r["same"] for r in parsed)
    return {
        "label": label,
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "parse_fails": len(ok) - len(parsed),
        "echo_requests": sum(1 for r in parsed if r["batch_echo"]),
        "items": total_cmp,
        "same": total_same,
        "ratio": round(total_same / total_cmp, 4) if total_cmp else 0.0,
        "kana_min": min((r["kana"] for r in ok), default=0),
        # 日译中成功时假名应当≈0；取「假名>0 的请求数」当失败信号
        "kana_requests": sum(1 for r in ok if r["kana"] > 0),
        "raw_echoes": sum(1 for r in ok if r["raw_echo"]),
        "marker": marker,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="回抄根因判别实验")
    parser.add_argument("--models", default="grok-chat-fast", help="逗号分隔的模型 id")
    parser.add_argument(
        "--tasks", default="translate,marker,userinstr,b1",
        help="逗号分隔：translate / marker / userinstr / b1",
    )
    parser.add_argument("--rounds", type=int, default=8, help="translate/marker/userinstr 的轮数")
    parser.add_argument("--b1-reps", type=int, default=8, help="b1 任务的请求次数")
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--source", default="ja")
    parser.add_argument("--out", default="output/probe_echo.txt")
    args = parser.parse_args()

    cfg = load_config()
    tcfg = cfg.translation

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    summaries: list[dict] = []

    def flush() -> None:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines.append("=" * 150)
    lines.append(
        f"回抄判别实验   tasks={args.tasks}   source={args.source} -> {TARGET_LANG}   "
        f"batch={args.batch}   rounds={args.rounds}"
    )
    lines.append(f"base_url = {tcfg.base_url}")
    lines.append("=" * 150)
    flush()

    wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        tcfg.model = model
        engine = OpenAICompatTranslator.from_config(tcfg)
        instr = engine._system_prompt(TARGET_LANG)

        lines.append(f"\n{'#' * 150}")
        lines.append(f"### model = {model}")
        lines.append(f"{'#' * 150}")
        flush()

        for task in wanted:
            if task == "translate":
                label, system, batch, reps, marker = (
                    "translate (指令走system, 20条/批)", instr, args.batch, args.rounds, False,
                )
            elif task == "marker":
                label, system, batch, reps, marker = (
                    "marker ([1]前缀, 机械任务)", MARKER_PROMPT, args.batch, args.rounds, True,
                )
            elif task == "userinstr":
                label, system, batch, reps, marker = (
                    "userinstr (指令搬进user, 无system)", None, args.batch, args.rounds, False,
                )
            elif task == "b1":
                label, system, batch, reps, marker = (
                    "b1 (翻译, 每批1条)", instr, 1, args.b1_reps, False,
                )
            else:
                lines.append(f"\n[跳过] 未知任务名: {task}")
                flush()
                continue

            lines.append(f"\n--- [{model}] {label} ---")
            flush()
            summary = run_task(
                engine, label, system, instr, args.source,
                batch=batch, reps=reps, marker=marker, lines=lines, flush=flush,
            )
            summary["model"] = model
            summaries.append(summary)
            flush()

    lines.append("\n" + "=" * 150)
    lines.append("汇总")
    lines.append("=" * 150)
    lines.append(
        f"{'模型':<30}{'任务':<32}{'请求':<6}{'报错':<6}{'格式坏':<8}"
        f"{'整批回抄':<9}{'条目同原文':<12}{'假名>0的请求':<13}{'原始体回吐':<10}"
    )
    lines.append("-" * 150)
    for s in summaries:
        lines.append(
            f"{s['model']:<30}{s['label']:<32}{s['requests']:<6}{s['errors']:<6}"
            f"{s['parse_fails']:<8}{s['echo_requests']:<9}"
            f"{s['same']:>3}/{s['items']:<8}{s['kana_requests']:<13}{s['raw_echoes']:<10}"
        )
    flush()
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
