"""Ollama 后端（原生 ``/api/chat``）的离线测试。

全部离线：HTTP 层用替身响应对象 / 替换 ``urllib.request.urlopen``。
真实服务端的端到端验证见 ``scripts/verify_ollama.py``。

这一套里最值得看的是三组「防静默故障」的用例：

- ``test_thinking_ate_the_whole_budget_*`` —— qwen3 这类模型会把输出预算
  全用在思维链上、``content`` 返回空串。空串往下走会被判成「整批没翻出来」，
  最后报一句和原因无关的「译文与原文完全相同」。
- ``test_prompt_that_cannot_fit_*`` —— Ollama 超限时**静默截断**提示词，
  而且砍掉的是开头（system 里那段输出格式要求）。发出去之前必须拦住。
- ``test_truncated_prompt_*`` —— 万一还是被截了，靠 ``prompt_eval_count``
  与 token 下界比对认出来，而不是把一份协议崩掉的译文当成品交出去。
"""
from __future__ import annotations

import json
import time
from email.message import Message
from typing import Any, Dict, List

import pytest

from app.config import AppConfig, TranslationConfig
from app.core.engines import ollama as ol
from app.core.engines.ollama import (
    OllamaTranslator,
    count_cjk,
    estimate_tokens,
    extract_tag_names,
    iter_ollama_stream,
    lower_bound_tokens,
    normalise_base_url,
)
from app.core.engines.openai_compat import OpenAICompatTranslator
from app.core.translator import (
    ENGINES,
    TranslationCancelled,
    TranslationError,
    TranslationRequest,
    create_engine_for,
)


def make_engine(**overrides) -> OllamaTranslator:
    kwargs: Dict[str, Any] = dict(
        base_url="http://192.168.1.50:11434", model="qwen2.5:14b"
    )
    kwargs.update(overrides)
    return OllamaTranslator(**kwargs)


def req(text: str, source: str = "en", target: str = "zh-CN") -> TranslationRequest:
    return TranslationRequest(text=text, source_lang=source, target_lang=target)


def reply(content: str, **extra) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": "qwen2.5:14b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
    }
    payload.update(extra)
    return payload


class JsonResponse:
    """一次性响应替身。"""

    def __init__(self, payload: Any, content_type: str = "application/json") -> None:
        self._raw = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._raw

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "JsonResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False


class NdjsonResponse:
    """按行吐出的 NDJSON 替身（Ollama 流式就是这种）。

    ``delay`` 让后台读线程每次只比调用方领先一行，否则一瞬间就读完了，
    「取消落在中途」的用例永远测不到战果。
    """

    def __init__(
        self,
        text: str,
        *,
        content_type: str | None = "application/x-ndjson",
        fail_after: int | None = None,
        delay: float = 0.0,
    ) -> None:
        self._lines = text.splitlines(keepends=True)
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        self.reads = 0
        self.fail_after = fail_after
        self.delay = delay
        self.closed = False

    def readline(self, limit: int = -1) -> bytes:
        # 间隔必须睡在 reads += 1 之前 —— «已交付的行数» 才是取消判据依赖的语义。
        if self.delay and self.reads:
            time.sleep(self.delay)
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise ConnectionResetError("connection reset by peer")
        if not self._lines:
            return b""
        return self._lines.pop(0).encode("utf-8")

    def read(self, size: int = -1) -> bytes:
        return b"".join(line.encode("utf-8") for line in self._lines)

    def close(self) -> None:
        self.closed = True


def stub(monkeypatch, *responses) -> List[Dict[str, Any]]:
    """替换 urlopen，返回「记录下来的请求体」列表。

    用完之后重复最后一个响应（流式那条路会重发一次请求）。
    """
    seen: List[Dict[str, Any]] = []
    queue = list(responses)

    def fake_urlopen(req, timeout=None, context=None):
        data = getattr(req, "data", None)
        if data:
            seen.append(json.loads(data.decode("utf-8")))
        return queue[0] if len(queue) == 1 else queue.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


def ndjson(*pieces: str, stats: Dict[str, Any] | None = None, error: str = "") -> str:
    """拼一段 NDJSON 流（若干内容分片 + 一个 done 收尾）。"""
    lines = [
        json.dumps(
            {"message": {"role": "assistant", "content": piece}, "done": False},
            ensure_ascii=False,
        )
        for piece in pieces
    ]
    if error:
        lines.append(json.dumps({"error": error}, ensure_ascii=False))
        return "\n".join(lines) + "\n"
    tail: Dict[str, Any] = {"message": {"content": ""}, "done": True, "done_reason": "stop"}
    tail.update(stats or {})
    lines.append(json.dumps(tail, ensure_ascii=False))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ 地址规整

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("192.168.1.50:11434", "http://192.168.1.50:11434"),
        ("192.168.1.50:11434/", "http://192.168.1.50:11434"),
        ("http://192.168.1.50:11434/v1", "http://192.168.1.50:11434"),
        ("http://192.168.1.50:11434/api", "http://192.168.1.50:11434"),
        ("http://192.168.1.50:11434/api/chat", "http://192.168.1.50:11434"),
        ("http://h:1/v1/chat/completions", "http://h:1"),
        ("http://h:1/api/tags", "http://h:1"),
        ("", ol.DEFAULT_BASE_URL),
        ("   ", ol.DEFAULT_BASE_URL),
    ],
)
def test_base_url_is_normalised(raw, expected):
    assert normalise_base_url(raw) == expected


def test_bare_host_gets_a_scheme():
    """不补 scheme 的话 urllib 会把主机名当协议名，报一句完全指不到点上的错。"""
    engine = make_engine(base_url="192.168.1.50:11434")
    assert engine.base_url == "http://192.168.1.50:11434"


def test_empty_base_url_falls_back_to_localhost():
    assert make_engine(base_url="").base_url == ol.DEFAULT_BASE_URL


# ------------------------------------------------------------------ 构造

def test_engine_does_not_need_an_api_key():
    """本地服务没有密钥这回事 —— 不能因此拦人。"""
    engine = make_engine()
    assert engine.api_key_required is False
    assert engine._api_key == ol.PLACEHOLDER_KEY
    assert ol.PLACEHOLDER_KEY not in repr(engine)


def test_from_config_without_any_key_is_not_an_error():
    cfg = TranslationConfig(engine="ollama", base_url="http://10.0.0.9:11434", model="m")
    engine = OllamaTranslator.from_config(cfg)
    assert engine.base_url == "http://10.0.0.9:11434"
    assert engine._api_key == ol.PLACEHOLDER_KEY


def test_from_config_reads_the_ollama_section():
    cfg = TranslationConfig(
        engine="ollama",
        base_url="http://10.0.0.9:11434",
        model="m",
        timeout=90,
        extra={
            "ollama": {
                "num_ctx": 16384,
                "keep_alive": "1h",
                "think": True,
                "check_truncation": False,
            }
        },
    )
    engine = OllamaTranslator.from_config(cfg)
    assert engine.num_ctx == 16384
    assert engine.keep_alive == "1h"
    assert engine.think is True
    assert engine.check_truncation is False
    assert engine.timeout == 90


def test_from_config_accepts_flat_extra_keys():
    cfg = TranslationConfig(
        engine="ollama", base_url="http://h:1", model="m", extra={"num_ctx": 4096}
    )
    assert OllamaTranslator.from_config(cfg).num_ctx == 4096


def test_engine_is_registered_and_creatable_by_name():
    assert "ollama" in ENGINES
    cfg = AppConfig(
        translation=TranslationConfig(
            engine="ollama", base_url="http://192.168.1.50:11434", model="qwen2.5:14b"
        )
    )
    engine = create_engine_for("ollama", cfg)
    assert isinstance(engine, OllamaTranslator)
    assert engine.model == "qwen2.5:14b"


def test_ollama_inherits_the_openai_parsing_layer():
    """解析层、回抄恢复、链路重试都从 OpenAI 兼容层继承，不再写第二遍。"""
    assert issubclass(OllamaTranslator, OpenAICompatTranslator)


def test_each_engine_pulls_models_from_its_own_endpoint():
    """两个引擎的「拉取模型」必须指向不同路径 —— 写死一个会让另一个打到 404。"""
    assert OpenAICompatTranslator.fetch_models is not OllamaTranslator.fetch_models


# ------------------------------------------------------------------ 请求体

def test_body_carries_think_keep_alive_and_generation_budget(monkeypatch):
    engine = make_engine(timeout=30)
    seen = stub(monkeypatch, JsonResponse(reply('["一"]')))

    engine._chat([{"role": "user", "content": '["one"]'}], max_tokens=512)

    body = seen[0]
    assert body["model"] == "qwen2.5:14b"
    assert body["stream"] is True  # 默认走流式
    assert body["think"] is False
    assert body["keep_alive"] == ol.OllamaTranslator.keep_alive
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["num_predict"] == 512
    # 没显式配 num_ctx 就不传：Ollama 会用模型自身的上限（实测 32768/40960），
    # 比手填的任何值都合理，也谈不上截断。
    assert "num_ctx" not in body["options"]


def test_num_ctx_is_sent_only_when_configured(monkeypatch):
    engine = make_engine(num_ctx=16384)
    seen = stub(monkeypatch, JsonResponse(reply('["一"]')))
    engine._chat([{"role": "user", "content": '["one"]'}], max_tokens=512)
    assert seen[0]["options"]["num_ctx"] == 16384


def test_non_stream_body_asks_for_json(monkeypatch):
    engine = make_engine(stream=False)
    seen = stub(monkeypatch, JsonResponse(reply('["一"]')))
    engine._chat([{"role": "user", "content": '["one"]'}], max_tokens=128)
    assert seen[0]["stream"] is False


# ------------------------------------------------------------------ 回复解析

def test_reply_content_is_read_from_the_native_shape(monkeypatch):
    engine = make_engine(stream=False)
    stub(monkeypatch, JsonResponse(reply('["一", "二"]', prompt_eval_count=999)))
    assert engine._chat([{"role": "user", "content": '"x"'}]) == '["一", "二"]'


def test_thinking_ate_the_whole_budget_is_reported_with_the_real_reason(monkeypatch):
    """qwen3 把预算全花在思维链上时 content 是空串。

    不在这里说清的话，空串会一路走到「整批没翻出来 → 逐条重译 → 还是空」，
    最后报一句和原因毫无关系的「译文与原文完全相同」。
    """
    engine = make_engine(stream=False)
    stub(
        monkeypatch,
        JsonResponse(
            {
                "message": {"role": "assistant", "content": "", "thinking": "嗯…" * 80},
                "done": True,
                "done_reason": "length",
            }
        ),
    )
    with pytest.raises(TranslationError) as excinfo:
        engine._chat([{"role": "user", "content": '"x"'}])
    message = str(excinfo.value)
    assert "思维链" in message
    assert "think" in message


def test_blank_content_without_thinking_is_an_error(monkeypatch):
    engine = make_engine(stream=False)
    stub(monkeypatch, JsonResponse({"message": {"content": ""}, "done": True}))
    with pytest.raises(TranslationError) as excinfo:
        engine._chat([{"role": "user", "content": '"x"'}])
    assert "空的 message.content" in str(excinfo.value)


def test_missing_message_object_is_an_error(monkeypatch):
    engine = make_engine(stream=False)
    stub(monkeypatch, JsonResponse({"done": True}))
    with pytest.raises(TranslationError):
        engine._chat([{"role": "user", "content": '"x"'}], max_tokens=64)


# ------------------------------------------------------------------ 上下文保护

def test_generation_budget_is_clamped_when_the_prompt_does_not_leave_room(monkeypatch):
    """放不下时压短生成长度（而不是让 Ollama 去砍提示词），并记一笔。"""
    engine = make_engine(num_ctx=2048, stream=False)
    seen = stub(monkeypatch, JsonResponse(reply('["一"]', prompt_eval_count=1200)))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "啊" * 992},
    ]
    engine._chat(messages, max_tokens=4096)

    assert engine.context_clamped_count == 1
    assert seen[0]["options"]["num_predict"] < 4096
    assert seen[0]["options"]["num_ctx"] == 2048
    assert any("被压短" in note for note in engine.quality_notes())


def test_prompt_that_cannot_fit_at_all_is_rejected_before_sending(monkeypatch):
    """连最低预算都放不下时必须拦在本地 —— Ollama 超限是静默截断，不报错。"""
    calls: List[Any] = []

    def explode(req, timeout=None, context=None):  # pragma: no cover - 不该被调到
        calls.append(req)
        raise AssertionError("不该发出这个请求")

    monkeypatch.setattr("urllib.request.urlopen", explode)

    engine = make_engine(num_ctx=512, stream=False)
    with pytest.raises(TranslationError) as excinfo:
        engine._chat([{"role": "user", "content": "啊" * 2000}], max_tokens=512)
    assert calls == [], "拦在本地，一个请求都不该发出去"
    message = str(excinfo.value)
    assert "num_ctx" in message and "上下文窗口" in message


def test_normal_prompt_is_not_flagged_as_truncated(monkeypatch):
    engine = make_engine(num_ctx=8192, stream=False)
    text = "啊" * 600
    stub(monkeypatch, JsonResponse(reply('["一"]', prompt_eval_count=420)))
    engine._chat([{"role": "user", "content": text}], max_tokens=512)
    assert engine.truncated_context_count == 0


def test_truncated_prompt_is_detected_by_comparing_with_the_lower_bound(monkeypatch):
    """服务端只吃下 100 tokens，而这段文本至少 200 tokens —— 提示词被砍了。

    砍掉的是开头，也就是 system 里那段「只输出 JSON 数组」的要求。
    继续跑只会得到协议崩掉的译文，所以宁可停下来报清楚。
    """
    engine = make_engine(num_ctx=4096, stream=False)
    stub(monkeypatch, JsonResponse(reply('["一"]', prompt_eval_count=100)))
    with pytest.raises(TranslationError) as excinfo:
        engine._chat([{"role": "user", "content": "啊" * 600}], max_tokens=512)
    assert engine.truncated_context_count == 1
    message = str(excinfo.value)
    assert "截断" in message and "100" in message


def test_truncation_check_can_be_turned_off(monkeypatch):
    engine = make_engine(num_ctx=4096, stream=False, check_truncation=False)
    stub(monkeypatch, JsonResponse(reply('["一"]', prompt_eval_count=100)))
    engine._chat([{"role": "user", "content": "啊" * 600}], max_tokens=512)
    assert engine.truncated_context_count == 0


# ------------------------------------------------------------------ token 估计

def test_cjk_and_latin_are_counted_differently():
    assert count_cjk("hello 世界") == 2
    assert estimate_tokens("这是一句中文") > estimate_tokens("hello")
    # 上界必须 ≥ 下界，否则截断判据就成了随机数
    for sample in ("这是一句中文", "hello world", "mixed 混合 text", ""):
        assert estimate_tokens(sample) >= lower_bound_tokens(sample)


def test_lower_bound_stays_below_realistic_token_counts():
    """下界的意义是「真实值不该低于它」。拿两段真实文本的估算中值验一下方向。"""
    chinese = "这是一段普通的中文字幕。" * 20
    english = "This is an ordinary subtitle line." * 20
    assert lower_bound_tokens(chinese) < estimate_tokens(chinese)
    assert lower_bound_tokens(english) < estimate_tokens(english)


# ------------------------------------------------------------------ 流式

def test_stream_parses_ndjson_deltas(monkeypatch):
    engine = make_engine()
    stub(monkeypatch, NdjsonResponse(ndjson('["一",', ' "二"]', stats={"prompt_eval_count": 900})))
    content = engine._chat([{"role": "user", "content": "x"}])
    assert content == '["一", "二"]'


def test_stream_falls_back_when_content_type_is_not_ndjson(monkeypatch):
    """中继忽略 stream、直接甩一整个 JSON 回来时必须认出来，否则一个字都读不到。"""
    engine = make_engine()
    stub(monkeypatch, JsonResponse(reply('["一"]'), content_type="application/json"))
    assert engine._chat([{"role": "user", "content": "x"}]) == '["一"]'
    assert engine.stream_fallback_count == 1


def test_stream_that_breaks_halfway_resends_without_stream(monkeypatch):
    engine = make_engine()
    stub(
        monkeypatch,
        NdjsonResponse(ndjson('["一",', ' "二"]'), fail_after=1),
        JsonResponse(reply('["一", "二"]')),
    )
    assert engine._chat([{"role": "user", "content": "x"}]) == '["一", "二"]'
    assert engine.stream_fallback_count == 1


def test_stream_that_yields_nothing_resends_without_stream(monkeypatch):
    engine = make_engine()
    stub(
        monkeypatch,
        NdjsonResponse(ndjson()),  # 只有 done 收尾，一个分片都没有
        JsonResponse(reply('["一"]')),
    )
    assert engine._chat([{"role": "user", "content": "x"}]) == '["一"]'
    assert engine.stream_fallback_count == 1


def test_stream_error_line_is_raised(monkeypatch):
    engine = make_engine()
    stub(monkeypatch, NdjsonResponse(ndjson(error="model 'nope' not found")))
    with pytest.raises(TranslationError) as excinfo:
        engine._chat([{"role": "user", "content": "x"}])
    assert "not found" in str(excinfo.value)


def test_cancel_midway_keeps_the_items_already_received(monkeypatch):
    """取消是本地模型最贵的一课 —— 已经收到的条目必须带走，不能白等。"""
    text = ndjson('["甲",', ' "乙",', ' "丙",', ' "丁"]')
    resp = NdjsonResponse(text, delay=0.02)
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None, context=None: resp)

    engine = make_engine()
    engine._stop_check = lambda: resp.reads >= 3

    order = ["甲", "乙", "丙", "丁"]
    with pytest.raises(TranslationCancelled) as excinfo:
        engine._chat([{"role": "user", "content": "x"}], max_tokens=256)

    kept = [item.strip('"') for item in excinfo.value.items]
    # 具体带走几条取决于线程调度（读线程总会比调用方领先一点），所以不钉死条数，
    # 钉性质：带走的必须是**顺序正确的前缀**，且不是全部（否则就没测到「中途」）。
    assert kept, "取消时必须带走已经收到的条目 —— 本地模型尤其等不起"
    assert kept == order[: len(kept)], "顺序不能乱"
    assert len(kept) < len(order), "不该在整段收完之后才取消"


def test_iter_ollama_stream_ignores_broken_lines():
    resp = NdjsonResponse(
        '{"message":{"content":"甲"},"done":false}\n'
        "这不是 JSON\n"
        '{"message":{"content":"乙"},"done":false}\n'
        '{"done":true,"prompt_eval_count":42}\n'
    )
    stats: Dict[str, Any] = {}
    assert list(iter_ollama_stream(resp, stats=stats)) == ["甲", "乙"]
    assert stats["prompt_eval_count"] == 42


# ------------------------------------------------------------------ 模型列表

@pytest.mark.parametrize(
    "payload",
    [
        {"models": [{"name": "a:1b"}, {"model": "b:2b"}]},
        [{"name": "a:1b"}, {"name": "b:2b"}],
        {"models": ["a:1b", "b:2b"]},
    ],
)
def test_tag_names_are_extracted_from_several_shapes(payload):
    assert extract_tag_names(payload) == ["a:1b", "b:2b"]


def test_tag_names_ignore_junk_and_duplicates():
    assert extract_tag_names({"models": [{"name": ""}, {"nope": 1}, "x", "x"]}) == ["x"]
    assert extract_tag_names(None) == []


def test_fetch_models_hits_the_native_tags_endpoint(monkeypatch):
    seen: List[str] = []

    def fake_urlopen(req, timeout=None, context=None):
        seen.append(req.full_url)
        return JsonResponse({"models": [{"name": "qwen2.5:14b"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert ol.fetch_models("http://192.168.1.50:11434/v1") == ["qwen2.5:14b"]
    assert seen == ["http://192.168.1.50:11434/api/tags"]


def test_fetch_models_surfaces_connection_errors(monkeypatch):
    def boom(req, timeout=None, context=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(TranslationError) as excinfo:
        ol.fetch_models("http://10.0.0.9:11434")
    assert "连接" in str(excinfo.value)


# ------------------------------------------------------------------ 端到端（离线）

def test_batch_translation_goes_through_our_own_parsing_path(monkeypatch):
    """走一遍真实的 translate_batch：请求体、解析、回抄判定都在链路上。"""
    engine = make_engine(stream=False)
    engine._chat = lambda messages, max_tokens=4096: '["你好", "世界"]'  # type: ignore[method-assign]
    out = engine.translate_batch([req("hello"), req("world")])
    assert out == ["你好", "世界"]


def test_untranslated_items_are_still_counted_not_silently_delivered(monkeypatch):
    """回抄恢复这套也自动继承过来了 —— 本地模型「原样返回」同样要被抓住。"""
    engine = make_engine(stream=False, think=False)
    engine._chat = lambda messages, max_tokens=4096: '["hello", "world"]'  # type: ignore[method-assign]
    with pytest.raises(TranslationError) as excinfo:
        engine.translate_batch([req("hello"), req("world"), req("again")])
    assert "完全相同" in str(excinfo.value)
