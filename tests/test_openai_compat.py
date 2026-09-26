"""OpenAI 兼容后端的离线测试：协议解析、换行处理、失败回退。

全部通过桩掉 ``_chat`` 完成，不发起任何网络请求。
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from app.config import ConfigError, TranslationConfig
from app.core.engines.openai_compat import (
    _LINE_MARK,
    OpenAICompatTranslator,
    extract_model_ids,
    fetch_models,
)
from app.core.translator import TranslationRequest, TranslationError, create_engine_for


def make_engine(**overrides) -> OpenAICompatTranslator:
    kwargs = dict(
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="g2a_test-key-not-real",
    )
    kwargs.update(overrides)
    return OpenAICompatTranslator(**kwargs)


def req(text: str) -> TranslationRequest:
    return TranslationRequest(text=text, source_lang="en", target_lang="zh-CN")


def user_payload(messages) -> list[str]:
    """取出发给模型的数组。源语言不是 auto 时 user 消息前面会带一行「源语言: xx」。"""
    content = messages[1]["content"]
    if content.startswith("源语言: "):
        content = content.split("\n", 1)[1]
    return json.loads(content)


# ------------------------------------------------------------------ 构造校验

def test_missing_required_fields_raise():
    with pytest.raises(TranslationError):
        OpenAICompatTranslator(base_url="", model="m", api_key="k")
    with pytest.raises(TranslationError):
        OpenAICompatTranslator(base_url="https://x/v1", model="", api_key="k")
    with pytest.raises(TranslationError):
        OpenAICompatTranslator(base_url="https://x/v1", model="m", api_key="")


def test_non_positive_batch_size_raises():
    with pytest.raises(TranslationError):
        make_engine(batch_size=0)
    with pytest.raises(TranslationError):
        make_engine(batch_size=-1)


def test_repr_hides_the_key():
    engine = make_engine(api_key="g2a_DO_NOT_LOG_ME")
    assert "g2a_DO_NOT_LOG_ME" not in repr(engine)
    assert "hidden" in repr(engine)


# ------------------------------------------------------------------ 协议解析

@pytest.mark.parametrize(
    "content",
    [
        '["一", "二"]',
        '```json\n["一", "二"]\n```',
        '```\n["一", "二"]\n```',
        '  ["一", "二"]  ',
        '好的，结果如下：\n["一", "二"]\n以上。',
        '{"translations": ["一", "二"]}',
    ],
)
def test_parse_array_accepts_common_shapes(content):
    assert OpenAICompatTranslator._parse_array(content, 2) == ["一", "二"]


@pytest.mark.parametrize(
    "content",
    [
        '["一"]',              # 数量不符
        '["一", "二", "三"]',  # 数量不符
        "not json at all",     # 不是 JSON
        '["一", 2]',           # 元素不是字符串
        '{"foo": "bar"}',      # 没有数组
    ],
)
def test_parse_array_rejects_bad_shapes(content):
    assert OpenAICompatTranslator._parse_array(content, 2) is None


# ------------------------------------------------------------------ 换行处理

def test_encode_decode_round_trip():
    engine = make_engine()
    original = "line one\nline two"
    encoded = engine._encode(original)
    assert _LINE_MARK in encoded and "\n" not in encoded
    assert engine._decode(encoded) == original


def test_decode_collapses_spaces_around_marker():
    engine = make_engine()
    assert engine._decode(f"前 {_LINE_MARK} 后") == "前\n后"


def test_encode_is_noop_when_preserve_disabled():
    engine = make_engine(preserve_line_breaks=False)
    assert engine._encode("a\nb") == "a\nb"


# ------------------------------------------------------------------ 批量翻译

def test_blank_cues_are_not_sent_to_the_model(monkeypatch):
    engine = make_engine()
    payloads: list[str] = []

    def fake_chat(messages, *, max_tokens=4096):
        payloads.append(messages[-1]["content"])
        return json.dumps(["你好"])

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = engine.translate_batch([req(""), req("hello"), req("   ")])

    assert out == ["", "你好", ""]
    assert len(payloads) == 1, "空条目不应该触发请求"
    assert "hello" in payloads[0]


def test_multiline_payload_uses_marker(monkeypatch):
    engine = make_engine()
    payloads: list[str] = []

    def fake_chat(messages, *, max_tokens=4096):
        payloads.append(messages[-1]["content"])
        return json.dumps([f"第一行{_LINE_MARK}第二行"])

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = engine.translate_batch([req("line one\nline two")])

    assert out == ["第一行\n第二行"]
    assert _LINE_MARK in payloads[0], "送出时应该用可见记号代替换行"
    assert "\\n" not in payloads[0], "不应把裸换行塞进 JSON"


def test_line_break_is_repaired_when_model_drops_the_marker(monkeypatch):
    engine = make_engine()
    calls: list[str] = []

    def fake_chat(messages, *, max_tokens=4096):
        calls.append(messages[-1]["content"])
        if len(calls) == 1:
            return json.dumps(["合并成一行了"])  # 模型把记号吃掉了
        return json.dumps(["第一行", "第二行"])   # 逐行重译

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = engine.translate_batch([req("line one\nline two")])

    assert out == ["第一行\n第二行"]
    assert engine.line_repair_count == 1
    assert len(calls) == 2


def test_single_line_cue_is_not_repaired(monkeypatch):
    engine = make_engine()

    def fake_chat(messages, *, max_tokens=4096):
        return json.dumps(["单行译文"])

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = engine.translate_batch([req("single line")])

    assert out == ["单行译文"]
    assert engine.line_repair_count == 0


def test_batch_protocol_breakdown_falls_back_to_one_by_one(monkeypatch):
    engine = make_engine()
    calls: list[str] = []

    def fake_chat(messages, *, max_tokens=4096):
        calls.append(messages[-1]["content"])
        if len(calls) == 1:
            return "抱歉，我无法完成"  # 不是 JSON -> 批量失败
        return json.dumps([f"单条{len(calls)}"])

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = engine.translate_batch([req("a"), req("b")])

    assert out == ["单条2", "单条3"]
    assert len(calls) == 3


def test_empty_request_list_returns_empty():
    assert make_engine().translate_batch([]) == []


def test_from_config_reads_everything(tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("g2a_file-key", encoding="utf-8")
    cfg = TranslationConfig(
        base_url="https://example.invalid/v1/",
        model="m1",
        api_key_file=str(key_file),
        batch_size=7,
        timeout=33,
        temperature=0.2,
        style_hint="人名保留原文",
        preserve_line_breaks=False,
    )
    engine = OpenAICompatTranslator.from_config(cfg)
    assert engine.base_url == "https://example.invalid/v1"
    assert engine.model == "m1"
    assert engine.batch_size == 7
    assert engine.timeout == 33
    assert engine.style_hint == "人名保留原文"
    assert engine.preserve_line_breaks is False
    assert engine._api_key == "g2a_file-key"
    # 风格约束应该进到 system prompt
    assert "人名保留原文" in engine._system_prompt("zh-CN")


def test_from_config_without_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("SUBTITLE_TRANSLATOR_API_KEY", raising=False)
    cfg = TranslationConfig(base_url="https://x/v1", model="m")
    with pytest.raises(ConfigError):
        OpenAICompatTranslator.from_config(cfg)


# ------------------------------------------------------------------ 注册表

def test_engine_is_registered_under_openai():
    from app.core.translator import ENGINES

    assert "openai" in ENGINES
    assert ENGINES["openai"] is OpenAICompatTranslator


def test_create_engine_for_uses_config(tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("g2a_x", encoding="utf-8")
    from app.config import AppConfig

    cfg = AppConfig(
        translation=TranslationConfig(
            base_url="https://example.invalid/v1", model="m", api_key_file=str(key_file)
        )
    )
    engine = create_engine_for("openai", cfg)
    assert isinstance(engine, OpenAICompatTranslator)
    assert engine.model == "m"


# ------------------------------------------------------------------ 模型列表

class _FakeResponse:
    """urlopen 的最小替身：只需支持 with 与 read。"""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _stub_urlopen(monkeypatch, payload, requests: list | None = None):
    """把 urllib.request.urlopen 换掉。payload 是 bytes 就原样回，否则序列化成 JSON。"""

    def fake_urlopen(req, timeout=None, context=None):
        if requests is not None:
            requests.append(req)
        if isinstance(payload, (bytes, bytearray)):
            body = bytes(payload)
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_extract_model_ids_handles_common_shapes():
    assert extract_model_ids({"data": [{"id": "a"}, {"id": "b"}]}) == ["a", "b"]
    assert extract_model_ids({"models": [{"name": "n"}]}) == ["n"]
    assert extract_model_ids(["m1", "m2"]) == ["m1", "m2"]
    assert extract_model_ids({"data": [{"id": "a"}, {"id": "a"}]}) == ["a"], "重复项要去掉"
    assert extract_model_ids({"data": [{"id": ""}, {"nope": 1}, 42]}) == []
    assert extract_model_ids({}) == []


def test_fetch_models_hits_models_endpoint_with_bearer(monkeypatch):
    requests: list = []
    _stub_urlopen(monkeypatch, {"data": [{"id": "grok-a"}, {"id": "grok-b"}]}, requests)

    models = fetch_models("https://x.invalid/v1/", "g2a_secret")

    assert models == ["grok-a", "grok-b"]
    assert requests[0].full_url == "https://x.invalid/v1/models", "末尾斜杠要归一化"
    assert requests[0].get_header("Authorization") == "Bearer g2a_secret"


def test_fetch_models_accepts_bare_list(monkeypatch):
    _stub_urlopen(monkeypatch, ["m1", "m2"])
    assert fetch_models("https://x.invalid/v1", "k") == ["m1", "m2"]


def test_fetch_models_requires_base_url():
    with pytest.raises(TranslationError):
        fetch_models("", "k")


def test_fetch_models_rejects_non_json(monkeypatch):
    _stub_urlopen(monkeypatch, b"<html>gateway error</html>")
    with pytest.raises(TranslationError, match="不是合法 JSON"):
        fetch_models("https://x.invalid/v1", "k")


def test_fetch_models_surfaces_http_error(monkeypatch):
    def fake_urlopen(req, timeout=None, context=None):
        raise urllib.error.HTTPError(
            req.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(
                json.dumps(
                    {"error": {"code": "invalid_api_key", "message": "bad key"}}
                ).encode("utf-8")
            ),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(TranslationError) as excinfo:
        fetch_models("https://x.invalid/v1", "k")
    message = str(excinfo.value)
    assert "401" in message
    assert "invalid_api_key" in message


def test_fetch_models_surfaces_connection_error(monkeypatch):
    def fake_urlopen(req, timeout=None, context=None):
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(TranslationError, match="连接"):
        fetch_models("https://x.invalid/v1", "k")


def test_list_models_method_delegates_to_fetch_models(monkeypatch):
    _stub_urlopen(monkeypatch, {"data": [{"id": "from-endpoint"}]})
    assert make_engine().list_models() == ["from-endpoint"]


def test_requires_api_flag_distinguishes_engines():
    from app.core.translator import EchoTranslator, Translator

    assert Translator.requires_api is False
    assert EchoTranslator.requires_api is False
    assert OpenAICompatTranslator.requires_api is True


# ------------------------------------------------------------------ 整批回抄重试
#
# 实测背景：中继会间歇性把整个数组原样返回（12 次请求里中 2 次，中→日和日→中
# 都出现过），而且不是零散几条 —— 是整批。这种失败对用户最危险：
# 拿到一份根本没翻译的字幕，界面上却看不出任何异常。


def bulk() -> list[TranslationRequest]:
    return [req(t) for t in ("hello", "world", "good morning")]


def test_whole_batch_echo_is_retried_with_a_sterner_prompt(monkeypatch):
    engine = make_engine()
    src = ["hello", "world", "good morning"]
    replies = [
        json.dumps(src, ensure_ascii=False),                       # 第一次：原样吐回
        json.dumps(["你好", "世界", "早上好"], ensure_ascii=False),   # 重试：正常
    ]
    prompts: list[str] = []

    def fake_chat(messages, *, max_tokens=4096):
        prompts.append(messages[0]["content"])
        return replies.pop(0)

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = engine.translate_batch(bulk())

    assert out == ["你好", "世界", "早上好"]
    assert engine.echo_retry_count == 1
    assert len(prompts) == 2
    assert "重试" not in prompts[0]
    assert "重试" in prompts[1], "重试的那次要把要求说死"


def test_exhausted_echo_retries_raise_instead_of_shipping_garbage(monkeypatch):
    engine = make_engine(echo_retries=1)
    src = ["hello", "world", "good morning"]

    def fake_chat(messages, *, max_tokens=4096):
        return json.dumps(src, ensure_ascii=False)  # 一直回抄

    monkeypatch.setattr(engine, "_chat", fake_chat)
    with pytest.raises(TranslationError) as info:
        engine.translate_batch(bulk())

    assert "完全相同" in str(info.value)
    assert engine.echo_retry_count == 1, "echo_retries=1 表示只重发一次"


def test_echo_retries_zero_means_no_batch_resend(monkeypatch):
    engine = make_engine(echo_retries=0, echo_downgrade=False)
    src = ["hello", "world", "good morning"]
    calls: list[int] = []

    def fake_chat(messages, *, max_tokens=4096):
        calls.append(1)
        return json.dumps(src, ensure_ascii=False)

    monkeypatch.setattr(engine, "_chat", fake_chat)
    with pytest.raises(TranslationError):
        engine.translate_batch(bulk())
    assert len(calls) == 1
    assert engine.echo_retry_count == 0


def test_minority_echo_is_repaired_item_by_item(monkeypatch):
    """3 条里 1 条没翻：不值得整批重发，但那条必须被单独拎出来重译。

    这条正是用户报的现象 —— 「有几率原样推回」。旧逻辑按批次过半判定，
    1/3 不过半就直接放行，未翻译的条目就这样混进成品。
    """
    engine = make_engine()
    calls: list[object] = []

    def fake_chat(messages, *, max_tokens=4096):
        calls.append(user_payload(messages))
        if len(calls) == 1:
            return json.dumps(["你好", "world", "早上好"], ensure_ascii=False)
        return json.dumps(["世界"], ensure_ascii=False)

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = engine.translate_batch(bulk())

    assert out == ["你好", "世界", "早上好"]
    assert engine.echo_retry_count == 0, "1/3 不过半，不该整批重发"
    assert engine.echo_item_count == 1
    assert engine.echo_repaired_count == 1
    assert engine.untranslated_count == 0
    assert len(calls) == 2
    assert calls[1] == ["world"], "只重译出问题的那一条"


def test_partially_untranslated_batch_is_counted_not_fatal(monkeypatch):
    """大部分翻好了、个别条目死活不翻：不该整批报错，但必须计数报出来。"""
    engine = make_engine()
    reqs = [
        TranslationRequest(text=t, source_lang="en", target_lang="zh-CN")
        for t in ("hello", "world", "good morning", "OK")
    ]

    def fake_chat(messages, *, max_tokens=4096):
        payload = user_payload(messages)
        if len(payload) > 1:
            return json.dumps(["你好", "世界", "早上好", "OK"], ensure_ascii=False)
        return json.dumps(payload, ensure_ascii=False)  # 单条请求也原样退回

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = engine.translate_batch(reqs)

    assert out == ["你好", "世界", "早上好", "OK"]
    assert engine.echo_retry_count == 0, "只有 1/4 没翻，不到整批重发的门槛"
    assert engine.echo_item_count == 1, "应该只挑出 'OK' 这一条去重译"
    assert engine.echo_item_attempts == 2, "逐条也要试够次数（echo_item_retries=2）"
    assert engine.echo_repaired_count == 0, "重译也没救回来"
    assert engine.untranslated_count == 1, "没救回来的必须计数"
    assert any("1 条疑似未翻译" in n for n in engine.quality_notes())


def test_auto_source_still_catches_echo_by_reading_the_text(monkeypatch):
    """源语言写 auto（界面默认值）时，整批回抄必须照样被抓到。

    实测背景：auto + 20 条日文字幕，8 批里 2 批整批 100% 原样退回，
    而旧的 should_check_echo 见到 auto 直接返回 False —— 全被静默交付。
    """
    engine = make_engine(echo_downgrade=False)
    src = ["hello", "world", "good morning"]

    def fake_chat(messages, *, max_tokens=4096):
        return json.dumps(src, ensure_ascii=False)

    monkeypatch.setattr(engine, "_chat", fake_chat)
    reqs = [
        TranslationRequest(text=t, source_lang="auto", target_lang="zh-CN")
        for t in src
    ]
    with pytest.raises(TranslationError, match="完全相同"):
        engine.translate_batch(reqs)
    assert engine.echo_retry_count == engine.echo_retries, "auto 也要走完整的重发流程"
    assert engine.echo_retries >= 3, (
        "chat-fast 这种免费档单次回抄率约 1/3，重试 3 次才能把「连续全中」压到 1%"
    )


def test_auto_source_with_matching_language_is_left_alone(monkeypatch):
    """原文本来就是中文、目标也是中文：整批原样是正确结果，不能报错。"""
    engine = make_engine()
    src = ["我已经说过很多次了。", "那种事做不到。", "明白了吗？"]

    def fake_chat(messages, *, max_tokens=4096):
        return json.dumps(src, ensure_ascii=False)

    monkeypatch.setattr(engine, "_chat", fake_chat)
    reqs = [
        TranslationRequest(text=t, source_lang="auto", target_lang="zh-CN")
        for t in src
    ]
    assert engine.translate_batch(reqs) == src
    assert engine.echo_retry_count == 0
    assert engine.echo_item_count == 0


def test_same_language_pair_skips_echo_detection(monkeypatch):
    engine = make_engine()
    src = ["hello", "world", "good morning"]

    def fake_chat(messages, *, max_tokens=4096):
        return json.dumps(src, ensure_ascii=False)

    monkeypatch.setattr(engine, "_chat", fake_chat)
    reqs = [
        TranslationRequest(text=t, source_lang="en-GB", target_lang="en-US")
        for t in src
    ]
    assert engine.translate_batch(reqs) == src
    assert engine.echo_retry_count == 0


def test_per_item_rescue_retries_before_giving_up(monkeypatch):
    """逐条救援也要试够次数：单条同样是概率性失败，只试一次会漏。

    实测 chat-fast 单条请求也有约 1/3 撞上原样退回。只试一次的话，
    「每批零星几条没翻」会一条条渗进成品 —— 一部 1000 条的字幕能累计到几十条。
    """
    engine = make_engine()
    item_calls = 0

    def fake_chat(messages, *, max_tokens=4096):
        nonlocal item_calls
        payload = user_payload(messages)
        if len(payload) > 1:
            return json.dumps(["你好", "world", "早上好"], ensure_ascii=False)
        item_calls += 1
        if item_calls == 1:
            return json.dumps(["world"], ensure_ascii=False)  # 第一次逐条也回抄
        return json.dumps(["世界"], ensure_ascii=False)        # 第二次才成功

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = engine.translate_batch(bulk())

    assert out == ["你好", "世界", "早上好"]
    assert engine.echo_retry_count == 0, "只有 1/3 没过半，不该整批重发"
    assert engine.echo_item_count == 1, "条目去重后只有 1 条"
    assert engine.echo_item_attempts == 2, "试了两次才救回来"
    assert engine.echo_repaired_count == 1
    assert engine.untranslated_count == 0


def test_echo_retry_also_catches_echo_from_the_per_item_fallback(monkeypatch):
    """协议坏掉走逐条回退时，逐条也回抄 —— 同样要被抓到。"""
    engine = make_engine(echo_retries=0)
    src = ["hello", "world", "good morning"]
    calls: list[int] = []

    def fake_chat(messages, *, max_tokens=4096):
        calls.append(1)
        if len(calls) == 1:
            return "抱歉，我无法完成"      # 批量协议失败
        return json.dumps([src[(len(calls) - 2) % len(src)]])  # 逐条也只回抄

    monkeypatch.setattr(engine, "_chat", fake_chat)
    with pytest.raises(TranslationError, match="完全相同"):
        engine.translate_batch(bulk())
    assert len(calls) >= 4, "1 次批量失败 + 至少 3 次逐条"
