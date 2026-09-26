"""流式接收：SSE 解析、增量切元素、取消时快速中断并保住已收到的条目。

为什么值得有这一整套：一次请求最多带 200 条字幕（上下文窗口上限），模型吐完
要十几秒甚至更久。非流式时取消只在**批与批之间**生效，用户点下取消还得干等
这一批 —— 屏幕上什么都没发生，很像程序卡死。

全部离线：HTTP 层用替身响应对象，不发真实网络请求。真实网络下的端到端验证
（含「服务端确实看到客户端提前断开」这条证据）见 ``scripts/verify_stream_cancel.py``。
"""
from __future__ import annotations

import json
import threading
import time
from email.message import Message

import pytest

from app.core.engines.openai_compat import (
    ArrayStreamParser,
    OpenAICompatTranslator,
    iter_sse_deltas,
)
from app.core.subtitle_io import parse_srt
from app.core.translator import (
    TranslationCancelled,
    TranslationError,
    TranslationRequest,
    Translator,
)

SAMPLE = """1
00:00:01,000 --> 00:00:02,000
hello

2
00:00:03,000 --> 00:00:04,000
world
"""


def make_engine(**overrides) -> OpenAICompatTranslator:
    kwargs = dict(base_url="https://example.invalid/v1", model="m", api_key="k")
    kwargs.update(overrides)
    return OpenAICompatTranslator(**kwargs)


def req(text: str, source: str = "en", target: str = "zh-CN") -> TranslationRequest:
    return TranslationRequest(text=text, source_lang=source, target_lang=target)


def event(text: str) -> str:
    """一个标准 SSE 事件（OpenAI 流式格式）。"""
    payload = {"choices": [{"delta": {"content": text}}]}
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def completion(content: str) -> str:
    """一次性响应的响应体（中继忽略 stream 时就是这么回的）。"""
    payload = {"choices": [{"message": {"content": content}}]}
    return json.dumps(payload, ensure_ascii=False)


class FakeStreamResponse:
    """可按行吐出的响应替身。

    ``reads`` 记录已经读了多少行 —— 测试用它决定「读到第几行时用户点了取消」，
    这比按时间判断稳得多。``fail_after`` 用来模拟读到一半连接断掉。

    ``delay`` 是**交付两道之间的间隔**，默认 0（立即返回）。要测「取消落在中途」
    的用例必须给个正值：真实连接上 ``readline`` 是阻塞的，数据没到就得等；而这里
    若立即返回，后台读取线程会一瞬间把整段读完，取消就永远落在「已经读完」之后，
    测出来的是空战果。给 0.05s 之后，读取线程每次只比调用方领先一行左右。
    """

    def __init__(
        self,
        text: str,
        *,
        content_type: str | None = "text/event-stream",
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
        # 第一行立刻给，好让调用方尽快开始消费；之后每行之间留一点间隔，
        # 模拟真实网络里分片是「一阵一阵来」的。
        #
        # ⚠️ 间隔必须睡在 ``reads += 1`` **之前**：``reads`` 的语义是「已经交付
        # 给调用方的行数」，而用它当取消判据的测试（``resp.reads >= N``）依赖这个
        # 语义。先记账再等待的话，线程一进入等待就已经把这一行算作「已交付」，
        # 判据会提前 N 行命中，测出来的战果少一截。
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

    def __enter__(self) -> "FakeStreamResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False


class BareResponse:
    """连 headers 都没有的极简替身：必须被当成「非流式」，不能崩。"""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self, size: int = -1) -> bytes:
        return self._raw

    def __enter__(self) -> "BareResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


def stub(monkeypatch, *responses) -> None:
    """把 urlopen 换成预置响应，按顺序返回；用完之后重复最后一个（已读空的会自然结束）。"""
    queue = list(responses)

    def fake_urlopen(req, timeout=None, context=None):
        return queue[0] if len(queue) == 1 else queue.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


# ------------------------------------------------------------------ 增量切元素


def test_parser_handles_elements_split_across_chunks():
    """模型是逐 token 吐字的，元素必然被切碎 —— 这正是必须增量解析的原因。"""
    parser = ArrayStreamParser()
    got: list[str] = []
    for piece in ['["你', '好", "世', '界"]']:
        got.extend(parser.feed(piece))
    assert got == ["你好", "世界"]
    assert parser.items == ["你好", "世界"]


def test_parser_skips_the_code_fence():
    parser = ArrayStreamParser()
    parser.feed('```json\n["一",')
    assert parser.feed(' "二"]\n```') == ["二"]


def test_parser_decodes_json_escapes():
    parser = ArrayStreamParser()
    assert parser.feed(r'["a\"b", "\u23ce"]') == ['a"b', "\u23ce"]


def test_parser_leaves_other_shapes_to_the_full_parser():
    """开头不是字符串数组就不猜 —— 整段解析那条路会把 ``[...]`` 片段找出来。"""
    assert ArrayStreamParser().feed('{"translations": ["一"]}') == []


def test_parser_stops_at_a_non_string_element_but_keeps_what_it_has():
    parser = ArrayStreamParser()
    assert parser.feed('["一", 2]') == ["一"]


def test_parser_stops_once_the_array_closes():
    parser = ArrayStreamParser()
    parser.feed('["一"]')
    assert parser.feed('["二"]') == [], "数组已经收尾，后面的内容不该再切"
    assert parser.items == ["一"]


# ------------------------------------------------------------------ SSE 逐行读


def test_iter_sse_deltas_skips_heartbeats_and_bad_lines():
    resp = FakeStreamResponse(
        ": keep-alive\n"
        "\n"
        'data: {"choices":[{"delta":{"content":"甲"}}]}\n'
        "data: not-json\n"
        'data: {"choices":[{"delta":{"content":"乙"}}]}\n'
        "data: [DONE]\n"
        'data: {"choices":[{"delta":{"content":"丙"}}]}\n'
    )
    assert list(iter_sse_deltas(resp)) == ["甲", "乙"]


def test_iter_sse_deltas_accepts_message_style_events():
    """有些中继在流式模式下仍按完整消息回复；只认 delta 会一个字都读不到。"""
    resp = FakeStreamResponse('data: {"choices":[{"message":{"content":"甲"}}]}\n')
    assert list(iter_sse_deltas(resp)) == ["甲"]


def test_iter_sse_deltas_ignores_reasoning_content():
    """推理模型的思维链不是译文。"""
    resp = FakeStreamResponse(
        'data: {"choices":[{"delta":{"reasoning_content":"想想…"}}]}\n'
    )
    assert list(iter_sse_deltas(resp)) == []


def test_iter_sse_deltas_stops_when_asked():
    resp = FakeStreamResponse(
        'data: {"choices":[{"delta":{"content":"甲"}}]}\n'
        'data: {"choices":[{"delta":{"content":"乙"}}]}\n'
    )
    # 每处理完一行查一次取消：第一行收下之后就喊停，第二行不该再往前走
    flags = iter([True])
    got: list[str] = []
    with pytest.raises(TranslationCancelled):
        for delta in iter_sse_deltas(resp, stop=lambda: next(flags, True)):
            got.append(delta)
    assert got == ["甲"], "取消之前已经收到的分片要留住"


# ------------------------------------------------------------------ 取消：核心收益


def test_cancel_mid_stream_closes_the_socket_and_keeps_what_arrived(monkeypatch):
    """流式真正买到的东西：不必等整批返回，已经到手的那几条也不丢。

    一次请求可能有 200 条字幕，丢掉一整批等于让用户白等十几秒。
    """
    engine = make_engine()
    resp = FakeStreamResponse(
        event('["你好", ') + event('"世界", ') + event('"早上好"]'), delay=0.1
    )
    stub(monkeypatch, resp)

    # 读到第 3 行（前两个元素已经完整到手）时用户点了取消
    engine._stop_check = lambda: resp.reads >= 3
    reqs = [req(t) for t in ("hello", "world", "good morning")]

    with pytest.raises(TranslationCancelled) as info:
        engine.translate_batch(reqs)

    assert resp.closed is True, "必须关掉连接 —— 服务端才知道别再往下生成"
    assert list(info.value.partial) == [(0, "你好"), (1, "世界")]
    assert engine.stream_fallback_count == 0, "取消不是降级"


def test_salvaged_items_map_back_through_skipped_blank_cues(monkeypatch):
    """空条目不送模型，元素下标 ≠ 请求下标 —— 换算错了会把译文写到别的条目上。"""
    engine = make_engine()
    resp = FakeStreamResponse(event('["你好", "世界"]'), delay=0.1)
    stub(monkeypatch, resp)
    engine._stop_check = lambda: resp.reads > 0

    reqs = [req(""), req("hello"), req("world")]
    with pytest.raises(TranslationCancelled) as info:
        engine.translate_batch(reqs)

    assert [index for index, _ in info.value.partial] == [1, 2]


def test_salvaged_items_that_were_not_translated_are_dropped(monkeypatch):
    """取消发生在回抄检查之前。把原样退回的条目当成果落盘，等于用「保住了几条」
    换来一份假译文 —— 宁可少救，不能救回没翻的。"""
    engine = make_engine()
    resp = FakeStreamResponse(event('["hello", "世界"]'), delay=0.1)
    stub(monkeypatch, resp)
    engine._stop_check = lambda: resp.reads > 0

    with pytest.raises(TranslationCancelled) as info:
        engine.translate_batch([req("hello"), req("world")])

    assert list(info.value.partial) == [(1, "世界")], "第 0 条是原文回抄，不能算成果"


def test_cancel_also_works_while_a_retry_is_in_flight(monkeypatch):
    """取消可能落在「整批重发」上，那条路径同样要能把战果带回来。"""
    engine = make_engine()
    first = FakeStreamResponse(event('["hello", "world", "good morning"]'))
    second = FakeStreamResponse(event('["你好", ') + event('"世界"]'), delay=0.1)
    stub(monkeypatch, first, second)

    def stop() -> bool:
        # 第一次请求放完，第二次（整批重发）读到一条就喊停
        return second.reads > 1

    engine._stop_check = stop
    reqs = [req(t) for t in ("hello", "world", "good morning")]
    with pytest.raises(TranslationCancelled) as info:
        engine.translate_batch(reqs)

    assert list(info.value.partial) == [(0, "你好")]
    assert engine.echo_retry_count == 1, "确实走了整批重发那条路"


def test_cancel_works_while_waiting_for_the_response_head(monkeypatch):
    """真实中继实测：网关把响应头憋到第一个分片就绪才发，``urlopen`` 一阻塞就是
    3.55s。这几秒里如果同步等着，取消照样要干等到响应头回来 —— 打开动作也必须
    离开调用方线程（``_open_cancellable``）。

    反证：把 ``_open_cancellable`` 换回同步的 ``self._open``，这里会等满
    ``release`` 的等待时间，耗时断言立刻失败。
    """
    engine = make_engine()
    resp = FakeStreamResponse(event('["你好", "世界"]'), delay=0.1)
    release = threading.Event()

    def slow_urlopen(req, timeout=None, context=None):
        release.wait(5.0)  # 模拟「响应头迟迟不来」
        return resp

    monkeypatch.setattr("urllib.request.urlopen", slow_urlopen)

    asked: list[float] = []

    def stop() -> bool:
        asked.append(time.monotonic())
        return True  # 第一次问就喊停

    engine._stop_check = stop
    started = time.monotonic()
    try:
        with pytest.raises(TranslationCancelled):
            engine.translate_batch([req("hello"), req("world")])
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert asked, "取消检查本该在等响应头期间被问到"
    assert elapsed < 1.5, f"不该干等 urlopen（它要 5s 才返回），实际 {elapsed:.2f}s"


# ------------------------------------------------------------------ 降级路径


def test_relay_that_ignores_stream_falls_back_to_a_single_read(monkeypatch):
    """中继完全可以忽略 ``stream: true`` 直接把整个 JSON 甩回来。"""
    engine = make_engine()
    resp = FakeStreamResponse(completion('["你好"]'), content_type="application/json")
    stub(monkeypatch, resp)

    assert engine.translate_batch([req("hello")]) == ["你好"]
    assert engine.stream_fallback_count == 1
    assert any("流式不可用" in note for note in engine.quality_notes())


def test_response_without_headers_counts_as_non_stream(monkeypatch):
    """没有 headers 的响应不能按 SSE 解析 —— 那种情况一个字都读不到。"""
    engine = make_engine()
    raw = completion('["你好"]').encode("utf-8")
    stub(monkeypatch, BareResponse(raw))

    assert engine.translate_batch([req("hello")]) == ["你好"]
    assert engine.stream_fallback_count == 1


def test_a_stream_that_breaks_midway_is_retried_as_a_whole_request(monkeypatch):
    """半截内容没法确认完整性：宁可多花一个请求，也不拿它下结论。"""
    engine = make_engine()
    broken = FakeStreamResponse(event('["你好",'), fail_after=2)
    whole = FakeStreamResponse(
        completion('["你好", "世界"]'), content_type="application/json"
    )
    stub(monkeypatch, broken, whole)

    assert engine.translate_batch([req("hello"), req("world")]) == ["你好", "世界"]
    assert engine.stream_fallback_count == 1
    assert broken.closed is True


def test_an_event_stream_with_no_content_falls_back_to_a_whole_request(monkeypatch):
    """声明了事件流却一个分片都没给。

    真实中继上这是**偶发**的：同一个中继、同一份请求，多数时候流得好好的，
    偶尔整条流空着回来（``scripts/probe_live_stream.py`` 前后两轮就撞到过）。
    流式只是优化，它偶尔空转不该让整批字幕翻不了 —— 退回整段接收再问一次。
    """
    engine = make_engine()
    stub(
        monkeypatch,
        FakeStreamResponse("\n"),
        FakeStreamResponse(completion('["你好"]'), content_type="application/json"),
    )

    assert engine.translate_batch([req("hello")]) == ["你好"]
    assert engine.stream_fallback_count == 1
    assert any("流式不可用" in note for note in engine.quality_notes())


def test_a_relay_that_answers_nothing_at_all_is_reported(monkeypatch):
    """两条路都空手而归时才该报错 —— 而且要说清是「空响应体」。"""
    engine = make_engine()
    stub(
        monkeypatch,
        FakeStreamResponse("\n"),
        FakeStreamResponse("", content_type="application/json"),
    )

    with pytest.raises(TranslationError) as info:
        engine.translate_batch([req("hello")])

    assert "空响应体" in str(info.value)


def test_streaming_can_be_turned_off(monkeypatch):
    """中继的流式实现有问题时得有条退路：一次请求、不带 SSE。"""
    engine = make_engine(stream=False)
    seen: dict = {}

    def fake_urlopen(req, timeout=None, context=None):
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["accept"] = req.get_header("Accept")
        return FakeStreamResponse(
            completion('["你好"]'), content_type="application/json"
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert engine.translate_batch([req("hello")]) == ["你好"]
    assert seen["body"]["stream"] is False
    assert seen["accept"] == "application/json"
    assert engine.stream_fallback_count == 0, "本来就没开流式，不算降级"


def test_streaming_is_on_by_default_and_asks_the_relay_to_stream(monkeypatch):
    engine = make_engine()
    seen: dict = {}

    def fake_urlopen(req, timeout=None, context=None):
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["accept"] = req.get_header("Accept")
        return FakeStreamResponse(event('["你好"]'))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert engine.translate_batch([req("hello")]) == ["你好"]
    assert seen["body"]["stream"] is True
    assert seen["accept"] == "text/event-stream"


def test_stream_setting_comes_from_the_config(tmp_path):
    from app.config import TranslationConfig

    key_file = tmp_path / "k.txt"
    key_file.write_text("g2a_x", encoding="utf-8")
    cfg = TranslationConfig(
        base_url="https://example.invalid/v1",
        model="m",
        api_key_file=str(key_file),
        stream=False,
    )
    assert OpenAICompatTranslator.from_config(cfg).stream is False


# ------------------------------------------------------------------ 取消钩子


class _Probe(Translator):
    """记录「在一次请求内部」查到的是不是外面那个取消回调。"""

    name = "probe"
    seen_hook: object = None

    def translate_batch(self, requests):
        self.seen_hook = self.should_stop()
        return [f"[zh-CN] {r.text}" for r in requests]


def test_the_stop_hook_is_visible_inside_a_request():
    engine = _Probe()
    engine.translate_cues(
        parse_srt(SAMPLE), source_lang="en", target_lang="zh-CN",
        should_stop=lambda: False,
    )
    assert engine.seen_hook is False, "引擎在一次请求内部必须能查到取消"


def test_the_stop_hook_is_cleared_when_the_run_ends():
    """引擎实例可能被复用。钩子留着，下一次任务会一开跑就「被取消」。"""
    engine = _Probe()
    with pytest.raises(TranslationCancelled):
        engine.translate_cues(
            parse_srt(SAMPLE), source_lang="en", target_lang="zh-CN",
            should_stop=lambda: True,
        )
    assert engine.should_stop() is False


# ------------------------------------------------------------------ 调度层写回


class _PartialEngine(Translator):
    """像流式引擎那样：翻到一半取消，把已经收到的条目带回来。"""

    name = "partial"

    def __init__(self, partial):
        self._partial = partial

    def translate_batch(self, requests):
        raise TranslationCancelled("已取消", partial=self._partial)


def test_partial_results_from_a_cancelled_batch_are_kept():
    cues = parse_srt(SAMPLE)
    seen: list[tuple[int, int]] = []
    engine = _PartialEngine([(0, "[zh-CN] 第一条"), (1, "[zh-CN] 第二条")])

    with pytest.raises(TranslationCancelled) as info:
        engine.translate_cues(
            cues,
            source_lang="en",
            target_lang="zh-CN",
            batch_size=10,
            progress=lambda done, total: seen.append((done, total)),
        )

    assert [cue.translation for cue in cues] == ["[zh-CN] 第一条", "[zh-CN] 第二条"]
    assert seen == [(2, 2)], "抢救回来的条目要计入进度 —— 断点靠这个数落盘"
    assert "完成 2/2" in str(info.value)


def test_sloppy_partial_entries_are_ignored():
    """部分结果来自第三方引擎：越界下标绝不能让它把译文写到别的条目上。"""
    cues = parse_srt(SAMPLE)
    engine = _PartialEngine([(9, "越界"), (0, "   "), (1, "好的")])

    with pytest.raises(TranslationCancelled):
        engine.translate_cues(
            cues, source_lang="en", target_lang="zh-CN", batch_size=10
        )

    assert [cue.translation for cue in cues] == ["", "好的"]


def test_partial_entries_from_the_previous_batch_do_not_leak(monkeypatch):
    """第一批正常翻完，第二批被取消并带回部分结果 —— 两批都要对得上。"""
    long_sample = SAMPLE + """
3
00:00:05,000 --> 00:00:06,000
foo

4
00:00:07,000 --> 00:00:08,000
bar
"""
    cues = parse_srt(long_sample)
    calls: list[int] = []

    class Mixed(Translator):
        name = "mixed"

        def translate_batch(self, requests):
            calls.append(len(requests))
            if len(calls) == 1:
                return [f"[zh-CN] {r.text}" for r in requests]
            # 第二批里只收到第一条就取消了
            raise TranslationCancelled("已取消", partial=[(0, "[zh-CN] 第三条")])

    with pytest.raises(TranslationCancelled):
        Mixed().translate_cues(
            cues, source_lang="en", target_lang="zh-CN", batch_size=2
        )

    assert [cue.translation for cue in cues][:3] == [
        "[zh-CN] hello",
        "[zh-CN] world",
        "[zh-CN] 第三条",
    ]
    assert cues[3].translation == "", "没收到的那一条保持未翻译"
