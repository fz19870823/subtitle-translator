"""断点续传：记录读写、复用判定、写入节流。不访问网络。"""
from __future__ import annotations

import json

from app.core import checkpoint
from app.core.subtitle_io import parse_srt

SAMPLE = """1
00:00:01,000 --> 00:00:02,000
hello

2
00:00:03,000 --> 00:00:04,000
world
"""

TASK = {
    "engine": "openai",
    "model": "m1",
    "source_lang": "en",
    "target_lang": "zh-CN",
}


def make_cues(translations=("", ""), sample: str = SAMPLE):
    cues = parse_srt(sample)
    for cue, text in zip(cues, translations):
        cue.translation = text
    return cues


def save_for(path, cues, source, **overrides):
    return checkpoint.save(path, cues, source_path=source, **{**TASK, **overrides})


def inspect_for(source, cues, tmp_path, **overrides):
    return checkpoint.inspect(
        source, cues, directory=tmp_path, **{**TASK, **overrides}
    )


# ------------------------------------------------------------------ 路径


def test_checkpoint_path_is_stable_and_per_file(tmp_path):
    first = checkpoint.checkpoint_path(tmp_path / "ep01.srt", directory=tmp_path)
    assert first == checkpoint.checkpoint_path(tmp_path / "ep01.srt", directory=tmp_path)
    assert first != checkpoint.checkpoint_path(tmp_path / "ep02.srt", directory=tmp_path)
    assert first.parent == tmp_path
    assert first.suffix == ".json"


def test_checkpoint_path_keeps_windows_illegal_characters_out(tmp_path):
    path = checkpoint.checkpoint_path(tmp_path / 'a<b>:"c?.srt', directory=tmp_path)
    for char in '<>:"?':
        assert char not in path.name


# ------------------------------------------------------------------ 读写


def test_inspect_round_trip(tmp_path):
    source = tmp_path / "ep01.srt"
    path = checkpoint.checkpoint_path(source, directory=tmp_path)
    save_for(path, make_cues(("[zh] 你好", "")), source)

    plan = inspect_for(source, make_cues(("[zh] 你好", "")), tmp_path)
    assert plan.exists is True
    assert plan.count == 1
    assert plan.usable == {0: "[zh] 你好"}
    assert plan.rejected == ""


def test_save_writes_nothing_when_nothing_is_translated(tmp_path):
    source = tmp_path / "ep01.srt"
    path = checkpoint.checkpoint_path(source, directory=tmp_path)
    assert save_for(path, make_cues(), source) is None
    assert not path.exists(), "一条都没翻就不该留下空记录"


def test_blank_translation_counts_as_unfinished(tmp_path):
    """空白译文按未翻译算 —— 模型偶尔吐回空串，那不是能交付的结果。"""
    source = tmp_path / "ep01.srt"
    cues = parse_srt(SAMPLE)
    cues[0].translation = "   "
    path = checkpoint.checkpoint_path(source, directory=tmp_path)
    assert save_for(path, cues, source) is None
    assert cues[0].is_translated is False


def test_load_tolerates_broken_or_outdated_records(tmp_path):
    path = tmp_path / "x.json"
    assert checkpoint.load(path) is None, "文件不存在"

    path.write_text("{ 不是 json", encoding="utf-8")
    assert checkpoint.load(path) is None, "坏文件必须当作没有断点，不能把翻译挡住"

    path.write_text(json.dumps({"version": 999, "entries": {}}), encoding="utf-8")
    assert checkpoint.load(path) is None, "版本不符"

    path.write_text(
        json.dumps({"version": checkpoint.CHECKPOINT_VERSION}), encoding="utf-8"
    )
    assert checkpoint.load(path) is None, "缺 entries"


def test_save_replaces_atomically_without_tmp_residue(tmp_path):
    source = tmp_path / "ep01.srt"
    path = checkpoint.checkpoint_path(source, directory=tmp_path)
    save_for(path, make_cues(("[zh] 你好",)), source)
    save_for(path, make_cues(("[zh] 你好", "[zh] 世界")), source)

    assert list(tmp_path.glob("*.tmp")) == [], "半截文件不能留在磁盘上"
    assert len(checkpoint.load(path)["entries"]) == 2, "第二次写要整体替换"


def test_clear(tmp_path):
    source = tmp_path / "ep01.srt"
    path = checkpoint.checkpoint_path(source, directory=tmp_path)
    save_for(path, make_cues(("[zh] 你好",)), source)

    assert checkpoint.clear(None) is False
    assert checkpoint.clear(tmp_path / "nope.json") is False
    assert checkpoint.clear(path) is True
    assert not path.exists()


# ------------------------------------------------------------------ 复用判定


def test_model_change_invalidates_the_record(tmp_path):
    source = tmp_path / "ep01.srt"
    path = checkpoint.checkpoint_path(source, directory=tmp_path)
    save_for(path, make_cues(("[zh] 你好", "")), source)

    plan = inspect_for(source, make_cues(("[zh] 你好", "")), tmp_path, model="m2")
    assert plan.count == 0
    assert "模型" in plan.rejected
    assert path.exists(), "不匹配不该删记录 —— 用户换回原模型还能接着用"


def test_edited_source_line_is_dropped(tmp_path):
    """条数没变但内容改了时，序号会整体错位，硬套等于把 A 的译文写到 B 上。"""
    source = tmp_path / "ep01.srt"
    path = checkpoint.checkpoint_path(source, directory=tmp_path)
    save_for(path, make_cues(("[zh] 你好", "[zh] 世界")), source)

    edited = make_cues(("[zh] 你好", ""), sample=SAMPLE.replace("world", "brave world"))
    plan = inspect_for(source, edited, tmp_path)
    assert plan.usable == {0: "[zh] 你好"}
    assert plan.count == 1


def test_all_entries_mismatched_is_reported_as_stale(tmp_path):
    source = tmp_path / "ep01.srt"
    path = checkpoint.checkpoint_path(source, directory=tmp_path)
    save_for(path, make_cues(("[zh] 你好", "[zh] 世界")), source)

    other = make_cues(
        ("[zh] 甲", "[zh] 乙"),
        sample=SAMPLE.replace("hello", "hi").replace("world", "earth"),
    )
    plan = inspect_for(source, other, tmp_path)
    assert plan.stale is True
    assert plan.rejected, "全对不上要说清原因，不能让界面以为无事发生"


def test_shorter_subtitle_file_ignores_out_of_range_entries(tmp_path):
    source = tmp_path / "ep01.srt"
    path = checkpoint.checkpoint_path(source, directory=tmp_path)
    save_for(path, make_cues(("[zh] 你好", "[zh] 世界")), source)

    plan = inspect_for(source, make_cues(("[zh] 你好", ""))[:1], tmp_path)
    assert plan.usable == {0: "[zh] 你好"}


def test_missing_record_is_not_an_error(tmp_path):
    source = tmp_path / "ep01.srt"
    plan = inspect_for(source, make_cues(), tmp_path)
    assert plan.exists is False and plan.stale is False and plan.rejected == ""
    assert plan.path == checkpoint.checkpoint_path(source, directory=tmp_path)


# ------------------------------------------------------------------ 写入节流


def writer_for(tmp_path, now, *, interval=5.0):
    source = tmp_path / "ep01.srt"
    return checkpoint.CheckpointWriter(
        checkpoint.checkpoint_path(source, directory=tmp_path),
        source_path=source,
        interval=interval,
        clock=lambda: now[0],
        **TASK,
    )


def test_writer_writes_the_first_batch_immediately(tmp_path):
    """第一批必须立刻写：开头几批崩掉时，前面两秒的成绩就靠它保住。"""
    now = [1000.0]
    writer = writer_for(tmp_path, now)
    assert writer.maybe(make_cues(("[zh] 你好",))) is True
    assert writer.path.exists()


def test_writer_throttles_between_batches(tmp_path):
    now = [1000.0]
    writer = writer_for(tmp_path, now)
    writer.maybe(make_cues(("[zh] 你好",)))

    now[0] += 0.5
    full = make_cues(("[zh] 你好", "[zh] 世界"))
    assert writer.maybe(full) is False, "间隔没到就不该写盘"
    assert len(checkpoint.load(writer.path)["entries"]) == 1

    now[0] += 5.0
    assert writer.maybe(full) is True
    assert len(checkpoint.load(writer.path)["entries"]) == 2

    # 收尾时不看时间，必须能强制落盘
    assert writer.flush(full) is True


def test_writer_reports_failure_instead_of_raising(tmp_path, monkeypatch):
    """写不进去（磁盘满/只读）不该毁掉整次翻译，但必须留下证据给界面。"""
    def boom(*args, **kwargs):
        raise OSError("磁盘已满")

    monkeypatch.setattr(checkpoint, "save", boom)
    writer = writer_for(tmp_path, [1000.0])
    assert writer.flush(make_cues(("[zh] 你好",))) is False
    assert "磁盘已满" in writer.error


def test_writer_flush_with_nothing_translated_is_not_a_failure(tmp_path):
    writer = writer_for(tmp_path, [1000.0])
    assert writer.flush(make_cues()) is False
    assert writer.error == ""
