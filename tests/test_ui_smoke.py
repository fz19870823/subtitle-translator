"""界面冒烟测试：离屏构建主窗口，验证各处理器。

需要同一解释器里同时有 PySide6 和 pytest。缺任一者自动跳过。
在只有单边的机器上可改用等价脚本手动验证。
"""
from __future__ import annotations

import json
import os
import time

import pytest

pytest.importorskip("PySide6", reason="需要 PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppConfig, TranslationConfig  # noqa: E402
from app.core import subtitle_io  # noqa: E402
from app.core.translator import ENGINES, TranslationError, Translator  # noqa: E402
from app.ui import main_window as mw  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

SAMPLE = """1
00:00:01,000 --> 00:00:02,000
hello

2
00:00:03,000 --> 00:00:04,000
world
"""

#: 断点续传的用例需要「取消时确实还剩东西没翻」。
#: 只有 2 条时，取消往往发生在第 2 批已经在路上之后 —— 那其实是跑完了，
#: 不是中断，测试会得出一个和真实场景无关的结论。所以用 4 条。
LONG_SAMPLE = SAMPLE + """
3
00:00:05,000 --> 00:00:06,000
foo

4
00:00:07,000 --> 00:00:08,000
bar
"""


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


#: 关掉的窗口一直留到进程结束。理由和 conftest 里的 ``_QT_KEEPALIVE`` 一样：
#: 本机 Python 3.14.7 + PySide6 6.11.2 会在退出期回收 Qt 对象时踩内存
#: （``gc_collect_harder`` → STATUS_HEAP_CORRUPTION 0xC0000374，测试全绿但退出码不对）。
#: 只钉住 QApplication 还不够 —— 窗口是它的后代，会连带销毁里面的 QThread、
#: QComboBox 等子对象。实测：同一批用例，只有对象数量不同，就能在崩与不崩之间翻转，
#: 所以这里索性让每个窗口活到进程结束，把「对象规模」这个变量消掉。
_WINDOW_KEEPALIVE: list = []


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    """构造主窗口，配置替换为离线 echo，避免测试联网。"""
    import app.config as config_module
    from app.core import checkpoint as checkpoint_store

    offline = AppConfig(translation=TranslationConfig(engine="echo"))
    monkeypatch.setattr(mw, "load_config", lambda *a, **k: offline)
    # 断点目录必须指向临时目录：否则测试之间会互相污染，还会让「有没有断点」
    # 变成随机决定走哪条分支 —— 那种测试绿得没有意义。
    monkeypatch.setattr(
        checkpoint_store, "checkpoints_dir", lambda: tmp_path / "checkpoints"
    )
    win = MainWindow()
    yield win
    win.close()
    _WINDOW_KEEPALIVE.append(win)


def load_sample(win, sample: str = SAMPLE) -> None:
    """模拟打开文件成功后的状态（绕过 QFileDialog）。"""
    from pathlib import Path

    cues = subtitle_io.parse_srt(sample)
    win._cues = cues
    win._source_path = Path("demo.srt")
    win.translate_button.setEnabled(True)
    win.export_button.setEnabled(True)
    # 真实路径上 ``_on_open`` 会顺手重算「哪几条没翻出来」并把重试按钮刷一遍；
    # 这里绕过了对话框，但这两步必须跟上 —— 否则上一份字幕留下的结论会挂在按钮上，
    # 后面断言「按钮该不该出现」就变成在测一个跟当前文件无关的状态。
    win._recompute_stuck()
    win._sync_retry_button()


def wait_idle(win, qapp, timeout_ms: int = 15000) -> None:
    """等后台线程与主线程槽都收尾完毕。

    翻译在后台线程跑，信号是**排队投递**的：线程结束后还得跑一轮事件循环，
    槽函数才会执行。只 ``wait()`` 线程是不够的。
    """
    deadline = time.monotonic() + timeout_ms / 1000
    while win._worker is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert win._worker is None, "翻译收尾（_finish_translation）没有执行"


def run_translate(win, qapp, timeout_ms: int = 15000) -> None:
    """触发翻译并等它彻底跑完（异步化后测试不能再同步断言）。"""
    win._on_translate()
    worker = win._worker
    if worker is not None:
        assert worker.wait(timeout_ms), f"翻译线程 {timeout_ms}ms 内没有结束"
    wait_idle(win, qapp, timeout_ms)


def one_line_per_request(window) -> None:
    """把上下文窗口设成 1 条/次。

    慢引擎那几个用例要观察的是「批」这个粒度：取消得落在中途、进度要一批一批地
    上报、失败要发生在第二批发请求的时候。界面上的上下文窗口默认是 20，
    4 条字幕一口气就发完了，上面那些全成了空话 —— 而且是**悄悄**成空话，
    断言照样写、只是再也测不到东西。所以这些用例必须显式把粒度拧到最小。
    """
    window.context_spin.setValue(1)


class SlowEngine(Translator):
    """每批睡一会儿，用来观察「翻译进行中」的界面状态。

    ``seen`` 记下真正被送去翻译的文本 —— 断点续传要断言「翻过的没被再翻一遍」，
    只看结果译文是看不出来的（覆盖写上去的还是同一句话）。
    """

    name = "openai"
    batch_size = 1

    def __init__(self, delay: float = 0.4) -> None:
        self._delay = delay
        self.seen: list[str] = []

    def translate_batch(self, requests):
        time.sleep(self._delay)
        self.seen.extend(r.text for r in requests)
        return ["[slow] " + r.text for r in requests]


def test_buttons_start_disabled(window):
    assert window.translate_button.isEnabled() is False
    assert window.export_button.isEnabled() is False


def test_engine_combo_lists_registered_engines(window):
    items = [window.engine_combo.itemText(i) for i in range(window.engine_combo.count())]
    assert items == sorted(ENGINES)
    assert "echo" in items and "openai" in items


def test_engine_combo_defaults_to_configured_engine(window):
    assert window.engine_combo.currentText() == "echo"


def test_config_label_shows_summary_without_crashing(window):
    text = window.config_label.text()
    assert "echo" in text
    assert "密钥" in text


def test_translate_fills_editor_and_progress(window, qapp):
    load_sample(window)
    run_translate(window, qapp)
    assert window.progress.value() == 100
    body = window.editor.toPlainText()
    assert "[echo] hello" in body or "[zh-CN] hello" in body


def test_export_round_trip(window, qapp, tmp_path):
    load_sample(window)
    run_translate(window, qapp)
    dst = subtitle_io.write_file(tmp_path / "out.srt", window._cues)
    reloaded = subtitle_io.parse_file(dst)
    assert len(reloaded) == 2
    assert reloaded[0].text.strip().startswith("[")


def test_unknown_engine_shows_error_dialog(window, qapp, monkeypatch):
    load_sample(window)
    captured: list[tuple[str, str]] = []

    class FakeBox:
        @staticmethod
        def critical(parent, title, text, *a, **k):
            captured.append((title, text))

    # main_window 里是 `from ... import QMessageBox`，必须替换它模块命名空间里的名字
    monkeypatch.setattr(mw, "QMessageBox", FakeBox)

    def boom(name, app_config=None):
        from app.core.translator import TranslationError

        raise TranslationError(f"未知的翻译引擎 {name!r}")

    monkeypatch.setattr(mw, "create_engine_for", boom)
    run_translate(window, qapp)

    assert captured and captured[0][0] == "翻译失败"
    assert window.statusBar().currentMessage() == "翻译失败"


def test_engine_combo_is_not_editable(window):
    # 不可编辑是有意为之：引擎必须从注册表里选，不能手打
    assert window.engine_combo.isEditable() is False


# ------------------------------------------------------------------ 模型选择


def test_settings_button_exists(window):
    assert window.settings_button.text() == "设置…"


def test_model_selector_follows_engine_capability(window):
    # 配置里是 echo：没有 API 概念，模型控件应整体禁用
    assert window.engine_combo.currentText() == "echo"
    assert window.model_selector.combo.isEnabled() is False

    window.engine_combo.setCurrentText("openai")
    assert window.model_selector.combo.isEnabled() is True

    window.engine_combo.setCurrentText("echo")
    assert window.model_selector.combo.isEnabled() is False


def test_model_selector_is_editable(window):
    # 必须可编辑：拉不到列表或想用没列出的模型时得能手输
    assert window.model_selector.combo.isEditable() is True


def test_engine_source_reports_configured_endpoint(window):
    window._config.translation.base_url = "https://cfg.invalid/v1"
    window._config.translation.api_key = "cfg-key"

    base_url, key, timeout = window._engine_source()
    assert base_url == "https://cfg.invalid/v1"
    assert key == "cfg-key"
    assert 0 < timeout <= 60


def test_translate_uses_model_picked_in_the_ui(window, qapp, monkeypatch):
    """界面换模型要即时生效，不能悄悄沿用配置文件里的旧值。"""
    load_sample(window)
    window.engine_combo.setCurrentText("openai")
    window.model_selector.set_current_model("ui-picked-model")

    seen: dict = {}

    class FakeEngine(Translator):
        # 必须是真 Translator 子类：主窗口会读它的质量统计
        # （quality_notes 之类），假引擎绕过契约就会漏掉界面依赖。
        name = "openai"

        def translate_batch(self, requests):
            return ["[fake] " + r.text for r in requests]

    def fake_create(name, app_config=None):
        seen["model"] = app_config.translation.model
        return FakeEngine()

    monkeypatch.setattr(mw, "create_engine_for", fake_create)
    run_translate(window, qapp)

    assert seen["model"] == "ui-picked-model"
    assert "[fake] hello" in window.editor.toPlainText()


# ------------------------------------------------------ 后台翻译：不阻塞界面


def test_translation_runs_off_the_ui_thread(window, qapp, monkeypatch):
    """翻译必须在后台线程跑。

    这正是要修的故障：以前直接在界面线程里调 ``translate_cues()``，
    一部上千条的字幕要发几十次请求、跑好几分钟，窗口在整个过程中完全冻结 ——
    进度条不重绘、连「取消」都点不了，用户只看到程序死了。
    """
    load_sample(window)
    window.engine_combo.setCurrentText("openai")
    one_line_per_request(window)
    monkeypatch.setattr(mw, "create_engine_for", lambda name, cfg=None: SlowEngine())

    started = time.monotonic()
    window._on_translate()
    elapsed = time.monotonic() - started

    assert elapsed < 0.2, "点「翻译」必须立刻返回，不能等翻译跑完"
    assert window._worker is not None and window._worker.isRunning()
    assert window.progress.value() == 0, "还没跑完不该显示完成"

    # 能处理事件 = 界面没被冻住，这是「不卡死」的直接判据
    qapp.processEvents()
    assert window.translate_button.isEnabled() is False, "翻译期间不该能重复点击"
    assert window.open_button.isEnabled() is False, "翻译期间不该能换文件"
    assert window.export_button.isEnabled() is False, "翻译期间不该能导出半成品"
    assert window.context_spin.isEnabled() is False, "翻译期间不该能改上下文窗口"
    assert window.cancel_button.isHidden() is False, "翻译期间要能看到取消按钮"

    wait_idle(window, qapp)
    assert window.progress.value() == 100
    assert window.translate_button.isEnabled() is True
    assert window.open_button.isEnabled() is True
    assert window.context_spin.isEnabled() is True, "跑完了要把上下文窗口放回可改"
    assert window.cancel_button.isHidden() is True, "空闲时取消按钮该收起来"


def test_cancel_stops_the_translation(window, qapp, monkeypatch):
    """取消要真的停下：不再发新请求，并保住已翻好的部分。"""
    load_sample(window)
    window.engine_combo.setCurrentText("openai")
    one_line_per_request(window)
    monkeypatch.setattr(mw, "create_engine_for", lambda name, cfg=None: SlowEngine())

    window._on_translate()
    worker = window._worker
    assert worker is not None and worker.isRunning()

    window._on_cancel()
    assert worker.wait(10000), "取消后线程没有退出"
    done, total = worker.done, worker.total
    wait_idle(window, qapp)

    assert "已取消" in window.statusBar().currentMessage()
    assert window.translate_button.isEnabled() is True, "取消后要能重新开始"
    assert done < total, "取消意味着没有全部翻完"


def test_progress_bar_follows_the_background_worker(window, qapp, monkeypatch):
    """进度必须随后台进度实时更新，而不是跑完才跳一下。"""
    load_sample(window)
    window.engine_combo.setCurrentText("openai")
    one_line_per_request(window)
    monkeypatch.setattr(mw, "create_engine_for", lambda name, cfg=None: SlowEngine())

    seen: list[int] = []
    window._on_translate()
    worker = window._worker
    assert worker is not None
    worker.progressed.connect(lambda done, total: seen.append(done))

    wait_idle(window, qapp)
    assert seen == [1, 2], "第一批处理完就应该上报一次进度"
    assert window.progress.value() == 100


def test_closing_while_translating_sends_the_thread_away(window, qapp, monkeypatch):
    """翻译中途关窗不能留一个还在跑的线程。

    窗口对象一被回收，运行中的 QThread 就会踩野指针，Qt 会直接报
    "Destroyed while thread is still running" 并可能崩掉进程。
    """
    load_sample(window)
    window.engine_combo.setCurrentText("openai")
    one_line_per_request(window)
    monkeypatch.setattr(mw, "create_engine_for", lambda name, cfg=None: SlowEngine())

    window._on_translate()
    worker = window._worker
    assert worker is not None and worker.isRunning()

    window.close()

    assert not worker.isRunning(), "closeEvent 必须把后台线程送走"


# ------------------------------------------------------ 断点续传


def install_engine(window, monkeypatch, delay: float = 0.4) -> list:
    """把会记账的慢引擎装进主窗口，返回每次创建的实例列表。"""
    one_line_per_request(window)
    created: list[SlowEngine] = []

    def factory(name, app_config=None):
        engine = SlowEngine(delay)
        created.append(engine)
        return engine

    monkeypatch.setattr(mw, "create_engine_for", factory)
    return created


def wait_for_progress(win, qapp, count: int, timeout_ms: int = 8000) -> None:
    """等后台线程完成至少 ``count`` 条。"""
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        worker = win._worker
        if worker is None or worker.done >= count:
            return
        qapp.processEvents()
        time.sleep(0.01)
    raise AssertionError(f"等不到 {count} 条的进度")


def cancel_midway(window, qapp, monkeypatch) -> tuple:
    """跑一次翻译，翻到一半就取消。返回 ``(已翻条数, 断点路径, 引擎列表)``。

    用 4 条字幕是有讲究的：只有 2 条时，等第 1 条翻完再点取消，第 2 批其实
    已经在路上了 —— 那一轮会正常跑完、根本不算中断，测出来的东西与真实场景无关。

    还有一点：断言里**不写「恰好翻完 1 条」**。取消请求和批次返回之间存在竞态，
    能确定的只有「没翻完，且落盘的条数恰好等于实际翻好的条数」。
    """
    load_sample(window, LONG_SAMPLE)
    window.engine_combo.setCurrentText("openai")
    created = install_engine(window, monkeypatch)

    window._on_translate()
    wait_for_progress(window, qapp, 1)
    window._on_cancel()
    assert window._worker.wait(15000)

    worker = window._worker
    done, ckpt = worker.done, worker.checkpoint_path
    assert 0 < done < len(window._cues), "取消必须发生在整份翻完之前"
    wait_idle(window, qapp)
    return done, ckpt, created


def test_cancel_leaves_a_resumable_checkpoint(window, qapp, monkeypatch):
    """取消要留下进度：不然用户点一次取消，前面几分钟就白跑了。"""
    done, ckpt, _ = cancel_midway(window, qapp, monkeypatch)

    assert ckpt is not None and ckpt.exists(), "取消必须把进度写下来"
    payload = json.loads(ckpt.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == done, "落盘的条数要与实际翻好的对上"
    assert "进度已保存" in window.statusBar().currentMessage()
    assert window.translate_button.text() == "继续翻译"


def test_resume_translates_only_what_is_left(window, qapp, monkeypatch):
    """续传只补没翻的条目，整份翻完后把记录清掉。"""
    done, ckpt, created = cancel_midway(window, qapp, monkeypatch)
    texts = [cue.text for cue in window._cues]

    asked: list[int] = []

    def fake_ask(plan):
        asked.append(plan.count)
        return "resume"

    monkeypatch.setattr(window, "_ask_resume", fake_ask)
    run_translate(window, qapp)

    assert asked == [done], "续传前要问一次，并把能复用的条数报给用户"
    assert created[-1].seen == texts[done:], "翻过的条目不该再发一次（白烧额度）"
    assert window.progress.value() == 100
    assert not ckpt.exists(), "整份翻完了，记录就该删掉"
    assert window.translate_button.text() == "翻译"


def test_resume_dialog_can_restart_from_scratch(window, qapp, monkeypatch):
    """用户可能就是想整份重来（换个模型试试效果），必须给这条出路。"""
    _, _, created = cancel_midway(window, qapp, monkeypatch)
    texts = [cue.text for cue in window._cues]

    monkeypatch.setattr(window, "_ask_resume", lambda plan: "restart")
    run_translate(window, qapp)

    assert created[-1].seen == texts, "选了重新开始就要整份重译"


def test_cancelling_the_resume_dialog_starts_nothing(window, qapp, monkeypatch):
    done, _, created = cancel_midway(window, qapp, monkeypatch)

    monkeypatch.setattr(window, "_ask_resume", lambda plan: "cancel")
    window._on_translate()

    assert window._worker is None, "选了取消就不该启动翻译"
    assert len(created) == 1, "不该再构造引擎"
    assert window._cues[done].translation == "", "取消不该动任何条目"
    assert f"保留着 {done} 条的进度" in window.statusBar().currentMessage()


def test_checkpoint_from_another_model_is_reported_not_silently_ignored(
    window, qapp, monkeypatch
):
    """换了模型就续不上，必须说出来 —— 静默从头翻，用户基本发现不了。"""
    _, ckpt, _ = cancel_midway(window, qapp, monkeypatch)
    window.model_selector.set_current_model("another-model")

    window._on_translate()
    message = window.statusBar().currentMessage()
    assert "已忽略上次的翻译记录" in message
    assert "模型" in message
    assert ckpt.exists(), "不匹配不该删记录：换回原模型还能接着用"

    window._on_cancel()
    assert window._worker.wait(15000)
    wait_idle(window, qapp)


def test_progress_bar_jumps_to_the_checkpoint(window, qapp, monkeypatch):
    """续传时进度条立刻落在断点处，不干等第一批请求回来。"""
    done, _, _ = cancel_midway(window, qapp, monkeypatch)

    monkeypatch.setattr(window, "_ask_resume", lambda plan: "resume")
    window._on_translate()
    expected = int(done / len(window._cues) * 100)
    assert window.progress.value() == expected, "进度条要反映已经翻好的部分"

    window._on_cancel()
    assert window._worker.wait(15000)
    wait_idle(window, qapp)


def test_closing_during_translation_keeps_the_progress(window, qapp, monkeypatch):
    """直接关窗是最常见的「我不想等了」—— 进度不能跟着窗口一起消失。"""
    load_sample(window, LONG_SAMPLE)
    window.engine_combo.setCurrentText("openai")
    install_engine(window, monkeypatch)

    window._on_translate()
    wait_for_progress(window, qapp, 1)
    worker = window._worker
    ckpt = worker.checkpoint_path
    window.close()  # closeEvent 会 cancel + wait，线程收尾时落盘

    assert not worker.isRunning(), "关窗必须把后台线程送走"
    assert ckpt is not None and ckpt.exists(), "关窗也要留下断点"
    assert 0 < worker.done < len(window._cues)
    worker.wait(15000)
    wait_idle(window, qapp)


def test_failed_translation_keeps_the_finished_part(window, qapp, monkeypatch):
    """失败同样要留住已翻好的部分：502 重试耗尽、中继突然挂掉都是这样。"""
    load_sample(window, LONG_SAMPLE)
    window.engine_combo.setCurrentText("openai")
    # 必须让它撑到第二批才断：一批一封全挂掉的话，一条都没翻好，也就无从谈「留住进度」
    one_line_per_request(window)

    class HalfBroken(SlowEngine):
        """第一批正常，之后整条链路都失联。"""

        def __init__(self) -> None:
            super().__init__(delay=0.05)
            self._batches = 0

        def translate_batch(self, requests):
            self._batches += 1
            if self._batches >= 2:
                raise TranslationError("HTTP 502（已自动重试 3 次）")
            return super().translate_batch(requests)

    monkeypatch.setattr(mw, "create_engine_for", lambda name, cfg=None: HalfBroken())

    captured: list[str] = []

    class FakeBox:
        @staticmethod
        def critical(parent, title, text, *a, **k):
            captured.append(text)

    monkeypatch.setattr(mw, "QMessageBox", FakeBox)
    from app.core import checkpoint as checkpoint_store

    ckpt = checkpoint_store.checkpoint_path(
        window._source_path, directory=window._checkpoint_dir
    )
    run_translate(window, qapp)

    assert captured and "已经存下来了" in captured[0], "失败时必须告诉用户进度还在"
    assert ckpt.exists(), "失败也要留下断点"
    assert "翻译失败" in window.statusBar().currentMessage()
    assert window.translate_button.text() == "继续翻译", "修好后应该能接着翻"


# ------------------------------------------------------ 上下文窗口与手动重试


class StubbornEngine(Translator):
    """有几条怎么都不肯翻（把原文原样退回），用来测「重试未翻译」。

    这正是真实故障的形态：中继把请求路由到不干活的通道时，模型把原文当译文吐回来。
    ``seen`` 记下每次真正发出去的文本 —— 「重试只发没翻的那几条」靠它验证，
    光看结果译文看不出来（重试成功后那几条也还是同一句话）。
    """

    name = "openai"
    #: 故意设成与界面范围都不同的值：能看出到底是引擎默认值还是界面上设的值在起作用
    batch_size = 999

    def __init__(self, stubborn=()) -> None:
        self._stubborn = set(stubborn)
        self.seen: list[str] = []
        self.untranslated_count = 0

    def translate_batch(self, requests):
        self.seen.extend(r.text for r in requests)
        out = []
        for request in requests:
            if request.text in self._stubborn:
                out.append(request.text)  # 原样退回 = 没翻
                self.untranslated_count += 1
            else:
                out.append(f"[zh] {request.text}")
        return out


class BatchRecorder(Translator):
    """只记每批发了几条，用来验证上下文窗口真的落到了请求切分上。"""

    name = "openai"
    batch_size = 999  # 同上：引擎默认值不该盖过界面上的设置

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def translate_batch(self, requests):
        self.sizes.append(len(requests))
        return [f"[zh] {r.text}" for r in requests]


def stuck_run(window, qapp, monkeypatch, stubborn=("foo",)) -> list:
    """跑一次翻译，让 ``stubborn`` 里的条目原样退回；返回每次创建的引擎列表。"""
    load_sample(window, LONG_SAMPLE)
    window.engine_combo.setCurrentText("openai")
    created: list[StubbornEngine] = []

    def factory(name, cfg=None):
        engine = StubbornEngine(stubborn)
        created.append(engine)
        return engine

    monkeypatch.setattr(mw, "create_engine_for", factory)
    # 成功后的弹窗会阻塞事件循环，自动化里点不到它，换成预设答案
    monkeypatch.setattr(window, "_ask_retry_stuck", lambda count, retried=False: False)
    run_translate(window, qapp)
    return created


def test_context_window_defaults_to_the_configured_value(window):
    # 配置里没写 batch_size，TranslationConfig 的默认值是 20
    assert window.context_spin.value() == 20
    assert window.context_spin.minimum() == 1
    assert window.context_spin.maximum() == 200


def test_context_window_clamps_an_out_of_range_config(window):
    """配置是人工写的，可能填 0 或 9999；界面显示的值必须是实际会生效的那个。"""
    window._config.translation.batch_size = 9999
    assert window._configured_batch_size() == 200
    window._config.translation.batch_size = 0
    assert window._configured_batch_size() == 1


def test_context_window_controls_how_many_lines_go_per_request(window, qapp, monkeypatch):
    """窗口决定每批发几条，且改完立刻生效 —— 不用重启程序、也不用进设置。"""
    load_sample(window, LONG_SAMPLE)
    window.engine_combo.setCurrentText("openai")
    recorder = BatchRecorder()
    monkeypatch.setattr(mw, "create_engine_for", lambda name, cfg=None: recorder)

    window.context_spin.setValue(2)
    run_translate(window, qapp)
    assert recorder.sizes == [2, 2], "4 条字幕按 2 条一批，就该发两批"

    recorder.sizes.clear()
    window.context_spin.setValue(3)
    run_translate(window, qapp)
    assert recorder.sizes == [3, 1], "窗口调大后每批就该多带几条"


def test_context_window_follows_saved_settings(window, monkeypatch, tmp_path):
    """设置里改了上下文窗口要反映到主界面控件上，否则界面显示的值是假的。"""
    from dataclasses import replace

    target = tmp_path / "config.local.json"
    window._config.source = target

    class FakeDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Accepted

        def result_config(self):
            return replace(
                window._config,
                translation=replace(window._config.translation, batch_size=5),
            )

    monkeypatch.setattr(mw, "SettingsDialog", FakeDialog)
    window._on_settings()

    assert window.context_spin.value() == 5


def test_untranslated_items_surface_a_manual_retry_button(window, qapp, monkeypatch):
    """自动重试用尽后仍没翻出来的条目，必须看得见、点得到。"""
    stuck_run(window, qapp, monkeypatch)

    assert window._stuck_indices == [2], "只有原样退回的那条该被定罪"
    assert window.retry_button.isHidden() is False, "有没翻出来的就该出现重试按钮"
    assert window.retry_button.text() == "重试未翻译（1）"
    assert "仍有 1 条未翻译" in window.statusBar().currentMessage()


def test_identical_but_legitimate_items_are_not_blamed(window, qapp, monkeypatch):
    """源与目标同族时不能定罪：中文译中文、英文译英文里，原样保留本就常见。"""
    load_sample(window, LONG_SAMPLE)
    window.engine_combo.setCurrentText("openai")
    monkeypatch.setattr(
        mw, "create_engine_for", lambda name, cfg=None: StubbornEngine({"foo"})
    )
    monkeypatch.setattr(window, "_ask_retry_stuck", lambda count, retried=False: False)
    window.target_combo.setCurrentText("en")  # 原文与目标都是拉丁族 → 放过
    run_translate(window, qapp)

    assert window._stuck_indices == [], "同族语言之间不该判成没翻"
    assert window.retry_button.isHidden() is True


def test_manual_retry_only_sends_the_stuck_items(window, qapp, monkeypatch):
    """重试只发那几条，不动已经翻好的部分（否则等于整份重跑，白烧额度）。"""
    stuck_run(window, qapp, monkeypatch)
    assert window._stuck_indices == [2]

    created: list[StubbornEngine] = []

    def factory(name, cfg=None):
        engine = StubbornEngine()  # 这次肯干活了
        created.append(engine)
        return engine

    monkeypatch.setattr(mw, "create_engine_for", factory)
    window._on_retry_stuck()
    worker = window._worker
    assert worker is not None and worker.wait(15000)
    wait_idle(window, qapp)

    assert created[-1].seen == ["foo"], "重试只该发没翻出来的那条"
    assert window._stuck_indices == []
    assert window.retry_button.isHidden() is True, "都翻好了按钮就该收起来"
    assert "1/1 条已翻好" in window.statusBar().currentMessage()


def test_manual_retry_does_not_write_a_checkpoint(window, qapp, monkeypatch):
    """重试不能写断点：它拿到的是子集下标，记下来会把译文安到别的条目名下。"""
    from app.core import checkpoint as checkpoint_store

    stuck_run(window, qapp, monkeypatch)
    ckpt = checkpoint_store.checkpoint_path(
        window._source_path, directory=window._checkpoint_dir
    )
    assert not ckpt.exists(), "整份已经翻完，断点早该清掉"

    monkeypatch.setattr(mw, "create_engine_for", lambda name, cfg=None: StubbornEngine())
    window._on_retry_stuck()
    worker = window._worker
    assert worker is not None
    assert worker.checkpoint_path is None, "重试路径不挂断点写入器"
    assert worker.wait(15000)
    wait_idle(window, qapp)
    assert not ckpt.exists(), "重试跑完也不该凭空冒出一个记录文件"


def test_warning_can_trigger_the_retry_on_the_spot(window, qapp, monkeypatch):
    """弹窗里直接点「重试这些条目」就该接着跑，不必再让用户去找按钮。"""
    load_sample(window, LONG_SAMPLE)
    window.engine_combo.setCurrentText("openai")
    created: list[StubbornEngine] = []

    def factory(name, cfg=None):
        # 第一次构造出来的引擎偷懒，重试时构造的那个肯翻
        engine = StubbornEngine({"foo"}) if not created else StubbornEngine()
        created.append(engine)
        return engine

    monkeypatch.setattr(mw, "create_engine_for", factory)
    monkeypatch.setattr(window, "_ask_retry_stuck", lambda count, retried=False: True)

    run_translate(window, qapp)  # wait_idle 会把弹窗接手的重试那一轮一起等完

    assert len(created) == 2, "第一次翻译 + 弹窗里接手的一次重试"
    assert created[-1].seen == ["foo"]
    assert window._stuck_indices == []
    assert window.retry_button.isHidden() is True


def test_retry_reports_what_still_fails_after_another_round(window, qapp, monkeypatch):
    """再试一轮还是不行，就得如实说「没翻好」，并保留按钮供用户再试。"""
    stuck_run(window, qapp, monkeypatch)

    asked: list[tuple] = []

    def fake_ask(count, *, retried=False):
        asked.append((count, retried))
        return False

    monkeypatch.setattr(window, "_ask_retry_stuck", fake_ask)
    window._on_retry_stuck()  # 引擎仍然偷懒
    worker = window._worker
    assert worker is not None and worker.wait(15000)
    wait_idle(window, qapp)

    assert asked == [(1, True)], "第二轮要说明「刚才的手动重试也没用」"
    assert window._stuck_indices == [2], "还是没翻出来就得继续认账"
    assert "0/1 条已翻好" in window.statusBar().currentMessage()
    assert window.retry_button.isHidden() is False, "还失败着就该留着按钮再试"


def test_manual_retry_refreshes_the_checkpoint_so_resume_keeps_it(window, qapp, monkeypatch):
    """中断后手动重试补上的几条，不能在下次续传时被记录里的旧值盖回原文。

    这是最容易漏掉的一种「静默错误」：断点记录里存的是回抄值（等于原文），
    续传会拿它无条件覆盖 ``cue.translation`` —— 用户刚花掉的那次重试就白费了，
    而且界面上看不出任何异常。
    """
    load_sample(window, LONG_SAMPLE)
    window.engine_combo.setCurrentText("openai")
    one_line_per_request(window)
    monkeypatch.setattr(window, "_ask_retry_stuck", lambda count, retried=False: False)

    class SlowStubborn(StubbornEngine):
        def translate_batch(self, requests):
            time.sleep(0.3)
            return super().translate_batch(requests)

    # 第 1 条（hello）回抄，其余正常 —— 翻过它之后取消
    monkeypatch.setattr(
        mw, "create_engine_for", lambda name, cfg=None: SlowStubborn({"hello"})
    )
    window._on_translate()
    wait_for_progress(window, qapp, 1)
    window._on_cancel()
    assert window._worker.wait(15000)
    ckpt = window._worker.checkpoint_path
    wait_idle(window, qapp)

    assert ckpt is not None and ckpt.exists()
    assert window._stuck_indices == [0]
    payload = json.loads(ckpt.read_text(encoding="utf-8"))
    assert payload["entries"]["0"]["t"] == "hello", "记录里存的就是那条回抄值"

    # 手动重试那一条，这次翻好了
    monkeypatch.setattr(mw, "create_engine_for", lambda name, cfg=None: StubbornEngine())
    window._on_retry_stuck()
    assert window._worker.wait(15000)
    wait_idle(window, qapp)
    assert window._cues[0].translation == "[zh] hello"
    assert window._stuck_indices == []

    refreshed = json.loads(ckpt.read_text(encoding="utf-8"))
    assert refreshed["entries"]["0"]["t"] == "[zh] hello", "记录要跟着刷新"

    # 接着续传：第 0 条不能被记录里的旧回抄值盖回去
    monkeypatch.setattr(window, "_ask_resume", lambda plan: "resume")
    run_translate(window, qapp)
    assert window._cues[0].translation == "[zh] hello", "续传不该把重试的成果退回原文"


# ------------------------------------------------------------------ 设置保存


def test_settings_dialog_result_is_persisted(window, monkeypatch, tmp_path):
    from dataclasses import replace

    target = tmp_path / "config.local.json"
    # 让保存落到临时目录：绝不能在测试里碰仓库里的真配置
    window._config.source = target

    class FakeDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Accepted

        def result_config(self):
            return replace(
                window._config,
                translation=replace(window._config.translation, model="saved-model"),
            )

    monkeypatch.setattr(mw, "SettingsDialog", FakeDialog)
    window._on_settings()

    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["translation"]["model"] == "saved-model"
    assert window._config.translation.model == "saved-model"
    assert window.model_selector.current_model() == "saved-model"
    assert "已保存" in window.statusBar().currentMessage()


def test_cancelled_settings_dialog_writes_nothing(window, monkeypatch, tmp_path):
    target = tmp_path / "config.local.json"
    window._config.source = target

    class CancelDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(mw, "SettingsDialog", CancelDialog)
    window._on_settings()

    assert not target.exists(), "点了取消就不该写文件"
