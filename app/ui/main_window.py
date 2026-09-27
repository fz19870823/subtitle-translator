"""主窗口。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import (
    APP_NAME,
    DEFAULT_SOURCE_LANG,
    DEFAULT_TARGET_LANG,
    OUTPUT_DIR,
    AppConfig,
    ConfigError,
    ensure_runtime_dirs,
    load_config,
    save_config,
)
from app.core import subtitle_io
from app.core import checkpoint as checkpoint_store
from app.core import queue as queue_store
from app.core.subtitle_io import (
    UNTRANSLATED_MARK,
    Cue,
    OutputSink,
    SubtitleFormatError,
)
from app.core.translator import (
    ENGINES,
    TranslationError,
    create_engine_for,
)
from app.ui.model_selector import MAX_FETCH_TIMEOUT, ModelSelector
from app.ui.queue_panel import QueuePanel
from app.ui.settings_dialog import SettingsDialog
from app.ui.translate_worker import TranslateWorker

#: 关窗时最多等后台翻译线程多久收尾（毫秒）。超过就放弃等待，避免窗口卡住。
CLOSE_WAIT_MS = 8000

#: 「上下文窗口」（单次请求携带多少条字幕）的取值范围与兜底默认值。
#: 上下限与设置对话框保持一致 —— 两处是同一个参数，范围不该各说各话。
CONTEXT_MIN = 1
CONTEXT_MAX = 200
CONTEXT_DEFAULT = 20


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_runtime_dirs()
        self._cues: List[Cue] = []
        self._source_path: Path | None = None
        #: 正在跑的后台翻译线程；None 表示当前空闲
        self._worker: TranslateWorker | None = None
        #: 本次翻译的 (引擎名, 源语言, 目标语言)，成功后写进状态栏
        self._translation_meta: Tuple[str, str, str] = ("", "", "")
        #: 断点（翻到一半的进度）存放目录
        self._checkpoint_dir = checkpoint_store.checkpoints_dir()
        #: 「有译文但译文与原文一字不差」的条目下标 —— 手动重试的目标。
        #: 从字幕本身数出来，和用户最终会导出的那份内容一一对应。
        self._stuck_indices: List[int] = []
        #: 这一跑是不是「重试未翻译」；是的话还要记住这轮的目标条数
        self._retry_mode = False
        self._retry_count = 0
        #: 批量队列：一次排入多份字幕，依次翻完并自动导出
        self._output_dir = Path(OUTPUT_DIR)
        self._queue = queue_store.TranslationQueue(self._output_dir)
        #: 当前这一跑是不是队列。是的话：收尾后接着跑下一项，且**不弹模态窗** ——
        #: 队列是人不在场时用的，一个弹窗就能把整条队列卡死在第一个文件上。
        self._queue_mode = False

        # 配置文件缺失不算错误（退回默认值），但 JSON 非法要明确告诉用户。
        self._config_error = ""
        try:
            self._config = load_config()
        except ConfigError as exc:
            self._config = AppConfig()
            self._config_error = str(exc)

        self.setWindowTitle(f"{APP_NAME} — 字幕翻译")
        self.resize(1000, 660)
        self._build_ui()

    # ---------- 界面搭建 ----------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        # 添加字幕只有**一个入口**：队列面板上的「添加文件…」（支持一次多选）。
        # 这里原来还有一个「打开字幕…」按钮，它和队列的添加功能重复，而且只认
        # 单选 —— 同一个动作有两个按钮、行为还不一样，用户得先猜哪个是哪个。
        # 现在这一行只回答「编辑器里现在看的是哪一份」。
        self.path_label = QLabel("未选择文件")

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("当前文件"))
        file_row.addWidget(self.path_label, 1)

        self.source_combo = self._language_combo(
            ("auto", "en", "ja", "ko", "zh-CN"), DEFAULT_SOURCE_LANG
        )
        self.target_combo = self._language_combo(
            ("zh-CN", "en", "ja", "ko"), DEFAULT_TARGET_LANG
        )

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(sorted(ENGINES))

        self.model_selector = ModelSelector()
        self.model_selector.set_source_provider(self._engine_source)
        self.model_selector.set_current_model(self._config.translation.model)

        # 上下文窗口 = 单次请求塞给模型多少条字幕。放在主界面而不是只藏在“设置”里，
        # 是因为它是**出问题时第一个该拧的旋钮**：某几条怎么都翻不动时，把它调到 1，
        # 那几条就能单独重试；平时则调大更省请求数。
        self.context_spin = QSpinBox()
        self.context_spin.setRange(CONTEXT_MIN, CONTEXT_MAX)
        self.context_spin.setValue(self._configured_batch_size())
        self.context_spin.setSuffix(" 条/次")
        self.context_spin.setToolTip(
            "单次请求携带多少条字幕（即“设置”里的批量大小）。\n"
            "调大：请求数更少、更省额度；调小：一批失败的影响面更小，"
            "个别翻不动的条目也更容易被单独拎出来重试。\n"
            "改完立刻对下一次翻译生效；想长期保留就写进“设置”。"
        )

        self.settings_button = QPushButton("设置…")
        self.settings_button.clicked.connect(self._on_settings)

        self.translate_button = QPushButton("翻译")
        self.translate_button.clicked.connect(self._on_translate)
        self.translate_button.setToolTip(
            "只翻译当前显示的这份（不自动保存）—— 用来先看看效果。\n"
            "要把队列里的文件翻完并自动保存，用下面的「依次翻译」。"
        )
        self.translate_button.setEnabled(False)

        # 翻译是分钟级的（上千条字幕要发几十次请求），必须给用户一条退路。
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self._on_cancel)
        self.cancel_button.setToolTip("中止翻译（当前这批请求返回后停止）")
        self.cancel_button.setVisible(False)

        # 自动重试（整批重发 + 逐条重译）都用尽后仍有条目是原文时，才出现。
        self.retry_button = QPushButton("重试未翻译")
        self.retry_button.clicked.connect(self._on_retry_stuck)
        self.retry_button.setVisible(False)

        self.export_button = QPushButton("导出…")
        self.export_button.clicked.connect(self._on_export)
        self.export_button.setEnabled(False)

        # 队列放在编辑器下面：单文件那套用法（打开 → 翻译 → 导出）是主流程，
        # 队列是「一次处理一批」的另一条路，不该挤在主流程中间。
        self.queue_panel = QueuePanel(self._queue)
        self.queue_panel.add_requested.connect(self._on_queue_add)
        self.queue_panel.start_requested.connect(self._on_queue_start)
        self.queue_panel.retry_requested.connect(self._on_retry_all_stuck)
        self.queue_panel.item_selected.connect(self._on_queue_item_selected)
        self.queue_panel.queue_changed.connect(self._on_queue_changed)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("源语言"))
        control_row.addWidget(self.source_combo)
        control_row.addWidget(QLabel("目标语言"))
        control_row.addWidget(self.target_combo)
        control_row.addWidget(QLabel("引擎"))
        control_row.addWidget(self.engine_combo)
        control_row.addWidget(self.settings_button)
        control_row.addStretch(1)
        control_row.addWidget(self.translate_button)
        control_row.addWidget(self.cancel_button)
        control_row.addWidget(self.retry_button)
        control_row.addWidget(self.export_button)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型"))
        model_row.addWidget(self.model_selector, 1)
        model_row.addWidget(QLabel("上下文窗口"))
        model_row.addWidget(self.context_spin)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(
            "点下面「添加文件…」选入字幕后，原文与译文会显示在这里…"
        )

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.config_label = QLabel(self._config_summary())
        self.config_label.setStyleSheet("color: palette(mid);")

        layout = QVBoxLayout(central)
        layout.addLayout(file_row)
        layout.addLayout(control_row)
        layout.addLayout(model_row)
        layout.addWidget(self.config_label)
        layout.addWidget(self.editor, 1)
        layout.addWidget(self.queue_panel)
        layout.addWidget(self.progress)

        # 连接放在最后：引擎一变化就要去改 model_selector，得先保证它已经建好。
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        configured_engine = self._config.translation.engine
        if configured_engine in ENGINES:
            self.engine_combo.setCurrentText(configured_engine)
        # 必须显式刷一次：setCurrentText 设成和当前相同的值**不会发信号**，
        # 光靠信号会让初始状态错掉（例如配置是 echo，模型控件却是可用的）。
        self._on_engine_changed(self.engine_combo.currentText())

    def _config_summary(self) -> str:
        """配置摘要。只报告密钥来源，绝不显示密钥内容。"""
        if self._config_error:
            return f"配置有误：{self._config_error}"
        info = self._config.describe()
        return (
            f"配置 {info['config_file']}　|　引擎 {info['engine']}　|　"
            f"模型 {info['model'] or '(未设置)'}　|　"
            f"地址 {info['base_url'] or '(未设置)'}　|　密钥 {info['key_origin']}"
        )

    def _configured_batch_size(self) -> int:
        """配置里的上下文窗口大小，夹到界面允许的范围内。

        配置是人工写的，可能填 0 或 9999。夹住是为了让控件显示的就是实际生效的
        那个值 —— 显示 200 却按 9999 跑，比显示错更糟。
        """
        try:
            value = int(self._config.translation.batch_size)
        except (TypeError, ValueError):
            return CONTEXT_DEFAULT
        return min(CONTEXT_MAX, max(CONTEXT_MIN, value))

    @staticmethod
    def _language_combo(values: Tuple[str, ...], current: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)  # 允许手填未列出的语言码
        combo.addItems(values)
        combo.setCurrentText(current)
        return combo

    # ---------- 连接参数 ----------

    def _engine_source(self) -> Tuple[str, str, int]:
        """给模型选择器用的 (base_url, api_key, timeout)。"""
        t = self._config.translation
        try:
            key = t.resolve_api_key()
        except ConfigError:
            key = ""
        return t.base_url, key, min(int(t.timeout or 30), MAX_FETCH_TIMEOUT)

    def _refresh_config_label(self) -> None:
        self.config_label.setText(self._config_summary())

    def _on_engine_changed(self, engine_name: str) -> None:
        """按引擎能力决定模型控件是否可用（echo 用不到地址和模型）。"""
        engine = ENGINES.get(engine_name)
        self.model_selector.set_active(bool(getattr(engine, "requires_api", False)))
        self._refresh_config_label()

    def _on_settings(self) -> None:
        engine_name = self.engine_combo.currentText().strip()
        dialog = SettingsDialog(
            self._config,
            requires_api=bool(getattr(ENGINES.get(engine_name), "requires_api", False)),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        updated = dialog.result_config()
        # 引擎是主界面上的选择，对话框不管它
        updated.translation.engine = engine_name
        try:
            saved = save_config(updated)
        except OSError as exc:
            QMessageBox.critical(self, "保存配置失败", str(exc))
            return

        updated.source = saved
        self._config = updated
        self._config_error = ""
        self.model_selector.set_current_model(self._config.translation.model)
        # 设置里改了上下文窗口，主界面上的控件要跟着走 —— 否则用户看的是一个值、
        # 实际跑的是另一个值，界面上还没有任何地方能看出这点。
        self.context_spin.setValue(self._configured_batch_size())
        self._refresh_config_label()
        self.statusBar().showMessage(f"配置已保存：{saved}")

    # ---------- 交互 ----------

    def _on_queue_item_selected(self, index: int) -> None:
        """用户点了队列里的某一项：把那一份摆进编辑器。

        队列是**唯一**的文件列表，所以「现在看哪一份」也由它决定 —— 编辑器里显示的
        始终是列表里被选中的那一份，两者不会打架，用户也就不会在译文区核对一份
        跟他以为的不是同一个的文件。真正耗时的翻译不在这里，所以直接读没问题。
        """
        if self._worker is not None or self._queue_mode:
            return  # 跑的时候列表是锁着的，这里只是兜底
        if not 0 <= index < len(self._queue):
            return
        item = self._queue[index]
        try:
            cues = subtitle_io.parse_file(item.path)
        except (SubtitleFormatError, OSError) as exc:
            QMessageBox.critical(self, "解析失败", f"{item.name}：{exc}")
            return

        item.cue_count = len(cues)
        plan = self._load_cues(item.path, cues)
        # 上次翻到一半的记录要主动说出来：用户不知道有断点，就会以为只能重来。
        if plan.count:
            self.statusBar().showMessage(
                f"{item.name}：{len(cues)} 条　|　上次翻到 {plan.count} 条，"
                "点「继续翻译」接着往下走"
            )
        else:
            self.statusBar().showMessage(f"{item.name}：{len(cues)} 条")

    def _load_cues(
        self, path: str | Path, cues: List[Cue]
    ) -> checkpoint_store.ResumePlan:
        """把一份字幕装进界面。点队列里某一项、队列轮到某一项，走的都是这里。

        返回它的断点计划：调用方需要知道「能复用几条」才能决定要不要续传。
        """
        self._cues = cues
        self._source_path = Path(path)
        self.path_label.setText(f"{path}    （{len(cues)} 条）")
        self.editor.setPlainText(subtitle_io.to_srt(cues))
        self.translate_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.progress.setValue(0)

        # 换了文件，「哪几条没翻出来」就重新算 —— 上一份字幕的结论对这份没有意义。
        self._recompute_stuck()
        self._sync_retry_button()

        # 列表里被选中的那一行必须跟着编辑器走。反过来的方向（点列表 → 载入）由
        # _on_queue_item_selected 负责；这里补的是另一个方向，否则「翻译自己翻完
        # 推进到下一份」之后，列表还高亮着上一份，看起来像翻译翻错了文件。
        row = self._queue.index_of(self._source_path)
        self.queue_panel.select_row(-1 if row is None else row)

        plan = self._resume_plan()
        self._sync_translate_button()
        return plan

    # ---------- 断点续传 ----------

    def _task_model(self, engine_name: str) -> str:
        """当前任务实际会用的模型名。

        不需要 API 的引擎（echo）没有模型概念，记空串 —— 否则配置里残留的
        模型名会让同一份字幕的两次翻译被判成不同任务。
        """
        if not getattr(ENGINES.get(engine_name), "requires_api", False):
            return ""
        return self.model_selector.current_model().strip()

    def _resume_plan(self) -> checkpoint_store.ResumePlan:
        """看这份字幕有没有能接着用的断点。"""
        if not self._cues or self._source_path is None:
            return checkpoint_store.ResumePlan()
        engine_name = self.engine_combo.currentText().strip()
        return checkpoint_store.inspect(
            self._source_path,
            self._cues,
            engine=engine_name,
            model=self._task_model(engine_name),
            source_lang=self.source_combo.currentText().strip(),
            target_lang=self.target_combo.currentText().strip(),
            directory=self._checkpoint_dir,
        )

    def _sync_translate_button(self) -> None:
        """按钮文案跟着状态走：有断点时叫「继续翻译」，否则叫「翻译」。"""
        plan = self._resume_plan()
        self.translate_button.setText("继续翻译" if plan.count else "翻译")

    def _ask_resume(self, plan: checkpoint_store.ResumePlan) -> str:
        """问用户要不要接着上次翻。返回 ``resume`` / ``restart`` / ``cancel``。

        必须是三选而不是两选：既然做不了主（用户可能刚换了模型想整份重来），
        就不能把「继续」做成唯一的出路，「取消」也不该和「重新开始」挤在一起。
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("继续上次的翻译")
        box.setText(f"这份字幕上次翻到一半：已译 {plan.count} / {len(self._cues)} 条。")
        box.setInformativeText(
            "继续：只翻译剩下的条目，接着上次的译文往下走（省时间也省额度）。\n"
            "重新开始：丢弃这份记录，整份字幕重译一遍。"
        )
        resume_button = box.addButton("继续", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("重新开始", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(resume_button)
        box.exec()

        clicked = box.clickedButton()
        if clicked is resume_button:
            return "resume"
        if clicked is None or clicked.text() == "取消":
            return "cancel"
        return "restart"

    def _refresh_checkpoint(self) -> bool:
        """按当前整份字幕刷新断点记录；返回是否真的刷新了。

        记录不存在、已经不属于当前任务（换了模型等）、或写盘失败时返回 False ——
        这三种情况都不该拦住翻译本身：前两种刷新了反而会毁掉一份还有用的记录，
        第三种最坏只是让用户在续传时多翻几条。
        """
        if self._source_path is None:
            return False
        plan = self._resume_plan()
        if not plan.exists or plan.rejected:
            return False
        engine_name, source_lang, target_lang = self._translation_meta
        try:
            return checkpoint_store.save(
                plan.path,
                self._cues,
                source_path=self._source_path,
                engine=engine_name,
                model=self._task_model(engine_name),
                source_lang=source_lang,
                target_lang=target_lang,
            ) is not None
        except OSError:
            return False

    # ---------- 未翻译条目的手动重试 ----------

    def _recompute_stuck(self) -> None:
        """重算「该翻译、但没能翻出来」的条目。

        判据就是 ``cue.failed`` —— 由翻译层在写回结果时统一打上
        （见 ``translator._apply_results``），这里不再重判一遍。

        不在这里重判的理由不是「慢」，而是**两处判据会漂移**：那边认定没翻、
        这边换了身标准又认为翻过了，用户就会看到「按钮上说有 3 条，点下去一条
        都没重试」。让「谁没翻出来」只有一个出口。
        """
        if not self._cues:
            self._stuck_indices = []
            return
        self._stuck_indices = [
            index for index, cue in enumerate(self._cues) if cue.failed
        ]

    def _sync_retry_button(self) -> None:
        """按钮只在真有几条没翻出来时出现；忙的时候收起来。

        平时不该占着位置 —— 一个长期灰着的按钮只会让人猜它什么时候能用。
        """
        count = len(self._stuck_indices)
        # 队列跑的时候一律收起来：那时「未翻译」是按文件记的（在队列列表里显示），
        # 按钮上再报一个总数只会让人不知道它指的是哪一份。
        show = count > 0 and self._worker is None and not self._queue_mode
        self.retry_button.setVisible(show)
        self.retry_button.setEnabled(show)
        self.retry_button.setText(f"重试未翻译（{count}）")
        self.retry_button.setToolTip(
            "只把这几条重新发一遍，不动已经翻好的部分。\n"
            "它们在引擎里已经自动重试过三次，原样再试往往没用 ——"
            "建议先换个模型、或把上下文窗口调小（例如 1 条/次），再点这里。"
            if show
            else ""
        )

    def _on_retry_stuck(self) -> None:
        """只重翻那几条怎么都翻不动的条目。

        和引擎内部的自动重试不是一回事：那两步（整批重发、逐条重译）用的是**同一份**
        请求参数，撞一百次也可能还是同样的结果。这里允许用户先换模型、或把上下文窗口
        调到很小，再点一次 —— 这才是能真正改变结局的那一手。
        """
        if self._worker is not None or not self._stuck_indices:
            return

        count = len(
            [
                index
                for index in self._stuck_indices
                if 0 <= index < len(self._cues)
            ]
        )
        if not count:
            self._recompute_stuck()
            self._sync_retry_button()
            return

        # 这几条的译文已经被翻译层置空并标了 failed（导出时显示 [未翻译]）。
        # 重试就是把它们重新送去翻；翻不出来标记就留着 —— 中途取消也不会退化成
        # 「看不出没翻」的原文。
        #
        # ⚠️ 重试传的是 **self._cues 这份整份字幕** —— _start_translation 已经没有
        # 「传哪几条」这个入口了（曾经有；传子集会把成品截断，见那里的说明）。
        # 只重翻那几条由 skip_translated 实现。
        reason = self._start_translation(
            self.engine_combo.currentText().strip(),
            self.source_combo.currentText().strip(),
            self.target_combo.currentText().strip(),
            skip_translated=True,
            note=f"重试 {count} 条未翻译的字幕",
            retry=True,
        )
        if reason:
            QMessageBox.critical(self, "翻译失败", reason)

    def _ask_retry_stuck(self, count: int, *, retried: bool = False) -> bool:
        """问要不要现在就重试这几条；返回 True 表示用户选了重试。

        单独拆成一个方法是为了能在测试里替换掉 —— 真实弹窗会阻塞事件循环，
        自动化测试里没法点它。
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("部分字幕没有翻译")
        box.setText(
            f"有 {count} 条字幕自动重试三次后仍然没翻出来，"
            "译文里已经标成 [未翻译]。"
        )
        box.setInformativeText(
            ("刚刚的手动重试也没能翻出来。"
             if retried
             else "")
            + "这几条多半是模型在偷懒（同一模型名下请求被分发到能力不齐的通道），"
            "隔一会儿重发往往就好了。\n\n"
            "若连续几次都不行，请先换个模型、或把上下文窗口调小"
            "（例如 1 条/次）再点「重试这些条目」。\n"
            "也可以先关掉本窗口，在译文区按 [未翻译] 找到它们人工处理。"
        )
        retry_button = box.addButton("重试这些条目", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("我自己核对", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(retry_button)
        box.exec()
        return box.clickedButton() is retry_button

    def _warn_untranslated(self, count: int, *, retried: bool) -> None:
        """把「还有几条纹丝不动」摆到台面上，并给一条能真正改变结局的出路。

        回抄是**静默**故障：不报告就没人会发现手里那份字幕根本没翻。
        """
        if self._ask_retry_stuck(count, retried=retried):
            self._on_retry_stuck()

    def _start_translation(
        self,
        engine_name: str,
        source_lang: str,
        target_lang: str,
        *,
        skip_translated: bool,
        note: str = "",
        retry: bool = False,
        output_path: Path | None = None,
    ) -> str:
        """构造引擎与后台线程并启动。断点相关的前置判断已经在外面做完了。

        返回**空串表示已启动**；非空是启动失败的原因（同时也写进了状态栏）。
        错误不在这里弹窗，是因为队列模式下一次要跑几十个文件：一路弹模态框会把
        整条队列卡死在第一个文件上，而队列本来就是给「人不在场」用的。
        弹不弹由调用方决定 —— 它才知道这次是单文件还是队列。

        ``note`` 是启动时顺带要告诉用户的一句话（比如「上次的记录用不上」）。
        它必须由这里一起写进状态栏 —— 在外面先写会被下面的「翻译中…」覆盖掉。

        ``retry`` 标记这是一轮「补重试」。**重试照样传整份字幕** —— 现在连传子集的
        入口都没有了：「只重翻哪几条」一律由 ``skip_translated`` 决定（曾经有个
        ``cues`` 参数可以传子集，那个 bug 把成品截断过，见下面对 ``output_path``
        的说明）。不是「记得别传」，而是压根没得传。

        重试路径**不写断点** —— 断点记录用 ``enumerate`` 的下标给条目定位，
        而重试是一个「补丁」轮次，不值得为它冒把 A 条的译文记到 B 条名下的风险。
        整份任务此时已经完成，重试丢了大不了再点一次。

        ``output_path`` 是**边翻边写**的目标。队列传自己算好的那个（入队时就定死，
        用户开跑前就能在列表里看到）；单文件走默认落点，见 :meth:`_default_output`。

        ⚠️ 落盘走的是 :class:`OutputSink`，它**每次整份重写**、拿到什么写什么。
        所以这里出去的一律是**整份**字幕：一旦只传其中几条，成品就会被「此刻只有
        这几条」覆盖掉 —— 已经翻好、用户可能还核对过的行**当场退回原文**，
        而且无声无息，连 ``[未翻译]`` 标记都不会有。实测过，这个 bug 是静默的。
        """
        # 界面上临时选的模型要即时生效，否则会悄悄沿用配置文件里的旧值。
        # 想持久化就走「设置…」。
        requires_api = bool(getattr(ENGINES.get(engine_name), "requires_api", False))
        if requires_api:
            self._config.translation.model = self.model_selector.current_model()

        # 引擎构造留在主线程：它只读配置、不发网络请求，出错时能同步报出来。
        try:
            engine = create_engine_for(engine_name, self._config)
        except (TranslationError, ConfigError) as exc:
            reason = str(exc)
        except Exception as exc:  # 第三方后端可能抛出任意异常，不能让它掀掉界面
            reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = ""
        if reason:
            self.statusBar().showMessage(f"翻译失败：{reason}")
            return reason

        writer = None
        if not retry and self._source_path is not None:
            writer = checkpoint_store.CheckpointWriter(
                checkpoint_store.checkpoint_path(
                    self._source_path, directory=self._checkpoint_dir
                ),
                source_path=self._source_path,
                engine=engine_name,
                model=self._task_model(engine_name),
                source_lang=source_lang,
                target_lang=target_lang,
            )

        # 译文落到哪个文件。重试路径也照写 —— 重试的目的就是把那几条补上，
        # 补完不落盘等于白跑。
        if output_path is None and self._source_path is not None:
            output_path = self._default_output(self._source_path)
        sink = OutputSink(output_path) if output_path is not None else None

        # 永远是**整份**字幕：sink 每次整份重写，传子集会把成品截断（见上）。
        # 这里没有「传哪几条」的参数，就是为了让那个 bug 写不出来。
        targets = list(self._cues)

        # 真正耗时的部分必须进后台线程：上千条字幕要发几十次请求、跑好几分钟，
        # 放在主线程里窗口会整个冻住（进度条不动、取消都点不了）。
        self._translation_meta = (engine_name, source_lang, target_lang)
        self._retry_mode = retry
        # 报的是「这一轮要补几条」。不能用 len(targets)：那是**整份**字幕
        # （见上），拿它当分母会把「补 1 条」说成「补 3 条」。
        self._retry_count = (
            sum(1 for cue in targets if not cue.is_translated) if retry else 0
        )
        worker = TranslateWorker(
            engine,
            targets,
            source_lang=source_lang,
            target_lang=target_lang,
            # 界面上的上下文窗口永远说了算：它比配置里的值更“新”，
            # 而且用户就是冲着「把这一批切小点」才去动它的。
            batch_size=int(self.context_spin.value()),
            writer=writer,
            sink=sink,
            skip_translated=skip_translated,
            parent=self,
        )
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(self._on_translate_succeeded)
        worker.cancelled.connect(self._on_translate_cancelled)
        worker.failed.connect(self._on_translate_failed)
        self._worker = worker  # 保留引用：线程被回收会连带丢掉信号连接
        self._set_busy(True)
        total = len(targets)
        # 进度按「整份里已经翻好多少」算。重试轮也一样 —— 那一轮传的是整份字幕，
        # 翻好的部分确实已经在那儿了，进度条落在它上面才诚实（见上面 cues 的说明）。
        already = sum(1 for cue in targets if cue.is_translated)
        # 续传时进度条立刻落在断点处，不干等第一批请求回来 —— 否则用户会以为
        # 之前的进度丢了，或者以为又在从头翻。
        self.progress.setValue(
            0 if not already or not total else int(already / total * 100)
        )
        if retry:
            message = f"重试未翻译的条目…（共 {self._retry_count} 条）"
        elif skip_translated and already:
            message = f"接着上次翻译…（还剩 {total - already} 条，共 {total} 条）"
        else:
            message = f"翻译中…（共 {total} 条）"
        self.statusBar().showMessage(f"{note}　|　{message}" if note else message)
        worker.start()
        return ""

    def _default_output(self, source: Path) -> Path:
        """一份字幕的默认译文落点。

        与队列同一规则、同一目录（**统一输出目录**）：``<名字>.translated.srt``。
        放到源文件旁边看着方便，但把用户的素材目录塞满 ``*.translated.srt``
        是另一种麻烦 —— 剧集目录里几十个文件铺开，谁是最新的反而看不出来。
        """
        return self._output_dir / f"{source.stem}.translated{source.suffix.lower()}"

    @staticmethod
    def _restore_previous_round(source_cues: List[Cue], output_path: Path) -> int:
        """把上一轮成品里的**译文**与「没翻出来」的标记搬回整份字幕，返回没翻出来几条。

        为什么必须把译文也搬回来：重试轮是重新解析**源文件**拿到整份字幕的
        （引擎要的就是原文），此时 ``cue.translation`` 全是空的。若就这么交给下游，
        :class:`OutputSink` 每次整份重写，成品里那些早就翻好、用户可能已经核对过的
        行会**全部退回原文** —— 而且是无声的，连 ``[未翻译]`` 标记都不会有。
        上一轮的成品里两样东西都在：译文就是译文，没翻出来的写着 :data:`UNTRANSLATED_MARK`。

        ``failed`` 这个标记**不落进源文件**（我们不改用户的素材），所以重新解析
        源文件时它是丢的 —— 那个标记就是为这种时刻准备的。

        条数与顺序都与源文件一一对应（两边都是同一份字幕），所以按下标对齐即可。
        成品不存在或读不了时返回 0 —— 那说明这一轮确实没什么可重试的。
        """
        if not output_path.exists():
            return 0
        try:
            previous = subtitle_io.parse_file(output_path)
        except (SubtitleFormatError, OSError):
            return 0
        restored = 0
        for index, cue in enumerate(source_cues):
            if index >= len(previous):
                break
            body = previous[index].text.strip()
            if body == UNTRANSLATED_MARK:
                cue.translation = ""
                cue.failed = True
                restored += 1
            elif body and body != cue.text.strip():
                # 真的翻过：接回来。与原文一字不差的**不接** —— 那可能只是
                # 本来就无需翻译的符号行/数字行，接进来会被当成「已翻好」而漏掉；
                # 留给 skip_translated 再送一次没有代价（它本来就判不出没翻）。
                cue.translation = body
                cue.failed = False
        return restored

    def _on_translate(self) -> None:
        """启动翻译。**立即返回** —— 耗时的部分在后台线程里跑。"""
        # 队列跑着的时候不接受单文件翻译：那时界面上的「当前文件」是队列临时摆上来的，
        # 用户再按一次「翻译」会让两条流程同时往同一批 cues 里写。
        if not self._cues or self._worker is not None or self._queue_mode:
            return
        engine_name = self.engine_combo.currentText().strip()
        source_lang = self.source_combo.currentText().strip()
        target_lang = self.target_combo.currentText().strip()

        # 有断点就先问一句：接上去还是从头来，这个决定只能由用户做。
        plan = self._resume_plan()
        skip_translated = False
        note = ""
        if plan.count:
            choice = self._ask_resume(plan)
            if choice == "cancel":
                self.statusBar().showMessage(f"已取消（保留着 {plan.count} 条的进度）")
                return
            if choice == "resume":
                # 把上次的译文先填回 cue，后台线程只补没翻的部分。
                for index, text in plan.usable.items():
                    self._cues[index].translation = text
                skip_translated = True
        elif plan.rejected:
            # 不能静默失效：用户以为在接着翻，实际从头开始了，很难察觉。
            note = f"已忽略上次的翻译记录：{plan.rejected}"

        reason = self._start_translation(
            engine_name,
            source_lang,
            target_lang,
            skip_translated=skip_translated,
            note=note,
        )
        if reason:
            QMessageBox.critical(self, "翻译失败", reason)

    # ---------- 批量队列 ----------

    def _on_queue_add(self) -> None:
        """一次选多个文件排进队列。"""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "添加字幕文件（可一次选多个）",
            "",
            "字幕文件 (*.srt *.vtt);;所有文件 (*)",
        )
        if not paths:
            return
        self._enqueue(paths)

    def _enqueue(self, paths: Sequence[str | Path]) -> None:
        """把文件排进队列：入队 + 试读体检 + 刷界面。

        与弹对话框分开，是为了让「一次选二十个文件，其中两个坏了」这条路径
        能被自动化测试直接走到 —— 只靠对话框进不去。
        """
        before = len(self._queue)
        added, notes = self._queue.add(paths)
        # 入队时先解析试读一遍，读不了或格式不对的当场剔掉。
        # 队列跑起来时人不在场 —— 等轮到它才发现「这份根本读不了」就太晚了，
        # 整条队列会停在一个用户以为没问题的文件上。
        kept: List[Tuple[queue_store.QueueItem, List[Cue]]] = []
        for item in list(added):
            try:
                cues = subtitle_io.parse_file(item.path)
            except (SubtitleFormatError, OSError) as exc:
                self._queue.remove(item)
                notes.append(f"{item.path.name}：{exc}")
            else:
                item.cue_count = len(cues)
                kept.append((item, cues))

        # 先刷列表再载入：`_load_cues` 会把列表的选中行对齐到它显示的那一份，
        # 那时候新条目必须已经在列表里了，否则行号落在列表外面、等于没选中。
        self.queue_panel.refresh()
        # 编辑器空着的时候顺手把第一份摆上去，用户立刻看到刚加的是什么（不用先去
        # 点列表）。已经有内容在看就**不动它** —— 换掉编辑器等于把那一份尚未导出的
        # 译文从内存里抹掉，而翻译成功后会清掉断点，抹掉就真找不回来了。
        if kept and self._source_path is None:
            first_item, first_cues = kept[0]
            self._load_cues(first_item.path, first_cues)

        counts = self._queue.summary()
        message = (
            f"已加入 {len(self._queue) - before} 个文件，队列共 {counts['total']} 个"
        )
        if counts["cue_count"]:
            message += f"（{counts['cue_count']} 条字幕）"
        if notes:
            message += "　|　跳过 " + "；".join(self._brief(notes))
        self.statusBar().showMessage(message)

    @staticmethod
    def _brief(notes: Sequence[str], limit: int = 3) -> List[str]:
        """把跳过的原因压成不超过 ``limit`` 条。

        一次选三十个文件全被跳过时，状态栏塞不下、也没人读得了那么长一句。
        """
        if len(notes) <= limit:
            return list(notes)
        return [*notes[:limit], f"等共 {len(notes)} 条"]

    def _on_queue_changed(self) -> None:
        """队列被增删移之后同步一句提示（面板自己已经把列表重绘过了）。"""
        # 被移出的可能正是编辑器里显示的那一份：把选中行重新对齐到编辑器，
        # 免得列表高亮着一份、译文区显示着另一份。
        row = self._queue.index_of(self._source_path)
        self.queue_panel.select_row(-1 if row is None else row)

        counts = self._queue.summary()
        if not counts["total"]:
            self.statusBar().showMessage("队列已清空")
        else:
            self.statusBar().showMessage(
                f"队列共 {counts['total']} 个文件、{counts['cue_count']} 条字幕"
            )

    def _on_queue_start(self) -> None:
        """开始（或继续）依次翻译。"""
        if self._worker is not None or self._queue_mode or not len(self._queue):
            return
        counts = self._queue.summary()
        if counts["unfinished"] == 0:
            # 全都成功过了：再点只能是「整份重来」。这个决定必须用户明确做 ——
            # 它会覆盖输出目录里已有的译文。
            if not self._ask_queue_restart():
                return
            self._queue.reset_all()
        else:
            # 上一轮失败、或没轮到的（用户中途取消）可以重来；
            # 已经翻好的不会白跑第二遍。
            self._queue.reset_unfinished()
        self._queue_mode = True
        self.queue_panel.refresh()
        self.queue_panel.set_busy(True)  # 立刻锁住，别等第一个文件启动
        self._run_next_queue_item()

    def _ask_queue_restart(self) -> bool:
        """队列里每一份都已经成功了，再点「依次翻译」只能是想整份重来。

        单独拆成一个方法是为了能在测试里替换掉 —— 真实弹窗会阻塞事件循环。
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("重新翻译全部")
        box.setText(f"队列里 {len(self._queue)} 个文件都已经翻译过了。")
        box.setInformativeText("重新翻译会覆盖输出目录里已有的译文。")
        again = box.addButton("重新翻译全部", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is again

    def _run_next_queue_item(self) -> None:
        """找下一个待翻项并启动；没有了就收尾。

        用循环而不是递归：一整个队列的文件都读不了时，递归会一层层压进调用栈，
        文件多了能把栈吃满 —— 而「每个文件都在失败」恰恰最容易发生在一批本身
        就有问题的时候。
        """
        while True:
            index = self._queue.next_pending()
            if index is None:
                self._finish_queue()
                return
            if self._start_queue_item(index):
                return

    def _start_queue_item(self, index: int) -> bool:
        """把队列第 ``index`` 项摆到界面上并开始翻译。返回是否真的启动了。"""
        item = self._queue[index]
        try:
            cues = subtitle_io.parse_file(item.path)
        except (SubtitleFormatError, OSError) as exc:
            self._queue.fail(index, f"解析失败：{exc}")
            self.queue_panel.refresh()
            return False

        item.cue_count = len(cues)
        self._queue.mark_running(index)
        # 正在翻的这一份要显示出来：用户随时能看一眼进度，也能在译文区核对。
        plan = self._load_cues(item.path, cues)

        note = f"队列 {index + 1}/{len(self._queue)}"

        if item.retry_only:
            # 「重试未翻译」轮：只把没翻出来的那几条重新送去翻。
            #
            # ⚠️ 交给翻译层的必须是**整份**字幕（_start_translation 只能拿到
            # self._cues，没法传子集）。译文落盘走 OutputSink，它每次整份重写、
            # 拿到什么写什么；只传子集的话，成品会被「此刻只有这几条」覆盖掉 ——
            # 上一轮翻好、用户可能已经核对过的行当场蒸发，而且无声无息
            # （连 [未翻译] 标记都不会有）。
            # 「只翻那几条」交给 skip_translated：先把上一轮的成果接回来，
            # 于是剩下的空条目正好就是没翻出来的那几条。
            #
            # 标记不在源文件里（我们不改用户的素材），但**上一轮的成品里有**：
            # [未翻译] 就是给这种时刻用的。
            restored = self._restore_previous_round(cues, item.output_path)
            if not restored:
                # 确实没有可重试的了（成品里已经找不到 [未翻译]）。
                # 直接算完成，别往引擎里塞一个空批次。
                self._queue.mark_done(index)
                self.queue_panel.refresh()
                return False
            reason = self._start_translation(
                self.engine_combo.currentText().strip(),
                self.source_combo.currentText().strip(),
                self.target_combo.currentText().strip(),
                skip_translated=True,
                note=f"{note}　重试未翻译（{restored} 条）",
                retry=True,
                output_path=item.output_path,
            )
            if reason:
                self._queue.fail(index, reason)
                self.queue_panel.refresh()
                return False
            self.queue_panel.refresh()
            return True

        skip = False
        if plan.count:
            # 队列模式**不弹那个三选窗**：人不在场，弹窗会把整条队列卡死在第一个
            # 文件上。有断点就默认接着翻 —— 这正是用户点「继续翻译」会选的那个。
            for position, text in plan.usable.items():
                self._cues[position].translation = text
            skip = True
            item.resumed = plan.count
            note += f"　接着上次翻（已译 {plan.count} 条）"
        elif plan.rejected:
            note += f"　上次的记录用不上：{plan.rejected}"

        reason = self._start_translation(
            self.engine_combo.currentText().strip(),
            self.source_combo.currentText().strip(),
            self.target_combo.currentText().strip(),
            skip_translated=skip,
            note=note,
            output_path=item.output_path,
        )
        if reason:
            self._queue.fail(index, reason)
            self.queue_panel.refresh()
            return False
        self.queue_panel.refresh()
        return True

    def _finish_queue_item(self, outcome: str, error: str = "") -> None:
        """一个文件跑完了：记账、导出、接着下一个。"""
        index = self._queue.running_index
        if index is None:
            # 状态机不该走到这里。真走到了就干净地收尾，
            # 别把队列永远卡在「翻译中」那个状态上。
            self._queue_mode = False
            self.queue_panel.refresh()
            self._finish_queue()
            return

        item = self._queue[index]
        if outcome == "done":
            item.untranslated = len(self._stuck_indices)
            try:
                # 自动导出是队列的关键一环：无人值守时没人来点「导出…」，
                # 不落盘就等于白跑 —— 翻完的一刻就是唯一该保存的时刻。
                item.output_path = subtitle_io.write_file(item.output_path, self._cues)
            except OSError as exc:
                # 翻好了却存不下来 —— 这一份算失败：没有产物，等于白跑。
                self._queue.fail(index, f"导出失败：{exc}")
            else:
                self._queue.mark_done(index)
        else:
            self._queue.fail(index, error)

        self.queue_panel.refresh()
        self._run_next_queue_item()

    def _finish_queue(self, cancelled: bool = False, note: str = "") -> None:
        """整条队列收尾：写状态栏，有必要时把明细摆出来。"""
        self._queue_mode = False
        # 队列真的结束了，面板这才解除忙碌 —— 否则「重试未翻译」按钮会被
        # _set_busy 那一刻的 _queue_mode（那时还挂着「队列在跑」）永久挡住。
        self.queue_panel.set_busy(False)
        counts = self._queue.summary()
        total, done = counts["total"], counts["done"]
        summary = (
            f"队列已停止：{done}/{total} 个文件已完成"
            if cancelled
            else f"队列完成：{done}/{total} 个文件已翻译并保存"
        )
        if counts["failed"]:
            summary += f"　|　{counts['failed']} 个失败"
        if counts["untranslated"]:
            summary += (
                f"　|　共 {counts['untranslated']} 条未翻译"
                "（可点「重试未翻译」，或先换个模型）"
            )
        if counts["resumed"]:
            summary += f"　|　沿用了上次的 {counts['resumed']} 条"
        self.statusBar().showMessage(f"{note}　|　{summary}" if note else summary)

        self.queue_panel.refresh()
        if not cancelled and (counts["failed"] or counts["untranslated"]):
            self._report_queue()

    def _report_queue(self) -> None:
        """队列收尾的明细。

        只在状态栏留一句「2 个失败」等于没说 —— 用户回来时得知道是哪两个、
        为什么、以及接下来该点什么。单独拆成一个方法便于测试替换
        （真实弹窗会阻塞事件循环，自动化里点不到它）。
        """
        failed = [item for item in self._queue if item.status == queue_store.FAILED]
        stuck = [
            item
            for item in self._queue
            if item.status == queue_store.DONE and item.untranslated
        ]
        lines: List[str] = []
        if failed:
            lines.append("以下文件没能翻完，点「继续队列」可以重试：")
            lines.extend(
                f"　• {item.name}：{item.one_line_error(120)}" for item in failed
            )
        if stuck:
            if lines:
                lines.append("")
            lines.append("以下文件翻完了，但有条目没能翻译（译文里标着 [未翻译]）：")
            lines.extend(
                f"　• {item.name}：{item.untranslated} 条" for item in stuck
            )
            lines.append("")
            lines.append(
                "这些条目在引擎里已经自动重试过三次。可以换个模型、"
                "或把上下文窗口调小，再点面板上的「重试未翻译」。"
            )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("队列结束")
        if failed and not stuck:
            headline = f"{len(failed)} 个文件失败"
        elif stuck and not failed:
            headline = f"{len(stuck)} 个文件里还有未翻译的条目"
        else:
            headline = "队列跑完了，但有文件需要你看一眼"
        box.setText(headline)
        box.setInformativeText("\n".join(lines))
        box.addButton("知道了", QMessageBox.ButtonRole.AcceptRole)
        box.exec()

    def _on_retry_all_stuck(self) -> None:
        """整条队列跑完后，把各文件里没翻出来的条目一起再试一遍。

        为什么非等到整条队列结束才给这个入口：那些条目在引擎里已经自动重试过
        三次，同一条通道、同一套参数，紧接着再来一遍基本还是同样的结果。等队列
        跑完才有意义 —— 用户这时有空换模型、把关卡（上下文窗口）调小，
        一次改动对所有文件生效。

        推进方式与队列完全一致（依次跑、一份一份来），所以它同样不弹模态窗、
        也不会因为某个文件出问题就停下来。
        """
        if self._worker is not None or self._queue_mode or not len(self._queue):
            return
        count = self._queue.mark_retry_round()
        if not count:
            self.statusBar().showMessage("没有需要重试的条目")
            return
        self._queue_mode = True
        self.queue_panel.refresh()
        self.queue_panel.set_busy(True)
        self._run_next_queue_item()

    # ---------- 后台线程的回调（信号跨线程排队投递，槽仍在主线程执行） ----------

    @staticmethod
    def _sink_note(error: str) -> str:
        """写译文文件失败要单独说一句。

        用户以为成品正一路落盘、实际一个字节都没写进去 —— 属于「不特意说一声
        就永远不会知道」的故障，比翻译本身失败更该报出来。
        """
        return f"　|　译文文件未能保存（{error}）" if error else ""

    def _on_translate_succeeded(self) -> None:
        worker = self._worker
        engine = worker.engine if worker is not None else None
        checkpoint_path = worker.checkpoint_path if worker is not None else None
        output_error = worker.output_error if worker is not None else ""
        retry_mode, retry_count = self._retry_mode, self._retry_count
        # 收尾里会重算「还剩哪几条没翻出来」——它既是弹窗里的数字，也是重试的目标。
        self._finish_translation()

        if not retry_mode:
            # 整份都翻完了，断点就没了意义。留着还会在下次打开同一份字幕时
            # 冒出一句「上次翻到 N 条」—— 那会把已经完成的任务说成没做完。
            checkpoint_store.clear(checkpoint_path)
            self._sync_translate_button()
        else:
            # 手动重试是在一份「只翻到一半」的字幕上打的补丁，而断点记录里那几条
            # 还是旧的**回抄值**（等于原文）。下次点「继续翻译」时，续传会拿记录里的
            # 值无条件覆盖 cue.translation —— 用户刚花钱花时间重试出来的结果会
            # 悄无声息地退回原文，界面上什么都看不出来。所以记录还在、且仍属于
            # 当前任务时，顺手按整份字幕刷一遍（写的是完整 cues，下标正确，
            # 与「重试本身不写子集断点」并不矛盾）。
            self._refresh_checkpoint()

        self.editor.setPlainText(subtitle_io.to_srt(self._cues))
        remaining = len(self._stuck_indices)
        engine_name, source_lang, target_lang = self._translation_meta
        # 未翻译条数由界面自己数（见 quality_notes 的说明），别和引擎各报一个数。
        notes = (
            engine.quality_notes(include_untranslated=False) if engine is not None else []
        )
        if retry_mode:
            summary = f"重试完成：{retry_count - remaining}/{retry_count} 条已翻好"
        else:
            summary = f"{engine_name} 翻译完成：{source_lang} → {target_lang}"
        # 队列模式下一份一份地报「仍有 N 条未翻译」＝ 每跑完一个文件就提示一次。
        # 但未翻译是**整条任务列表**的事：那些条目在引擎里已经自动重试过三次，
        # 要等整条队列跑完，用户才有空换模型、调小上下文窗口，一次改动对所有文件
        # 生效。中途喊一路，只会让人以为要一份一份去处理 —— 而且那时队列还在跑，
        # 「重试未翻译」按钮按设计根本不给点。
        # 这一份有几条没翻出来在队列列表那一行上写着（`_describe`），不丢信息；
        # 总数由 `_finish_queue` 在整条队列结束时统一报。
        if remaining and not self._queue_mode:
            summary += f"　|　仍有 {remaining} 条未翻译（可点「重试未翻译」）"
        if notes:
            summary += "　|　" + "；".join(notes)
        summary += self._sink_note(output_error)
        self.statusBar().showMessage(summary)

        # 回抄是**静默**故障：不报告就没人会发现手里那份字幕根本没翻。
        # 救回来了在状态栏记账；没救回来的必须弹出来，并当场给一条手动重试的出路。
        # 队列模式例外：那时人不在场，弹窗只会把队列卡在这儿，改成记在该项上、
        # 等整条队列跑完一起报（见 _report_queue）。
        if remaining and not self._queue_mode:
            self._warn_untranslated(remaining, retried=retry_mode)

        if self._queue_mode:
            self._finish_queue_item("done")

    def _on_translate_cancelled(self, done: int, total: int) -> None:
        worker = self._worker
        error = worker.checkpoint_error if worker is not None else ""
        sink_error = worker.output_error if worker is not None else ""
        retry_mode = self._retry_mode
        self._finish_translation()
        # 已翻好的部分留着：用户能核对或导出半成品，也不必从头再来。
        self.editor.setPlainText(subtitle_io.to_srt(self._cues))
        message = (
            f"已取消重试：{done}/{total} 条已处理"
            if retry_mode
            else f"已取消：{done}/{total} 条已翻译"
        )
        if error:
            # 断点没写成功要说出来：用户以为能续传、实际只能重来，比不给断点更糟。
            message += f"　|　进度未能保存（{error}），下次需要重来"
        elif done and retry_mode:
            # 重试不写断点（见 _start_translation 的说明），别谎称“进度已保存”。
            message += "　|　已翻好的部分留在译文区"
        elif done:
            message += "　|　进度已保存，下次可从中断处继续"
        message += self._sink_note(sink_error)

        if self._queue_mode:
            # 队列里点「取消」= 停下整条队列。当前这一份打回「待翻译」，断点还在，
            # 下次点「依次翻译」就从它接着往下走，已经翻好的文件不会白跑第二遍。
            index = self._queue.running_index
            if index is not None:
                self._queue.reset(index)
            self._finish_queue(cancelled=True, note=message)
            return

        self.statusBar().showMessage(message)
        self._sync_translate_button()

    def _on_translate_failed(self, kind: str, message: str) -> None:
        worker = self._worker
        resumable = (
            worker is not None
            and worker.checkpoint_path is not None
            and worker.done > 0
        )
        sink_error = worker.output_error if worker is not None else ""
        self._finish_translation()
        text = message if kind == "TranslationError" else f"{kind}: {message}"
        if resumable:
            text += (
                "\n\n已翻好的部分已经存下来了，处理完问题点「继续翻译」就能接着来。"
            )
        if sink_error:
            text += f"\n\n译文文件也没能写出来：{sink_error}"

        if self._queue_mode:
            # 队列里失败不回弹窗：人不在场，弹一个整条队列就再也走不下去了。
            # 记在该项上、跳过它继续下一个（用户的明确选择），最后一起汇报。
            self._finish_queue_item("failed", text)
            self._sync_translate_button()
            return

        QMessageBox.critical(self, "翻译失败", text)
        self.statusBar().showMessage("翻译失败")
        self._sync_translate_button()

    def _on_cancel(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.statusBar().showMessage("正在取消…（等当前这批请求返回）")

    def _finish_translation(self) -> None:
        """收尾：回收线程、恢复控件。成功 / 取消 / 失败三条路径都要走这里。"""
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait(CLOSE_WAIT_MS)  # 信号已到主线程，线程此时基本已退出
            worker.deleteLater()  # worker 是窗口的子对象，不显式回收会随翻译次数累积
        # 重算「还有哪几条没翻出来」：它决定了重试按钮出不出现、弹窗报几。放在
        # _set_busy 之前，按钮状态才跟得上（_sync_retry_button 也要看 _worker 是不是空）。
        self._recompute_stuck()
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        """翻译期间锁住会改变任务输入或结果的控件。

        不锁的话用户能在翻译跑着时换字幕文件 —— 后台线程还在往旧对象里写译文，
        界面最终显示一份与当前文件对不上的结果，而且看不出哪里错了。

        上下文窗口同样要锁：它决定这一跑怎么切批，中途改只会让「已经翻到哪」
        变得说不清，而界面上的那个数字又给不出任何提示。
        """
        busy = bool(busy)
        # 取消按钮只在忙碌时出现；每次重新进入忙碌都要把它的禁用状态复位，
        # 否则「取消」过一次之后，下一轮翻译的取消按钮点不动。
        self.cancel_button.setVisible(busy)
        if busy:
            self.cancel_button.setEnabled(True)
        self.settings_button.setEnabled(not busy)
        self.engine_combo.setEnabled(not busy)
        self.source_combo.setEnabled(not busy)
        self.target_combo.setEnabled(not busy)
        self.model_selector.setEnabled(not busy)
        self.context_spin.setEnabled(not busy)
        # 队列模式下「翻译」一直灰着：那时界面上的当前文件是队列摆上来的，
        # 单文件翻译和队列同时跑会让两条流程往同一批 cues 里写。
        self.translate_button.setEnabled(
            not busy and not self._queue_mode and bool(self._cues)
        )
        self.export_button.setEnabled(not busy and bool(self._cues))
        # 面板的「忙碌」按**整条队列**算，不是按单个文件算：队列在两个文件之间会
        # 短暂退出忙碌（前一个 worker 收尾、下一个还没启动），那一瞬间「重试未翻译」
        # 按钮就会冒出来又缩回去 —— 显示的是刚跑完那一份的条数，点它又没有反应
        # （队列还在跑，重试入口按设计要等全部跑完）。队列期间它必须一直是收起的。
        self.queue_panel.set_busy(busy or self._queue_mode)
        self._sync_retry_button()

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setValue(0 if not total else int(done / total * 100))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 的命名约定
        """关窗时必须先送走后台线程。

        窗口对象一旦被回收，还在跑的 QThread 会踩到野指针（Qt 会直接报
        "Destroyed while thread is still running" 并可能崩进程）。
        """
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            if not worker.wait(CLOSE_WAIT_MS):
                # 当前这批 HTTP 请求还挂着（中继无响应时可等到超时）。宁可多等一会儿，
                # 也不能让线程在运行中被销毁。
                self.statusBar().showMessage("等待当前请求结束…")
                worker.wait()
        super().closeEvent(event)

    def _on_export(self) -> None:
        if not self._cues:
            return
        stem = self._source_path.stem if self._source_path else "translated"
        suffix = self._source_path.suffix if self._source_path else ".srt"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出字幕",
            f"{stem}.translated{suffix}",
            "SRT (*.srt);;WebVTT (*.vtt)",
        )
        if not path:
            return
        try:
            written = subtitle_io.write_file(path, self._cues)
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.statusBar().showMessage(f"已导出: {written}")
