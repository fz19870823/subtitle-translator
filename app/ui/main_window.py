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
    AppConfig,
    ConfigError,
    ensure_runtime_dirs,
    load_config,
    save_config,
)
from app.core import subtitle_io
from app.core import checkpoint as checkpoint_store
from app.core.subtitle_io import Cue, SubtitleFormatError
from app.core.translator import (
    ENGINES,
    TranslationError,
    create_engine_for,
    locate_untranslated,
)
from app.ui.model_selector import MAX_FETCH_TIMEOUT, ModelSelector
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

        self.open_button = QPushButton("打开字幕…")
        self.open_button.clicked.connect(self._on_open)
        self.path_label = QLabel("未选择文件")

        file_row = QHBoxLayout()
        file_row.addWidget(self.open_button)
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
        self.editor.setPlaceholderText("打开字幕文件后，原文与译文会显示在这里…")

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

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开字幕文件", "", "字幕文件 (*.srt *.vtt);;所有文件 (*)"
        )
        if not path:
            return
        try:
            cues = subtitle_io.parse_file(path)
        except (SubtitleFormatError, OSError) as exc:
            QMessageBox.critical(self, "解析失败", str(exc))
            return

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

        # 上次翻到一半的记录要主动说出来：用户不知道有断点，就会以为只能重来。
        plan = self._resume_plan()
        self._sync_translate_button()
        if plan.count:
            self.statusBar().showMessage(
                f"已载入 {len(cues)} 条字幕　|　上次翻到 {plan.count} 条，"
                "点「继续翻译」接着往下走"
            )
        else:
            self.statusBar().showMessage(f"已载入 {len(cues)} 条字幕")

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
        """重算「有译文、但译文与原文一字不差，据此可断定没翻」的条目。

        定罪标准与引擎内部同一套（见 :func:`translator.locate_untranslated`）：
        只有**原文的书写族根本不是目标语言的族**而译文仍与原文相同时才算。
        日文「学校」译成中文仍是「学校」这类同形词一律放过 —— 那不是失败。
        没动过的条目（译文为空）自然也不算，所以取消/失败之后不会误报。
        """
        if not self._cues:
            self._stuck_indices = []
            return
        target_lang = self.target_combo.currentText().strip()
        self._stuck_indices = locate_untranslated(
            [cue.text for cue in self._cues],
            [cue.translation or "" for cue in self._cues],
            target_lang,
        )

    def _sync_retry_button(self) -> None:
        """按钮只在真有几条没翻出来时出现；忙的时候收起来。

        平时不该占着位置 —— 一个长期灰着的按钮只会让人猜它什么时候能用。
        """
        count = len(self._stuck_indices)
        show = count > 0 and self._worker is None
        self.retry_button.setVisible(show)
        self.retry_button.setEnabled(show)
        self.retry_button.setText(f"重试未翻译（{count}）")
        self.retry_button.setToolTip(
            "只把这几条重新发一遍，不动已经翻好的部分。\n"
            "自动重试用尽后仍失败时，建议先把上下文窗口调小（例如 1 条/次）"
            "或换个模型，再点这里。"
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

        targets = [
            self._cues[index]
            for index in self._stuck_indices
            if 0 <= index < len(self._cues)
        ]
        if not targets:
            self._recompute_stuck()
            self._sync_retry_button()
            return

        # 刻意**不清空**这几条的旧“译文”：它其实等于原文，留着正好当兜底 ——
        # 重试中途取消的话，字幕里还有原文可看，而不是变成一片空白。
        self._start_translation(
            self.engine_combo.currentText().strip(),
            self.source_combo.currentText().strip(),
            self.target_combo.currentText().strip(),
            skip_translated=False,
            note=f"重试 {len(targets)} 条未翻译的字幕",
            cues=targets,
            retry=True,
        )

    def _ask_retry_stuck(self, count: int, *, retried: bool = False) -> bool:
        """问要不要现在就重试这几条；返回 True 表示用户选了重试。

        单独拆成一个方法是为了能在测试里替换掉 —— 真实弹窗会阻塞事件循环，
        自动化测试里没法点它。
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("部分字幕没有翻译")
        box.setText(f"有 {count} 条字幕与原文完全相同，自动重试没能让模型改写它们。")
        box.setInformativeText(
            ("刚刚的手动重试也没能翻出来。"
             if retried
             else "")
            + "这几条多半是模型在偷懒（同一模型名下请求被分发到能力不齐的通道），"
            "隔一会儿重发往往就好了。\n\n"
            "若连续几次都不行，请先换个模型、或把上下文窗口调小"
            "（例如 1 条/次）再点「重试这些条目」。\n"
            "也可以先关掉本窗口，在译文区人工核对这几条。"
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
        cues: Sequence[Cue] | None = None,
        retry: bool = False,
    ) -> None:
        """构造引擎与后台线程并启动。断点相关的前置判断已经在外面做完了。

        ``note`` 是启动时顺带要告诉用户的一句话（比如「上次的记录用不上」）。
        它必须由这里一起写进状态栏 —— 在外面先写会被下面的「翻译中…」覆盖掉。

        ``cues`` / ``retry`` 供「重试未翻译」使用：只处理传进来的那几条。
        注意重试路径**不写断点** —— 断点记录用 ``enumerate`` 的下标给条目定位，
        而重试拿到的是原列表的一个子集，子集下标会被当成整份字幕的下标，
        把 A 条的译文记到 B 条名下。整份任务此时已经完成，重试丢了大不了再点一次，
        不值得为它冒这个险。
        """
        # 界面上临时选的模型要即时生效，否则会悄悄沿用配置文件里的旧值。
        # 想持久化就走「设置…」。
        requires_api = bool(getattr(ENGINES.get(engine_name), "requires_api", False))
        if requires_api:
            self._config.translation.model = self.model_selector.current_model()

        # 引擎构造留在主线程：它只读配置、不发网络请求，出错时能同步弹窗。
        try:
            engine = create_engine_for(engine_name, self._config)
        except (TranslationError, ConfigError) as exc:
            QMessageBox.critical(self, "翻译失败", str(exc))
            self.statusBar().showMessage("翻译失败")
            return
        except Exception as exc:  # 第三方后端可能抛出任意异常，不能让它掀掉界面
            QMessageBox.critical(self, "翻译失败", f"{type(exc).__name__}: {exc}")
            self.statusBar().showMessage("翻译失败")
            return

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

        targets = list(cues) if cues is not None else self._cues

        # 真正耗时的部分必须进后台线程：上千条字幕要发几十次请求、跑好几分钟，
        # 放在主线程里窗口会整个冻住（进度条不动、取消都点不了）。
        self._translation_meta = (engine_name, source_lang, target_lang)
        self._retry_mode = retry
        self._retry_count = len(targets) if retry else 0
        worker = TranslateWorker(
            engine,
            targets,
            source_lang=source_lang,
            target_lang=target_lang,
            # 界面上的上下文窗口永远说了算：它比配置里的值更“新”，
            # 而且用户就是冲着「把这一批切小点」才去动它的。
            batch_size=int(self.context_spin.value()),
            writer=writer,
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
        # 重试的这几条译文都等于原文（所以才被判为没翻），照「已完成」算会把进度条
        # 一上来就顶到 100%；重试是按条数重新走的，起点就是 0。
        already = 0 if retry else sum(1 for cue in targets if cue.is_translated)
        # 续传时进度条立刻落在断点处，不干等第一批请求回来 —— 否则用户会以为
        # 之前的进度丢了，或者以为又在从头翻。
        self.progress.setValue(
            0 if not already or not total else int(already / total * 100)
        )
        if retry:
            message = f"重试未翻译的条目…（共 {total} 条）"
        elif skip_translated and already:
            message = f"接着上次翻译…（还剩 {total - already} 条，共 {total} 条）"
        else:
            message = f"翻译中…（共 {total} 条）"
        self.statusBar().showMessage(f"{note}　|　{message}" if note else message)
        worker.start()

    def _on_translate(self) -> None:
        """启动翻译。**立即返回** —— 耗时的部分在后台线程里跑。"""
        if not self._cues or self._worker is not None:
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

        self._start_translation(
            engine_name,
            source_lang,
            target_lang,
            skip_translated=skip_translated,
            note=note,
        )

    # ---------- 后台线程的回调（信号跨线程排队投递，槽仍在主线程执行） ----------

    def _on_translate_succeeded(self) -> None:
        worker = self._worker
        engine = worker.engine if worker is not None else None
        checkpoint_path = worker.checkpoint_path if worker is not None else None
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
        if remaining:
            summary += f"　|　仍有 {remaining} 条未翻译（可点「重试未翻译」）"
        if notes:
            summary += "　|　" + "；".join(notes)
        self.statusBar().showMessage(summary)

        # 回抄是**静默**故障：不报告就没人会发现手里那份字幕根本没翻。
        # 救回来了在状态栏记账；没救回来的必须弹出来，并当场给一条手动重试的出路。
        if remaining:
            self._warn_untranslated(remaining, retried=retry_mode)

    def _on_translate_cancelled(self, done: int, total: int) -> None:
        worker = self._worker
        error = worker.checkpoint_error if worker is not None else ""
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
        self.statusBar().showMessage(message)
        self._sync_translate_button()

    def _on_translate_failed(self, kind: str, message: str) -> None:
        worker = self._worker
        resumable = (
            worker is not None
            and worker.checkpoint_path is not None
            and worker.done > 0
        )
        self._finish_translation()
        text = message if kind == "TranslationError" else f"{kind}: {message}"
        if resumable:
            text += (
                "\n\n已翻好的部分已经存下来了，处理完问题点「继续翻译」就能接着来。"
            )
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
        self.open_button.setEnabled(not busy)
        self.settings_button.setEnabled(not busy)
        self.engine_combo.setEnabled(not busy)
        self.source_combo.setEnabled(not busy)
        self.target_combo.setEnabled(not busy)
        self.model_selector.setEnabled(not busy)
        self.context_spin.setEnabled(not busy)
        self.translate_button.setEnabled(not busy and bool(self._cues))
        self.export_button.setEnabled(not busy and bool(self._cues))
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
