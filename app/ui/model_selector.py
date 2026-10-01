"""模型选择控件：可编辑下拉框 + 从服务端拉取模型列表。

拉取要发网络请求，不能放在 UI 线程 —— 否则点一下按钮整个窗口就假死。
请求在后台 QThread 里跑，结果通过信号回到主线程。
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from app.core.engines.openai_compat import fetch_models

#: 拉取模型列表的超时上限。列表接口不该让人干等，比翻译超时短得多。
MAX_FETCH_TIMEOUT = 60

_ERROR_STYLE = "color: #d9534f;"
_HINT_STYLE = "color: palette(mid);"

#: ``set_source_provider`` 注入的回调：返回 (base_url, api_key, timeout)
SourceProvider = Callable[[], Tuple[str, str, int]]

#: ``set_fetcher`` 注入的取数函数。**按引擎而异** —— OpenAI 兼容层在
#: ``/models``，Ollama 在 ``/api/tags``，不能写死。
ModelFetcher = Callable[..., List[str]]


class ModelFetchWorker(QThread):
    """后台拉取模型列表。"""

    fetched = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        fetcher: ModelFetcher,
        base_url: str,
        api_key: str,
        timeout: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._fetcher = fetcher
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout

    def run(self) -> None:  # 在工作线程里执行，不能碰任何界面对象
        try:
            models = self._fetcher(
                self._base_url, self._api_key, timeout=self._timeout
            )
        except Exception as exc:  # 网络层什么异常都可能抛，绝不能让它掀掉线程
            self.failed.emit(str(exc))
            return
        self.fetched.emit(models)


class ModelSelector(QWidget):
    """模型 id 输入控件：下拉可选、可手输、可一键拉取服务端列表。

    控件不关心地址和密钥从哪来 —— 通过 :meth:`set_source_provider` 注入一个
    返回 ``(base_url, api_key, timeout)`` 的回调。于是同一个控件既能读已保存的
    配置，也能读设置对话框里刚填进去、还没保存的值。
    """

    #: 拉取成功，携带模型 id 列表
    models_loaded = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.combo = QComboBox()
        self.combo.setEditable(True)  # 允许手输列表里没有的模型 id
        self.combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combo.setMinimumWidth(240)
        self.combo.lineEdit().setPlaceholderText("模型 id：拉取后选择，或直接手输")

        self.fetch_button = QPushButton("拉取模型")
        self.fetch_button.setToolTip("向服务端 /models 请求可用模型列表")

        self.status = QLabel("")
        self.status.setStyleSheet(_HINT_STYLE)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.fetch_button)
        layout.addWidget(self.status, 1)

        self._source_provider: SourceProvider | None = None
        #: None 表示「用模块级的默认实现」，到 ``fetch()`` 那一刻才解析 ——
        #: 这样测试里替换 ``fetch_models`` 对已经建好的控件同样生效。
        self._fetcher: ModelFetcher | None = None
        self._worker: ModelFetchWorker | None = None
        self._active = True
        self._requires_key = True

        self.fetch_button.clicked.connect(self.fetch)

    # ------------------------------------------------------------ 外部接口

    def set_source_provider(self, provider: SourceProvider) -> None:
        """注册数据源，回调需返回 ``(base_url, api_key, timeout)``。"""
        self._source_provider = provider

    def set_fetcher(self, fetcher: ModelFetcher | None) -> None:
        """注册取模型列表的函数：``fetcher(base_url, api_key, *, timeout)``。

        按引擎注入 —— 本地服务（Ollama）的列表在 ``/api/tags``，拿 OpenAI 那套
        打到 ``/models`` 上会 404，而错误信息只会说「拉取模型列表失败: HTTP 404」，
        完全指不到「地址填错了还是引擎选错了」。传 None 则回到默认实现。
        """
        self._fetcher = fetcher

    def set_requires_key(self, required: bool) -> None:
        """这个引擎没有密钥这回事（本地推理服务）时，别再拿密钥拦人。"""
        self._requires_key = bool(required)

    def set_active(self, active: bool) -> None:
        """引擎不需要 API 参数（如 echo）时整体禁用。"""
        self._active = bool(active)
        self.combo.setEnabled(self._active)
        if not self.is_fetching():
            self.fetch_button.setEnabled(self._active)

    def current_model(self) -> str:
        return self.combo.currentText().strip()

    def set_current_model(self, model: str) -> None:
        """设置当前模型 id。

        不用 ``setCurrentText``：它在「值不在列表里」时的行为依 Qt 版本而异，
        对不可编辑的下拉框还是静默 no-op（踩过一次）。这里显式区分两种情况。
        """
        model = model or ""
        index = self.combo.findText(model)
        if model and index >= 0:
            self.combo.setCurrentIndex(index)
        else:
            self.combo.setEditText(model)  # 手输的值必须能写进编辑框

    def set_models(self, models: Sequence[str]) -> None:
        """填充候选列表，并保留用户当前已选/已手输的值。"""
        current = self.current_model()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(list(models))
        self.combo.blockSignals(False)
        if current:
            self.set_current_model(current)

    def is_fetching(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # ------------------------------------------------------------ 拉取

    def fetch(self) -> None:
        if self.is_fetching():
            return  # 已经在拉了，忽略重复点击
        if self._source_provider is None:
            self._warn("未配置数据源")
            return
        try:
            base_url, api_key, timeout = self._source_provider()
        except Exception as exc:
            self._warn(f"读取配置失败：{exc}")
            return

        if not base_url:
            self._warn("请先填写 API 地址")
            return
        if not api_key and self._requires_key:
            self._warn("请先填写密钥")
            return

        self.fetch_button.setEnabled(False)
        self.status.setStyleSheet(_HINT_STYLE)
        self.status.setText("拉取中…")

        # 在这里解析默认实现（而不是构造时缓存），测试替换 fetch_models 才有效。
        fetcher = self._fetcher or fetch_models
        worker = ModelFetchWorker(fetcher, base_url, api_key, int(timeout), self)
        worker.fetched.connect(self._on_fetched)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker  # 保留引用：worker 一旦被回收，信号连接也会消失
        worker.start()

    def _on_fetched(self, models: List[str]) -> None:
        self.set_models(models)
        if models:
            self.status.setText(f"共 {len(models)} 个模型")
        else:
            self._warn("服务端没有返回任何模型")
        self.models_loaded.emit(list(models))

    def _on_failed(self, message: str) -> None:
        self._warn(message)

    def _on_finished(self) -> None:
        self.fetch_button.setEnabled(self._active)

    def _warn(self, message: str) -> None:
        self.status.setStyleSheet(_ERROR_STYLE)
        self.status.setText(message)
        self.fetch_button.setEnabled(self._active and not self.is_fetching())
