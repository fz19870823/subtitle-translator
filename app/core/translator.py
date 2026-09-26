"""翻译后端。

界面层只依赖 :class:`Translator` 抽象；具体引擎（LLM API、本地模型、机翻服务）
通过 :func:`register` 注册进 ``ENGINES``，新增后端不必改动界面代码。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence, Type

from app.core.subtitle_io import Cue

if TYPE_CHECKING:  # 只为类型标注，运行时不引入，避免循环依赖
    from app.config import AppConfig


class TranslationError(RuntimeError):
    """后端无法完成本次请求。"""


@dataclass
class TranslationRequest:
    """一次翻译请求的最小描述。

    先留出 source/target 字段，后续接入带上下文或术语表的引擎时直接扩展。
    """

    text: str
    source_lang: str
    target_lang: str


ProgressCallback = Callable[[int, int], None]


class Translator(abc.ABC):
    """所有翻译引擎的基类。"""

    name = "base"

    #: 单次请求最多携带多少条字幕
    batch_size = 10

    @abc.abstractmethod
    def translate_batch(self, requests: Sequence[TranslationRequest]) -> List[str]:
        """翻译一批文本，返回等长列表。"""

    def translate_cues(
        self,
        cues: Sequence[Cue],
        *,
        source_lang: str,
        target_lang: str,
        batch_size: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> Sequence[Cue]:
        """就地写入 ``cue.translation``，返回同一序列。"""
        # None = 没传（用引擎默认值）；0 或负数 = 调用方写错了，必须报错而不是静默兜底。
        size = self.batch_size if batch_size is None else batch_size
        if size <= 0:
            raise ValueError(f"batch_size 必须为正整数，收到 {size!r}")

        total = len(cues)
        for offset in range(0, total, size):
            window = list(cues[offset: offset + size])
            requests = [
                TranslationRequest(
                    text=cue.text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                for cue in window
            ]
            results = self.translate_batch(requests)
            if len(results) != len(window):
                raise TranslationError(
                    f"引擎 {self.name!r} 返回 {len(results)} 条结果，期望 {len(window)} 条"
                )
            for cue, text in zip(window, results):
                cue.translation = text
            if progress is not None:
                progress(min(offset + size, total), total)
        return cues


class EchoTranslator(Translator):
    """离线占位实现：给每行加上目标语言前缀。

    存在的意义是让「读取 -> 翻译 -> 导出」这条链路在没有 API key 时也能跑通
    并被测试覆盖。它不是可用的翻译引擎。
    """

    name = "echo"

    def translate_batch(self, requests: Sequence[TranslationRequest]) -> List[str]:
        return [f"[{request.target_lang}] {request.text}" for request in requests]


ENGINES: Dict[str, Type[Translator]] = {}


def register(engine: Type[Translator]) -> Type[Translator]:
    """把引擎类注册进 ``ENGINES``，可作装饰器使用。"""
    key = getattr(engine, "name", "")
    if not key or key == "base":
        raise ValueError("翻译引擎必须定义唯一的 name 属性")
    ENGINES[key] = engine
    return engine


def create_engine(name: str, **kwargs) -> Translator:
    try:
        factory = ENGINES[name]
    except KeyError:
        raise TranslationError(
            f"未知的翻译引擎 {name!r}，可用: {sorted(ENGINES)}"
        ) from None
    return factory(**kwargs)


def create_engine_for(name: str, app_config: "AppConfig | None" = None) -> Translator:
    """按配置构造引擎。

    需要外部参数的引擎（如 OpenAI 兼容后端）实现 ``from_config(cfg)`` 类方法；
    不需要的（如 echo）直接无参实例化。这样界面层不必知道每个引擎要什么参数。
    """
    try:
        factory = ENGINES[name]
    except KeyError:
        raise TranslationError(
            f"未知的翻译引擎 {name!r}，可用: {sorted(ENGINES)}"
        ) from None

    from_config = getattr(factory, "from_config", None)
    if callable(from_config) and app_config is not None:
        return from_config(app_config.translation)
    return factory()


register(EchoTranslator)
