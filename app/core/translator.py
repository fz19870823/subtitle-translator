"""翻译后端。

界面层只依赖 :class:`Translator` 抽象；具体引擎（LLM API、本地模型、机翻服务）
通过 :func:`register` 注册进 ``ENGINES``，新增后端不必改动界面代码。
"""
from __future__ import annotations

import abc
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence, Type

from app.core.subtitle_io import Cue

if TYPE_CHECKING:  # 只为类型标注，运行时不引入，避免循环依赖
    from app.config import AppConfig


class TranslationError(RuntimeError):
    """后端无法完成本次请求。"""


#: 任意语言的字母（拉丁、西里尔、希腊、假名、汉字、谚文…）。
#: 换行记号 ⏎、数字、各类标点都不属于字母 —— 这正是我们想要的：
#: 「123」「---」「♪♪」这种行不该参与回抄判定。
_LETTER_RE = re.compile(
    r"[^\W\d_]", re.UNICODE
)
_TAG_RE = re.compile(r"<[^>]+>")


def is_translatable(text: str) -> bool:
    """这条文本是否含需要翻译的实义字符。

    纯粹的符号行、数字行、空行在跨语言翻译里本就该原样保留，
    拿它们去判断「模型是不是没翻译」会得出错误的结论。
    """
    if not text or not text.strip():
        return False
    return bool(_LETTER_RE.search(_TAG_RE.sub(" ", text)))


def _norm_lang(code: str) -> str:
    return (code or "").strip().lower().replace("_", "-")


def should_check_echo(source_lang: str, target_lang: str) -> bool:
    """是否对该批次做「原样回抄」检查。

    只在能确定**是两门不同语言**时才检查，判不准就宁可漏检：

    - 语言码为空或是 ``auto``：无从判断，误报会打断正常任务；
    - 只是地区/字形变体（``en-GB`` 对 ``en-US``、``zh-CN`` 对 ``zh-TW``）：
      这类转换里大量条目本来就该原样，回抄比例天然偏高，判定必然误报。
    """
    src, dst = _norm_lang(source_lang), _norm_lang(target_lang)
    if not src or not dst or "auto" in (src, dst):
        return False
    return src.split("-")[0] != dst.split("-")[0]


def looks_like_verbatim_echo(
    sources: Sequence[str],
    outputs: Sequence[str],
    source_lang: str,
    target_lang: str,
    *,
    min_items: int = 3,
    threshold: float = 0.5,
) -> bool:
    """译文是否整批与原文一模一样。

    真实故障形态：中继把请求路由到能力不足的上游时，会把整个数组**原样返回**，
    不是零散几条 —— 所以按「批次」判定，命中率高、误报少。
    单条相同是正常的（专有名词、缩写），因此只在可比条目够多时判定，
    且要求超过半数。
    """
    if not should_check_echo(source_lang, target_lang):
        return False

    pairs = [(s, o) for s, o in zip(sources, outputs) if is_translatable(s)]
    if len(pairs) < min_items:
        return False
    echoed = sum(1 for s, o in pairs if s.strip() == o.strip())
    return echoed / len(pairs) > threshold


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

    #: 是否需要外部 API 参数（地址/密钥/模型）。界面据此启用模型选择控件。
    requires_api = False

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
