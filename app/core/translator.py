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


class TranslationCancelled(RuntimeError):
    """用户主动中止了本次翻译。

    **刻意不继承** :class:`TranslationError`：取消是用户的正常选择，不是故障。
    若继承，界面上「翻译失败」的弹窗会把主动取消也报成错误。
    """


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


#: 书写系统特征 → 归一化「族」名。**顺序即优先级**：
#: 日语同时含汉字与假名，必须先认假名，否则会被判成汉字的族。
_SCRIPT_FAMILIES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("ja", re.compile(r"[\u3040-\u309f\u30a0-\u30ff\uff66-\uff9d]")),   # 平/片假名
    ("ko", re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")),   # 谚文
    ("cyrillic", re.compile(r"[\u0400-\u04ff\u0500-\u052f]")),
    ("arabic", re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff]")),
    ("hebrew", re.compile(r"[\u0590-\u05ff]")),
    ("devanagari", re.compile(r"[\u0900-\u097f]")),
    ("thai", re.compile(r"[\u0e00-\u0e7f]")),
    ("greek", re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")),
    ("han", re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")),  # 汉字
    ("latin", re.compile(r"[A-Za-z\u00c0-\u024f]")),
)

#: 语言主码 → 族。表里没有的语言以它自己的主码为族，于是与任何别的语言都不同 ——
#: 拿不准时宁可判成「两门不同语言」，因为漏检的代价（交付未翻译的字幕）远大于误检。
_PRIMARY_FAMILY: Dict[str, str] = {
    "zh": "han", "cmn": "han", "yue": "han", "wuu": "han", "nan": "han", "hak": "han",
    "ja": "ja", "jp": "ja",
    "ko": "ko", "kr": "ko",
    "ru": "cyrillic", "uk": "cyrillic", "be": "cyrillic", "bg": "cyrillic",
    "sr": "cyrillic", "mk": "cyrillic", "kk": "cyrillic",
    "ar": "arabic", "fa": "arabic", "ur": "arabic",
    "he": "hebrew",
    "hi": "devanagari", "bn": "devanagari", "mr": "devanagari", "ne": "devanagari",
    "th": "thai",
    "el": "greek",
    "en": "latin", "de": "latin", "fr": "latin", "es": "latin", "pt": "latin",
    "it": "latin", "nl": "latin", "id": "latin", "ms": "latin", "vi": "latin",
    "tr": "latin", "pl": "latin", "cs": "latin", "sk": "latin", "sv": "latin",
    "da": "latin", "nb": "latin", "no": "latin", "fi": "latin", "ro": "latin",
    "hu": "latin", "hr": "latin", "sl": "latin", "lt": "latin", "lv": "latin",
    "et": "latin", "tl": "latin", "sw": "latin", "af": "latin", "ca": "latin",
    "gl": "latin", "eu": "latin", "is": "latin", "sq": "latin", "az": "latin",
}

_FAMILY_ORDER = [name for name, _ in _SCRIPT_FAMILIES]


def _family_of(code: str) -> str | None:
    """语言码 → 书写族；``auto`` / 空 / 认不出返回 None。"""
    primary = _norm_lang(code).split("-")[0]
    if not primary or primary == "auto":
        return None
    return _PRIMARY_FAMILY.get(primary, primary)


def detect_script(text: str) -> str | None:
    """从一段文本猜它的书写族（不看语言码）。

    日语只要出现一个假名就归 ``ja`` —— 日语的助词、词尾几乎全是假名，
    这个信号比汉字可靠得多（纯汉字的句子中日都有，判不出来）。
    """
    body = _TAG_RE.sub(" ", text)
    for family, pattern in _SCRIPT_FAMILIES:
        if pattern.search(body):
            return family
    return None


def guess_text_family(samples: "Sequence[str]") -> str | None:
    """整批文本的族：按条目投票取众数。

    单条不可靠（一条纯汉字的日文句子会被认成汉字族），整批看多数就稳了。
    """
    counts: Dict[str, int] = {}
    for text in samples:
        if not is_translatable(text):
            continue
        family = detect_script(text)
        if family:
            counts[family] = counts.get(family, 0) + 1
    if not counts:
        return None
    # 票数相同时按 _SCRIPT_FAMILIES 的先后定序，保证结果稳定可测。
    return max(counts, key=lambda fam: (counts[fam], -_FAMILY_ORDER.index(fam)))


def should_check_echo(
    source_lang: str,
    target_lang: str,
    *,
    samples: "Sequence[str]" = (),
) -> bool:
    """是否对该批次做「原样回抄」检查。

    关键点：**源语言是 ``auto`` 不等于放弃检查**。界面默认就是 ``auto``，
    若在这里直接返回 False，等于给最常见的使用方式关掉了唯一的防护
    （实测：auto + 日文字幕，8 批里 2 批整批 20 条原样退回，全部静默交付）。
    传了 ``samples`` 时就按文本的书写族去认源语言。

    只有一种情况确定要放过：两边的族相同（``en-GB`` 对 ``en-US``、
    ``zh-CN`` 对 ``zh-TW``）—— 这类转换里大量条目本来就该原样，
    回抄比例天然偏高，判定必然误报。
    """
    dst_family = _family_of(target_lang)
    if dst_family is None:
        return False  # 目标语言都定不下来，无从判断

    src_family = _family_of(source_lang)
    if src_family is None:
        src_family = guess_text_family(samples)
        if src_family is None:
            return False

    return src_family != dst_family


def locate_untranslated(
    sources: "Sequence[str]",
    outputs: "Sequence[str]",
    target_lang: str,
) -> List[int]:
    """找出「一字未改、且据此可断定没翻」的条目下标。

    单独一条与原文相同**不足以定罪** —— 日文「学校」译成中文仍是「学校」，
    这类同形汉字词本来就该原样。真正能定罪的是：这条文本的书写族
    压根不是目标语言的族（假名 vs 中文汉字、拉丁 vs 汉字），
    那它一字未改就只可能是没翻。
    """
    dst_family = _family_of(target_lang)
    if dst_family is None:
        return []

    hits: List[int] = []
    for index, (source, output) in enumerate(zip(sources, outputs)):
        if not is_translatable(source) or source.strip() != output.strip():
            continue
        family = detect_script(source)
        if family is None or family == dst_family:
            continue
        hits.append(index)
    return hits


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
    if not should_check_echo(source_lang, target_lang, samples=sources):
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
#: 返回 True 表示应当中止。在**批与批之间**被查询，批内无法中断。
StopCallback = Callable[[], bool]


class Translator(abc.ABC):
    """所有翻译引擎的基类。"""

    name = "base"

    #: 单次请求最多携带多少条字幕
    batch_size = 10

    #: 是否需要外部 API 参数（地址/密钥/模型）。界面据此启用模型选择控件。
    requires_api = False

    #: 质量相关的可观测统计。基类给出 0，引擎按需覆盖；
    #: 界面可以无条件读取，不必知道具体是哪个引擎。
    #: 因模型丢换行而被单独重译的条目数
    line_repair_count = 0
    #: 因整批原样回抄而重发的批次数
    echo_retry_count = 0
    #: 因残留未翻译条目而触发的逐条重译条数
    echo_item_count = 0
    #: 逐条重译救回来的条数
    echo_repaired_count = 0
    #: 用尽手段后仍未翻译的条数（> 0 表示译文不完整，必须让用户知道）
    untranslated_count = 0

    def quality_notes(self) -> List[str]:
        """用一句话概括本次翻译的质量插曲，供界面提示。

        回抄是**静默**故障：不报告就没人会发现手里那份字幕根本没翻。
        所以宁可啰嗦，也要把「救了几条、还剩几条没救回来」摆到台面上。
        """
        notes: List[str] = []
        if self.echo_retry_count:
            notes.append(f"整批原样退回，重发 {self.echo_retry_count} 次")
        if self.echo_repaired_count:
            notes.append(f"逐条重译救回 {self.echo_repaired_count} 条")
        if self.line_repair_count:
            notes.append(f"换行丢失后按行重译 {self.line_repair_count} 条")
        if self.untranslated_count:
            notes.append(f"仍有 {self.untranslated_count} 条疑似未翻译")
        return notes

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
        should_stop: StopCallback | None = None,
    ) -> Sequence[Cue]:
        """就地写入 ``cue.translation``，返回同一序列。

        ``should_stop`` 是「还要不要继续」的回调，**每批之间**查一次。
        批内没法中断 —— 一次 HTTP 请求已经在路上，只能等它返回；实测单批
        3–5 秒，所以取消的延迟上限就是一批的时间。有了它，用户点取消后
        不必干等几十批跑完（上千条字幕要几分钟）。
        """
        # None = 没传（用引擎默认值）；0 或负数 = 调用方写错了，必须报错而不是静默兜底。
        size = self.batch_size if batch_size is None else batch_size
        if size <= 0:
            raise ValueError(f"batch_size 必须为正整数，收到 {size!r}")

        total = len(cues)
        for offset in range(0, total, size):
            if should_stop is not None and should_stop():
                raise TranslationCancelled(f"已取消，完成 {offset}/{total} 条")
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
