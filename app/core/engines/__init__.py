"""具体翻译引擎实现。

导入本包即完成引擎注册（各模块底部调用 :func:`~app.core.translator.register`）。
"""
from app.core.engines.ollama import OllamaTranslator  # noqa: F401
from app.core.engines.openai_compat import OpenAICompatTranslator  # noqa: F401

__all__ = ["OllamaTranslator", "OpenAICompatTranslator"]
