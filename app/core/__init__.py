"""解析、翻译等与界面无关的核心逻辑。

导入本包会连带注册具体翻译引擎，因此 ``ENGINES`` 在任何入口下都是完整可用的。
"""
from app.core import engines  # noqa: F401  # 导入即注册

__all__ = ["engines"]
