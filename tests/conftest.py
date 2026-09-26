"""pytest 全局配置。

这里只做一件事：**把 QApplication 的引用一直留到进程结束**。

## 为什么要这么写

本机 Python 3.14.7 + PySide6 6.11.2 在 ``offscreen`` 平台下有个退出期问题：
pytest 收尾时会调 ``gc_collect_harder()``（见 ``_pytest/unraisableexception.py``），
如果此时它去回收 ``QApplication``，PySide6 的 C++ 析构会踩到已经拆掉的东西，
进程以 ``STATUS_HEAP_CORRUPTION (0xC0000374)`` 结束 —— 测试全部通过，
只有退出码不对，会让「全绿」这个信号失真。

实测结论（都是对照出来的，不是猜的）：

- 改动前 107 个测试：连跑 3 次，退出码全是 0；
- 测试涨到 131 个：连跑 2 次，全是 ``0xC0000374``；
- 去掉 14 个参数化用例（117 个）：退出码恢复正常 —— 说明**与具体测试内容无关**，
  是对象规模改变了 GC 时机；
- 单独跑非 UI（99 个）或单独跑 UI（32 个）都正常，混跑才触发；
- ``PYTHONMALLOC=debug`` + faulthandler 抓到的栈：崩在 ``gc_collect_harder``，
  即 pytest 退出期，不是任何一条用例。

规避方式：用模块级变量把 ``QApplication`` 引用住，退出期 GC 就不会去析构它，
留给操作系统在进程结束时回收。Qt 官方本来也不建议手动销毁 ``QApplication``。

没有装 PySide6 的解释器（如 3.13）上，这个 fixture 直接空转，无副作用。
"""
from __future__ import annotations

import os

import pytest

# 必须在 import QApplication 之前设置；单个测试模块里的 setdefault 会太晚。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

#: 故意一直持有，不清空 —— 详见模块 docstring
_QT_KEEPALIVE: list = []


@pytest.fixture(scope="session", autouse=True)
def _qt_app_keepalive():
    """尽早建出 QApplication 并持有引用，避免它在退出期被 GC 回收。"""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        yield  # 这个解释器没有 PySide6，什么都没得管
        return

    app = QApplication.instance() or QApplication([])
    _QT_KEEPALIVE.append(app)
    yield
    # 这里刻意什么都不做：放开引用反而会重新引回那个退出期崩溃。
