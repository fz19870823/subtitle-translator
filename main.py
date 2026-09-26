"""应用入口。

PySide6 在 ``main()`` 内部导入，而不是放在模块顶层 —— 这样 pytest 采集
``app.core`` 时不需要一个可用的 Qt 运行时。
"""
from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.config import APP_NAME, APP_VERSION
    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("fz19870823")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
