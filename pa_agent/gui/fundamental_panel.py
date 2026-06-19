"""侧边栏面板「基本面」：展示当前标的的基本面/资金面/宏观/情绪分栏。

数据来自 ``pa_agent.context.fundamental_context.build_sections_for_symbol``，
分析完成后在后台线程抓取(命中缓存不会重复打网)，回主线程刷新，不阻塞 UI。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from pa_agent.config.settings import Settings

logger = logging.getLogger(__name__)

_PLACEHOLDER = "该标的暂无可用基本面，或未开启多维上下文。"


class _FetchWorker(QObject):
    """后台抓取 worker：在子线程调用统一入口，完成后发信号。"""

    done = pyqtSignal(list)  # list[tuple[str, str]]

    def __init__(self, symbol: str, settings: Any) -> None:
        super().__init__()
        self._symbol = symbol
        self._settings = settings

    def run(self) -> None:
        sections: list[tuple[str, str]] = []
        try:
            from pa_agent.context import fundamental_context

            sections = fundamental_context.build_sections_for_symbol(
                self._symbol, settings=self._settings
            )
        except Exception:
            logger.warning("fundamental panel fetch failed", exc_info=True)
            sections = []
        self.done.emit(sections)


class FundamentalPanel(QWidget):
    """只读分栏展示基本面/资金面/宏观/情绪。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings: Any = None
        self._thread: QThread | None = None
        self._worker: _FetchWorker | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(8)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(container)
        outer.addWidget(scroll)

        self._placeholder = QLabel(_PLACEHOLDER)
        self._placeholder.setObjectName("mutedLabel")
        self._placeholder.setWordWrap(True)
        self._layout.addWidget(self._placeholder)

    def bind_settings(self, settings: Settings | None) -> None:
        self._settings = settings.prompt if settings is not None else None

    def _clear_sections(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _render(self, sections: list[tuple[str, str]]) -> None:
        self._clear_sections()
        if not sections:
            ph = QLabel(_PLACEHOLDER)
            ph.setObjectName("mutedLabel")
            ph.setWordWrap(True)
            self._layout.addWidget(ph)
            return
        for title, body in sections:
            t = QLabel(title)
            t.setStyleSheet("font-weight: bold; color: #58a6ff;")
            self._layout.addWidget(t)
            b = QLabel(body)
            b.setWordWrap(True)
            b.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._layout.addWidget(b)

    def update_for_symbol(self, symbol: str) -> None:
        """分析完成后调用：后台抓取并刷新分栏。"""
        if not symbol:
            return
        # 复用入口的缓存；抓取放后台线程避免阻塞 UI
        self._thread = QThread()
        self._worker = _FetchWorker(symbol, self._settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_done(self, sections: list) -> None:
        self._render(list(sections))

    def clear(self) -> None:
        self._render([])
