"""侧边栏面板「基本面」：展示当前标的的基本面/资金面/宏观/情绪分栏。

数据来自 ``pa_agent.context.fundamental_context.build_sections_for_symbol``，
分析完成后在后台线程抓取(命中缓存不会重复打网)，回主线程刷新，不阻塞 UI。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from pa_agent.config.settings import Settings

logger = logging.getLogger(__name__)

_PLACEHOLDER = "该标的暂无可用基本面，或未开启多维上下文。"
_LOADING = "正在获取基本面/宏观/情绪数据…（港股/美股需联网，首次约数秒）"


class _FetchWorker(QObject):
    """后台抓取 worker：在子线程调用统一入口，完成后发信号。"""

    done = pyqtSignal(list)  # list[tuple[str, str]]

    def __init__(self, symbol: str, settings: Any, exchange: str = "") -> None:
        super().__init__()
        self._symbol = symbol
        self._settings = settings
        self._exchange = exchange

    def run(self) -> None:
        sections: list[tuple[str, str]] = []
        try:
            from pa_agent.context import fundamental_context

            sections = fundamental_context.build_sections_for_symbol(
                self._symbol,
                exchange=self._exchange or None,
                settings=self._settings,
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
        # 当前在抓的 (symbol, exchange)；以及在途期间用户切到的最新请求。
        self._current: tuple[str, str] | None = None
        self._pending: tuple[str, str] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # 不确定进度条(转圈)：抓取期间显示，抓完隐藏。
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # range=(0,0) → 忙碌/不确定动画
        self._progress.setTextVisible(False)
        self._progress.setMaximumHeight(4)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

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

    def _show_message(self, text: str) -> None:
        self._clear_sections()
        ph = QLabel(text)
        ph.setObjectName("mutedLabel")
        ph.setWordWrap(True)
        self._layout.addWidget(ph)

    def _render(self, sections: list[tuple[str, str]]) -> None:
        self._clear_sections()
        if not sections:
            self._show_message(_PLACEHOLDER)
            return
        for title, body in sections:
            t = QLabel(title)
            t.setStyleSheet("font-weight: bold; color: #58a6ff;")
            self._layout.addWidget(t)
            b = QLabel(body)
            b.setWordWrap(True)
            b.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._layout.addWidget(b)

    def update_for_symbol(self, symbol: str, exchange: str = "") -> None:
        """获取数据/分析完成后调用：后台抓取并刷新分栏。

        *exchange* 为用户选中的交易所(如 NASDAQ/HKEX)，用于交易所优先判定市场。
        """
        if not symbol:
            return
        req = (symbol, exchange)
        # 已有在途抓取：不丢弃，记下最新请求，待当前抓完自动补抓最新标的。
        # (否则从慢标的<如黄金18s>快速切到新标的时，新标的的抓取会被永久跳过。)
        if self._thread is not None and self._thread.isRunning():
            self._pending = req
            return
        self._start_fetch(req)

    def _start_fetch(self, req: tuple[str, str]) -> None:
        symbol, exchange = req
        self._current = req
        self._pending = None
        self._progress.setVisible(True)
        self._show_message(_LOADING)
        # 复用入口的缓存；抓取放后台线程避免阻塞 UI
        self._thread = QThread()
        self._worker = _FetchWorker(symbol, self._settings, exchange)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.done.connect(self._thread.quit)
        # 不在 _on_done 里换线程(那时线程尚未真正停止，重建会触发
        # “QThread: Destroyed while running” 崩溃)；等 finished(事件循环已退出、
        # 线程确实停了)后再补抓 pending、并安全 deleteLater。
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_done(self, sections: list) -> None:
        # 抓取期间用户已切到别的标的：丢弃本次(旧标的)结果，不渲染，
        # 待 _on_thread_finished 再补抓最新标的；避免把旧标的(如黄金
        # 「只有宏观」)的结果糊到新标的上。
        if self._pending is not None and self._pending != self._current:
            return
        self._pending = None
        self._progress.setVisible(False)
        self._render(list(sections))

    def _on_thread_finished(self) -> None:
        """线程真正结束后回收，并在用户已切标的时补抓最新的。"""
        th = self.sender()
        if isinstance(th, QThread):
            th.deleteLater()
        if self._thread is th:
            self._thread = None
        # 期间切了别的标的 → 现在线程已停，安全地起新一轮抓取。
        if self._pending is not None and self._pending != self._current:
            nxt = self._pending
            self._pending = None
            self._start_fetch(nxt)

    def clear(self) -> None:
        self._render([])
