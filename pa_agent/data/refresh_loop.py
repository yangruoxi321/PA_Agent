"""1 Hz data refresh loop running on a dedicated QThread."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pa_agent.data.base import DataSource, DataSourceTransientError, KlineBar
from pa_agent.data.kline_buffer import KlineBuffer

if TYPE_CHECKING:
    from pa_agent.util.threading import CancelToken

logger = logging.getLogger(__name__)

from PyQt6.QtCore import QThread, pyqtSignal, QObject


class RefreshLoop(QThread):
    """Fetches the latest K-line snapshot every *interval_ms* milliseconds.

    Signals
    -------
    frame_ready(list[KlineBar])
        Emitted after each successful fetch with the raw bar list.
    status_changed(str)
        Emitted with a human-readable status string (e.g. "数据延迟").
    """

    frame_ready = pyqtSignal(list)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        data_source: DataSource,
        buffer: KlineBuffer,
        n_bars: int,
        interval_ms: int = 1000,
        cancel_token: "CancelToken | None" = None,
        parent: "QObject | None" = None,
    ) -> None:
        super().__init__(parent)
        self._source = data_source
        self._buffer = buffer
        self._n_bars = n_bars
        self._interval_ms = interval_ms
        self._cancel_token = cancel_token
        self._consecutive_failures = 0
        self._failure_threshold_s = 5.0

    def run(self) -> None:  # noqa: C901
        """Main loop — runs on the worker thread."""
        failure_start: float | None = None

        while True:
            # Check cancellation
            if self._cancel_token is not None and self._cancel_token.is_set():
                logger.debug("RefreshLoop cancelled")
                break

            t0 = time.monotonic()
            try:
                bars = self._source.latest_snapshot(self._n_bars + 5)
                self._consecutive_failures = 0
                failure_start = None

                # Only update the forming bar in the buffer.
                # Closed bars are managed by the buffer's own append logic;
                # re-appending all bars every tick would corrupt the ordering
                # because appendleft() inserts at the front each time.
                if bars:
                    self._buffer.update_forming(bars[0])
                    # Promote newly-closed bars: if the previous forming bar's
                    # ts_open no longer matches bars[0], it has closed — append it.
                    if len(bars) > 1:
                        self._buffer.append(bars[1])

                self.frame_ready.emit(bars)

            except DataSourceTransientError as exc:
                logger.warning("RefreshLoop transient error: %s", exc)
                self._consecutive_failures += 1
                if failure_start is None:
                    failure_start = time.monotonic()
                elapsed = time.monotonic() - failure_start
                if elapsed >= self._failure_threshold_s:
                    self.status_changed.emit("数据延迟")
            except Exception as exc:  # noqa: BLE001
                # Never let unexpected exceptions bubble out of the thread
                logger.error("RefreshLoop unexpected error: %s", exc, exc_info=True)

            # Sleep for the remainder of the interval
            elapsed_ms = (time.monotonic() - t0) * 1000
            sleep_ms = max(0.0, self._interval_ms - elapsed_ms)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)
