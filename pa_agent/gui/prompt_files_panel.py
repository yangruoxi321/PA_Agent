"""Sidebar panel: which .txt prompt files were sent to AI in the latest run."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _fill_list(widget: QListWidget, files: list[str], *, empty_hint: str) -> None:
    widget.clear()
    if not files:
        item = QListWidgetItem(empty_hint)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        widget.addItem(item)
        return
    for i, name in enumerate(files, 1):
        widget.addItem(QListWidgetItem(f"{i}. {name}"))


class PromptFilesPanel(QWidget):
    """Shows ordered .txt files injected into Stage 1 / Stage 2 for the latest analysis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        hint = QLabel("本次分析注入到 system 提示词中的 .txt 文件（按发送顺序）")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # user 消息动态注入（基本面/资金流/宏观/分析方法走 user 消息，非 system txt）
        dyn_title = QLabel("user 消息动态注入（阶段一，非 txt）")
        dyn_title.setStyleSheet("font-weight: bold; color: #3fb950;")
        layout.addWidget(dyn_title)
        self._dynamic_label = QLabel("（尚未开始）")
        self._dynamic_label.setWordWrap(True)
        self._dynamic_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._dynamic_label)

        s1_title = QLabel("阶段一 · 市场诊断")
        s1_title.setStyleSheet("font-weight: bold; color: #a371f7;")
        layout.addWidget(s1_title)

        self._stage1_list = QListWidget()
        self._stage1_list.setObjectName("promptFileList")
        layout.addWidget(self._stage1_list, stretch=1)

        s2_title = QLabel("阶段二 · 交易决策")
        s2_title.setStyleSheet("font-weight: bold; color: #58a6ff;")
        layout.addWidget(s2_title)

        self._stage2_list = QListWidget()
        self._stage2_list.setObjectName("promptFileList")
        layout.addWidget(self._stage2_list, stretch=1)

        self._extra_label = QLabel("")
        self._extra_label.setObjectName("mutedLabel")
        self._extra_label.setWordWrap(True)
        layout.addWidget(self._extra_label)

        self.clear()

    def set_stage1_files(self, files: list[str]) -> None:
        _fill_list(self._stage1_list, files, empty_hint="（尚未开始阶段一）")

    def set_stage2_files(self, files: list[str]) -> None:
        _fill_list(self._stage2_list, files, empty_hint="（阶段二尚未开始）")

    def set_extras(
        self,
        *,
        stage1_builtin: bool = True,
        stage2_builtin: bool = False,
        experience_count: int = 0,
    ) -> None:
        parts: list[str] = []
        if stage1_builtin:
            parts.append("阶段一另含内置 JSON 输出格式说明（非 txt）")
        if stage2_builtin:
            parts.append("阶段二另含内置 JSON 决策契约（非 txt）")
        if experience_count > 0:
            parts.append(f"阶段二另注入经验库 {experience_count} 条（非 txt）")
        self._extra_label.setText(" · ".join(parts))

    def set_dynamic_injection(self, stage1_user_content: str) -> None:
        """从阶段一真实 user 消息检测动态注入项，逐项显示 ✅/❌。"""
        c = stage1_user_content or ""
        checks = [
            ("基本面分析方法.txt（分析方法引导）", "以下为程序抓取的基本面" in c),
            ("个股基本面（估值/财报…）",
             ("## 基本面" in c) or ("### 估值" in c) or ("估值现状" in c)),
            ("主力资金流（特大/大/中/小单）", "主力资金流" in c),
            ("量价资金面", "量价资金面" in c),
            ("宏观环境", "宏观环境" in c),
        ]
        if not c:
            self._dynamic_label.setText("（尚未开始）")
            return
        lines = [("✅ " if ok else "❌ ") + name for name, ok in checks]
        self._dynamic_label.setText("\n".join(lines))

    def clear(self) -> None:
        self.set_stage1_files([])
        self.set_stage2_files([])
        self.set_extras(stage1_builtin=False, stage2_builtin=False, experience_count=0)
        self._dynamic_label.setText("（尚未开始）")

    def set_latest_run(
        self,
        stage1_files: list[str],
        stage2_files: list[str],
        *,
        experience_count: int = 0,
    ) -> None:
        self.set_stage1_files(stage1_files)
        self.set_stage2_files(stage2_files)
        self.set_extras(
            stage1_builtin=bool(stage1_files),
            stage2_builtin=bool(stage2_files),
            experience_count=experience_count,
        )
