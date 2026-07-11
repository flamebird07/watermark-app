"""Task list model for managing image processing queue."""

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt
from PySide6.QtGui import QIcon
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TaskItem:
    path: str
    status: str = 'pending'  # pending, processing, success, failed, skipped, no_watermark
    output_path: str = ''
    error: str = ''
    result: object = None

    @property
    def name(self):
        return Path(self.path).name

    @property
    def status_icon(self):
        icons = {
            'pending': '○',
            'processing': '…',
            'success': '✓',
            'failed': '✗',
            'skipped': '—',
            'no_watermark': '?',
        }
        return icons.get(self.status, '○')

    @property
    def status_text(self):
        texts = {
            'pending': '待处理',
            'processing': '处理中',
            'success': '已完成',
            'failed': '失败',
            'skipped': '跳过',
            'no_watermark': '无水印',
        }
        return texts.get(self.status, self.status)


class TaskModel(QAbstractListModel):
    """List model for the task queue."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tasks: List[TaskItem] = []

    def rowCount(self, parent=QModelIndex()):
        return len(self._tasks)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._tasks):
            return None

        task = self._tasks[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return f"{task.status_icon} {task.name}"

        if role == Qt.ItemDataRole.ToolTipRole:
            if task.error:
                return f"{task.name}\n状态: {task.status_text}\n错误: {task.error}"
            return f"{task.name}\n状态: {task.status_text}"

        if role == Qt.ItemDataRole.UserRole:
            return task

        return None

    def add_task(self, path: str):
        """Add a task to the queue."""
        row = len(self._tasks)
        self.beginInsertRows(QModelIndex(), row, row)
        self._tasks.append(TaskItem(path=path))
        self.endInsertRows()

    def add_tasks(self, paths: List[str]):
        """Add multiple tasks."""
        if not paths:
            return
        row = len(self._tasks)
        self.beginInsertRows(QModelIndex(), row, row + len(paths) - 1)
        for p in paths:
            self._tasks.append(TaskItem(path=p))
        self.endInsertRows()

    def clear(self):
        """Clear all tasks."""
        self.beginResetModel()
        self._tasks.clear()
        self.endResetModel()

    def update_status(self, index: int, status: str, output_path: str = '', error: str = ''):
        """Update task status."""
        if 0 <= index < len(self._tasks):
            self._tasks[index].status = status
            self._tasks[index].output_path = output_path
            self._tasks[index].error = error
            self.dataChanged.emit(self.index(index), self.index(index))

    def find_by_path(self, path: str) -> int:
        """Find task index by path."""
        for i, task in enumerate(self._tasks):
            if task.path == path:
                return i
        return -1

    def get_tasks(self) -> List[TaskItem]:
        return self._tasks

    def get_task(self, index: int) -> Optional[TaskItem]:
        if 0 <= index < len(self._tasks):
            return self._tasks[index]
        return None
