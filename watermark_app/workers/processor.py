"""Worker thread for image processing tasks."""

import time
import traceback
from pathlib import Path
from PySide6.QtCore import QThread, Signal, QObject
from dataclasses import dataclass, field
from typing import Optional, List
import logging

from watermark_app.core.pipeline import process, ProcessResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessTask:
    input_path: str
    output_path: str = ''
    mode: str = 'corner'
    reference_path: str = ''
    corner: str = 'bottom-right'
    region: tuple = None
    mask_path: str = ''
    backend: str = 'lama'
    scan_pct: float = 0.18
    padding: int = 15
    max_size_mb: float = 5.0
    fixed_position: tuple = None


class CancelToken:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def reset(self):
        self.cancelled = False


class WorkerProcess(QThread):
    """Background worker thread for processing images."""

    # Signals
    progress = Signal(float, str)  # (progress_0_to_1, message)
    file_started = Signal(str)     # input_path
    file_completed = Signal(str, object)  # input_path, ProcessResult
    file_failed = Signal(str, str)  # input_path, error_message
    batch_completed = Signal(int, int, int)  # success, skipped, failed
    all_done = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tasks: List[ProcessTask] = []
        self._cancel_token = CancelToken()
        self._paused = False
        self._current_idx = 0

    def set_tasks(self, tasks: List[ProcessTask]):
        self._tasks = tasks
        self._current_idx = 0

    def cancel(self):
        self._cancel_token.cancel()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    @property
    def is_paused(self):
        return self._paused

    def run(self):
        success_count = 0
        skip_count = 0
        fail_count = 0
        total = len(self._tasks)

        for i, task in enumerate(self._tasks):
            self._current_idx = i

            # Check cancel
            if self._cancel_token.cancelled:
                break

            # Check pause
            while self._paused and not self._cancel_token.cancelled:
                self.msleep(100)

            if self._cancel_token.cancelled:
                break

            self.file_started.emit(task.input_path)

            try:
                base_progress = i / total
                file_progress_range = 1.0 / total

                def file_progress(pct, msg):
                    self.progress.emit(base_progress + pct * file_progress_range, msg)

                result = process(
                    input_path=task.input_path,
                    output_path=task.output_path,
                    mode=task.mode,
                    reference_path=task.reference_path,
                    corner=task.corner,
                    region=task.region,
                    mask_path=task.mask_path,
                    backend=task.backend,
                    scan_pct=task.scan_pct,
                    padding=task.padding,
                    max_size_mb=task.max_size_mb,
                    progress_callback=file_progress,
                    cancel_token=self._cancel_token,
                    fixed_position=task.fixed_position,
                )

                if result.status == 'cancelled':
                    skip_count += 1
                elif result.status == 'success':
                    success_count += 1
                elif result.status == 'no_watermark':
                    skip_count += 1
                else:
                    fail_count += 1

                self.file_completed.emit(task.input_path, result)

            except Exception as e:
                logger.error(f"Worker error on {task.input_path}: {e}\n{traceback.format_exc()}")
                fail_count += 1
                self.file_failed.emit(task.input_path, str(e))

        self.batch_completed.emit(success_count, skip_count, fail_count)
        self.all_done.emit()
