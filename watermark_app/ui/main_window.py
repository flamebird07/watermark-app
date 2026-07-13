"""Main application window with three-column layout."""

import os
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
    QProgressBar, QFileDialog, QListView,
    QGroupBox, QFormLayout, QCheckBox, QLineEdit, QStatusBar,
    QToolBar, QMessageBox, QAbstractItemView, QFrame, QSlider
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
import logging

from watermark_app.core.presets import PRESETS, list_presets
from watermark_app.workers.processor import WorkerProcess, ProcessTask, CancelToken
from watermark_app.ui.preview_canvas import PreviewCanvas
from watermark_app.ui.task_model import TaskModel, TaskItem

logger = logging.getLogger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水印去除工具")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)
        self.setAcceptDrops(True)

        self._worker: WorkerProcess = None
        self._task_model = TaskModel(self)
        self._current_preview_idx = -1
        self._fixed_position = None

        self._init_ui()
        # Default to 豆包右下角 preset (most common use case)
        # Must be AFTER all UI controls are created
        self._preset_combo.setCurrentIndex(1)  # index 1 = douyin_bottom_right
        self._on_preset_changed(1)

        self._connect_signals()

    def _init_ui(self):
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件(&F)")
        open_action = QAction("打开图片(&O)", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_files)
        file_menu.addAction(open_action)

        open_dir_action = QAction("打开文件夹(&D)", self)
        open_dir_action.setShortcut("Ctrl+Shift+O")
        open_dir_action.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_dir_action)

        file_menu.addSeparator()
        exit_action = QAction("退出(&Q)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Toolbar
        toolbar = QToolBar("工具栏")
        self.addToolBar(toolbar)
        toolbar.addAction("添加图片", self._on_open_files)
        toolbar.addAction("添加文件夹", self._on_open_folder)
        toolbar.addAction("清空列表", self._on_clear_list)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Splitter for three columns
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # Left: file list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("文件列表"))

        self._file_list = QListView()
        self._file_list.setModel(self._task_model)
        self._file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._file_list.selectionModel().currentChanged.connect(self._on_list_select)
        left_layout.addWidget(self._file_list)

        btn_row = QHBoxLayout()
        btn_row.addWidget(QPushButton("添加图片", clicked=self._on_open_files))
        btn_row.addWidget(QPushButton("添加文件夹", clicked=self._on_open_folder))
        btn_row.addWidget(QPushButton("清空", clicked=self._on_clear_list))
        left_layout.addLayout(btn_row)

        splitter.addWidget(left_panel)

        # Center: preview
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # View mode buttons
        view_row = QHBoxLayout()
        self._btn_original = QPushButton("原图", clicked=lambda: self._set_view_mode('original'))
        self._btn_mask = QPushButton("掩膜", clicked=lambda: self._set_view_mode('mask'))
        self._btn_result = QPushButton("结果", clicked=lambda: self._set_view_mode('result'))
        self._btn_compare = QPushButton("对比", clicked=lambda: self._set_view_mode('compare'))
        for btn in [self._btn_original, self._btn_mask, self._btn_result, self._btn_compare]:
            btn.setCheckable(True)
            view_row.addWidget(btn)
        self._btn_original.setChecked(True)

        # Comparison slider
        self._compare_slider = QSlider(Qt.Orientation.Horizontal)
        self._compare_slider.setRange(0, 1000)
        self._compare_slider.setValue(500)
        self._compare_slider.setVisible(False)
        self._compare_slider.valueChanged.connect(
            lambda v: self._preview.set_comparison(v / 1000.0))

        center_layout.addLayout(view_row)
        center_layout.addWidget(self._compare_slider)

        self._preview = PreviewCanvas()
        center_layout.addWidget(self._preview, stretch=1)

        splitter.addWidget(center_panel)

        # Right: parameters
        right_panel = QWidget()
        right_panel.setMaximumWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)

        # Preset selection
        preset_group = QGroupBox("预设")
        preset_form = QFormLayout(preset_group)
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("自定义", "custom")
        for key, info in PRESETS.items():
            self._preset_combo.addItem(info['name'], key)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_form.addRow("预设:", self._preset_combo)
        right_layout.addWidget(preset_group)

        # Mode settings
        mode_group = QGroupBox("处理模式")
        mode_form = QFormLayout(mode_group)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["角落检测", "模板匹配", "区域指定", "外部掩膜", "自动检测"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_form.addRow("模式:", self._mode_combo)

        self._corner_combo = QComboBox()
        self._corner_combo.addItems(["右下角", "左下角", "右上角", "左上角"])
        mode_form.addRow("角落:", self._corner_combo)

        self._ref_path = QLineEdit()
        self._ref_path.setPlaceholderText("选择参考图片...")
        ref_row = QHBoxLayout()
        ref_row.addWidget(self._ref_path)
        ref_btn = QPushButton("...", clicked=self._on_select_ref)
        ref_btn.setMaximumWidth(30)
        ref_row.addWidget(ref_btn)
        mode_form.addRow("参考图:", ref_row)

        self._mask_path_edit = QLineEdit()
        self._mask_path_edit.setPlaceholderText("选择掩膜文件...")
        mask_row = QHBoxLayout()
        mask_row.addWidget(self._mask_path_edit)
        mask_btn = QPushButton("...", clicked=self._on_select_mask)
        mask_btn.setMaximumWidth(30)
        mask_row.addWidget(mask_btn)
        mode_form.addRow("掩膜:", mask_row)

        # Region input widget (only visible in region mode)
        self._region_widget = QWidget()
        region_layout = QHBoxLayout(self._region_widget)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.setSpacing(4)
        self._region_x = QSpinBox()
        self._region_x.setRange(0, 99999)
        self._region_x.setPrefix("x:")
        self._region_y = QSpinBox()
        self._region_y.setRange(0, 99999)
        self._region_y.setPrefix("y:")
        self._region_w = QSpinBox()
        self._region_w.setRange(0, 99999)
        self._region_w.setPrefix("w:")
        self._region_h = QSpinBox()
        self._region_h.setRange(0, 99999)
        self._region_h.setPrefix("h:")
        region_layout.addWidget(self._region_x)
        region_layout.addWidget(self._region_y)
        region_layout.addWidget(self._region_w)
        region_layout.addWidget(self._region_h)
        self._region_widget.setVisible(False)
        mode_form.addRow("区域:", self._region_widget)

        right_layout.addWidget(mode_group)

        # Processing settings
        proc_group = QGroupBox("处理参数")
        proc_form = QFormLayout(proc_group)

        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["LaMa (高质量)", "OpenCV (快速)"])
        proc_form.addRow("后端:", self._backend_combo)

        self._scan_pct = QDoubleSpinBox()
        self._scan_pct.setRange(0.05, 1.0)
        self._scan_pct.setSingleStep(0.01)
        self._scan_pct.setValue(0.18)
        self._scan_pct.setSuffix("  (占比)")
        proc_form.addRow("扫描比例:", self._scan_pct)

        self._padding_spin = QSpinBox()
        self._padding_spin.setRange(0, 50)
        self._padding_spin.setValue(15)
        self._padding_spin.setSuffix(" px")
        proc_form.addRow("掩膜扩张:", self._padding_spin)

        self._max_size = QDoubleSpinBox()
        self._max_size.setRange(0.5, 50.0)
        self._max_size.setSingleStep(0.5)
        self._max_size.setValue(5.0)
        self._max_size.setSuffix(" MB")
        proc_form.addRow("最大文件:", self._max_size)

        right_layout.addWidget(proc_group)

        # Output settings
        out_group = QGroupBox("输出设置")
        out_form = QFormLayout(out_group)

        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("默认: 同目录/cleaned/")
        out_row = QHBoxLayout()
        out_row.addWidget(self._output_dir)
        out_btn = QPushButton("...", clicked=self._on_select_output)
        out_btn.setMaximumWidth(30)
        out_row.addWidget(out_btn)
        out_form.addRow("输出目录:", out_row)

        self._keep_format = QCheckBox("保持原格式")
        self._keep_format.setChecked(True)
        out_form.addRow("", self._keep_format)

        right_layout.addWidget(out_group)
        right_layout.addStretch()

        splitter.addWidget(right_panel)
        splitter.setSizes([250, 700, 300])

        # Bottom: progress and controls
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 5, 0, 0)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p% - %v")
        bottom_layout.addWidget(self._progress_bar, stretch=3)

        self._status_label = QLabel("就绪")
        self._status_label.setMinimumWidth(120)
        bottom_layout.addWidget(self._status_label)

        self._time_label = QLabel("")
        bottom_layout.addWidget(self._time_label)

        self._count_label = QLabel("0/0")
        bottom_layout.addWidget(self._count_label)

        self._btn_start = QPushButton("开始处理")
        self._btn_start.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; "
                                       "font-weight: bold; padding: 8px 20px; }")
        self._btn_start.clicked.connect(self._on_start)
        bottom_layout.addWidget(self._btn_start)

        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        bottom_layout.addWidget(self._btn_cancel)

        main_layout.addWidget(bottom)

        # Status bar
        self.statusBar().showMessage("就绪")

    def _connect_signals(self):
        self._preview.comparison_changed.connect(self._on_compare_changed)

    def _on_open_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp);;所有文件 (*)"
        )
        if files:
            self._add_files(files)

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            images = []
            for f in Path(folder).rglob('*'):
                if 'cleaned' in f.parts:
                    continue
                if f.suffix.lower() in IMAGE_EXTS:
                    images.append(str(f))
            if images:
                self._add_files(images)
            else:
                self.statusBar().showMessage("文件夹中没有找到图片文件")

    def _add_files(self, files):
        existing = {self._task_model.get_task(i).path
                    for i in range(self._task_model.rowCount())}
        new_files = [f for f in files if f not in existing]
        self._task_model.add_tasks(new_files)
        self._count_label.setText(f"{self._task_model.rowCount()} 个文件")
        self.statusBar().showMessage(f"添加了 {len(new_files)} 个文件")

    def _on_clear_list(self):
        self._task_model.clear()
        self._preview.clear()
        self._count_label.setText("0 个文件")
        self._progress_bar.setValue(0)

    def _on_list_select(self, current, previous=None):
        if isinstance(current, int):
            row = current
        else:
            row = current.row() if current and current.isValid() else -1
        if row < 0:
            return
        task = self._task_model.get_task(row)
        if task:
            self._current_preview_idx = row
            from watermark_app.core.io import cv2_read
            img = cv2_read(task.path)
            if img is not None:
                self._preview.set_image(img)
            if task.result and hasattr(task.result, 'mask_generated') and task.result.mask_generated is not None:
                self._preview.set_mask(task.result.mask_generated)
            if task.output_path and Path(task.output_path).exists():
                result_img = cv2_read(task.output_path)
                if result_img is not None:
                    self._preview.set_result(result_img)

    def _set_view_mode(self, mode):
        self._preview.set_mode(mode)
        for btn in [self._btn_original, self._btn_mask, self._btn_result, self._btn_compare]:
            btn.setChecked(False)
        if mode == 'original':
            self._btn_original.setChecked(True)
        elif mode == 'mask':
            self._btn_mask.setChecked(True)
        elif mode == 'result':
            self._btn_result.setChecked(True)
        elif mode == 'compare':
            self._btn_compare.setChecked(True)
        self._compare_slider.setVisible(mode == 'compare')

    def _on_compare_changed(self, pos):
        pass  # Handled by PreviewCanvas

    def _on_preset_changed(self, index):
        key = self._preset_combo.currentData()
        if key == 'custom':
            self._fixed_position = None
            return
        preset = PRESETS.get(key)
        if not preset:
            return
        mode_map = {"corner": 0, "template": 1, "region": 2, "mask": 3, "auto": 4}
        self._mode_combo.setCurrentIndex(mode_map.get(preset.get('mode', 'corner'), 0))

        corner_map = {"bottom-right": 0, "bottom-left": 1, "top-right": 2, "top-left": 3}
        self._corner_combo.setCurrentIndex(corner_map.get(preset.get('corner', 'bottom-right'), 0))

        if 'scan_pct' in preset:
            self._scan_pct.setValue(preset['scan_pct'])
        if 'padding' in preset:
            self._padding_spin.setValue(preset['padding'])

        backend_idx = 0 if preset.get('backend', 'lama') == 'lama' else 1
        self._backend_combo.setCurrentIndex(backend_idx)

        self._fixed_position = preset.get('fixed_position')

    def _on_mode_changed(self, index):
        modes = ['corner', 'template', 'region', 'mask', 'auto']
        mode = modes[index] if index < len(modes) else 'corner'
        self._corner_combo.setEnabled(mode == 'corner')
        self._ref_path.setEnabled(mode == 'template')
        self._mask_path_edit.setEnabled(mode == 'mask')
        self._region_widget.setVisible(mode == 'region')

    def _on_select_ref(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择参考图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)")
        if path:
            self._ref_path.setText(path)

    def _on_select_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择掩膜文件", "",
            "掩膜文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)")
        if path:
            self._mask_path_edit.setText(path)

    def _on_select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder:
            self._output_dir.setText(folder)

    def _build_tasks(self):
        modes = ['corner', 'template', 'region', 'mask', 'auto']
        mode = modes[self._mode_combo.currentIndex()]
        corners = ['bottom-right', 'bottom-left', 'top-right', 'top-left']
        corner = corners[self._corner_combo.currentIndex()]
        backend = 'lama' if self._backend_combo.currentIndex() == 0 else 'opencv'
        output_dir = self._output_dir.text().strip()

        tasks = []
        for i in range(self._task_model.rowCount()):
            task_item = self._task_model.get_task(i)
            if task_item.status in ('success', 'processing'):
                continue

            output_path = ''
            if output_dir:
                out_p = Path(output_dir)
                out_p.mkdir(parents=True, exist_ok=True)
                output_path = str(out_p / Path(task_item.path).name)

            task = ProcessTask(
                input_path=task_item.path,
                output_path=output_path,
                mode=mode,
                reference_path=self._ref_path.text().strip(),
                corner=corner,
                region=(self._region_x.value(), self._region_y.value(),
                        self._region_w.value(), self._region_h.value()) if mode == 'region' else None,
                mask_path=self._mask_path_edit.text().strip(),
                backend=backend,
                scan_pct=self._scan_pct.value(),
                padding=self._padding_spin.value(),
                max_size_mb=self._max_size.value(),
                fixed_position=self._fixed_position,
            )
            tasks.append((i, task))
        return tasks

    def _on_start(self):
        tasks = self._build_tasks()
        if not tasks:
            self.statusBar().showMessage("没有待处理的文件")
            return

        self._worker = WorkerProcess(self)
        self._worker.set_tasks([t[1] for t in tasks])
        self._task_indices = {t[1].input_path: t[0] for t in tasks}

        self._worker.progress.connect(self._on_progress)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_completed.connect(self._on_file_completed)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.batch_completed.connect(self._on_batch_completed)
        self._worker.all_done.connect(self._on_all_done)

        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._start_time = __import__('time').time()

        self._worker.start()
        self.statusBar().showMessage("开始处理...")

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
            self.statusBar().showMessage("正在取消...")

    def _on_progress(self, pct, msg):
        self._progress_bar.setValue(int(pct * 100))
        self._status_label.setText(msg)
        elapsed = __import__('time').time() - self._start_time
        self._time_label.setText(f"耗时: {elapsed:.1f}s")

    def _on_file_started(self, path):
        idx = self._task_indices.get(path, -1)
        if idx >= 0:
            self._task_model.update_status(idx, 'processing')

    def _on_file_completed(self, path, result):
        idx = self._task_indices.get(path, -1)
        if idx < 0:
            return
        task_item = self._task_model.get_task(idx)
        task_item.result = result

        if result.status == 'success':
            self._task_model.update_status(idx, 'success', result.output_path)
        elif result.status == 'no_watermark':
            self._task_model.update_status(idx, 'no_watermark')
        elif result.status == 'cancelled':
            self._task_model.update_status(idx, 'skipped')
        else:
            error_msg = result.error_code
            if result.warnings:
                error_msg += ': ' + '; '.join(result.warnings)
            self._task_model.update_status(idx, 'failed', error=error_msg)

        # Update preview if this is the selected item
        if idx == self._current_preview_idx:
            self._on_list_select(idx)

    def _on_file_failed(self, path, error):
        idx = self._task_indices.get(path, -1)
        if idx >= 0:
            self._task_model.update_status(idx, 'failed', error=error)

    def _on_batch_completed(self, success, skipped, failed):
        self.statusBar().showMessage(
            f"批处理完成: 成功 {success}, 跳过 {skipped}, 失败 {failed}")

    def _on_all_done(self):
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress_bar.setValue(100)
        elapsed = __import__('time').time() - self._start_time
        self._time_label.setText(f"总耗时: {elapsed:.1f}s")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in IMAGE_EXTS:
                files.append(path)
            elif Path(path).is_dir():
                for f in Path(path).rglob('*'):
                    if 'cleaned' in f.parts:
                        continue
                    if f.suffix.lower() in IMAGE_EXTS:
                        files.append(str(f))
        if files:
            self._add_files(files)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.requestInterruption()

            # Wait up to 8 seconds for graceful shutdown
            if not self._worker.wait(8000):
                reply = QMessageBox.warning(
                    self, '任务仍在运行',
                    '后台任务未能在8秒内结束。\n是否强制终止？（可能导致当前文件处理不完整）',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._worker.terminate()
                    self._worker.wait(2000)
                else:
                    event.ignore()
                    return
        event.accept()
