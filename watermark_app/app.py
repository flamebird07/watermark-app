"""Watermark removal desktop application entry point."""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QIcon
from PySide6.QtCore import Qt

from watermark_app.ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('watermark_app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("水印去除工具")
    app.setOrganizationName("WatermarkApp")

    # Set app icon
    icon_path = Path(__file__).parent.parent / "assets" / "app.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Dark theme
    app.setStyleSheet("""
        QMainWindow { background-color: #353535; }
        QWidget { color: #e0e0e0; font-size: 13px; }
        QGroupBox { border: 1px solid #555; border-radius: 4px;
                    margin-top: 10px; padding-top: 15px; font-weight: bold; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QListWidget { background-color: #2b2b2b; border: 1px solid #555;
                      alternate-background-color: #303030; }
        QListWidget::item:selected { background-color: #4a90d9; }
        QListWidget::item:hover { background-color: #3a3a3a; }
        QListView { background-color: #2b2b2b; border: 1px solid #555; }
        QListView::item:selected { background-color: #4a90d9; }
        QListView::item:hover { background-color: #3a3a3a; }
        QPushButton { background-color: #4a4a4a; border: 1px solid #666;
                      border-radius: 4px; padding: 5px 12px; }
        QPushButton:hover { background-color: #5a5a5a; }
        QPushButton:pressed { background-color: #3a3a3a; }
        QPushButton:disabled { background-color: #3a3a3a; color: #777; }
        QComboBox { background-color: #3a3a3a; border: 1px solid #555;
                    border-radius: 3px; padding: 4px 8px; }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { background-color: #3a3a3a; selection-background-color: #4a90d9; }
        QSpinBox, QDoubleSpinBox { background-color: #3a3a3a; border: 1px solid #555;
                                   border-radius: 3px; padding: 3px; }
        QLineEdit { background-color: #3a3a3a; border: 1px solid #555;
                    border-radius: 3px; padding: 4px; }
        QProgressBar { border: 1px solid #555; border-radius: 3px;
                       text-align: center; background-color: #2b2b2b; }
        QProgressBar::chunk { background-color: #4CAF50; border-radius: 2px; }
        QToolBar { background-color: #353535; border-bottom: 1px solid #555; spacing: 5px; }
        QToolBar QToolButton { background-color: #4a4a4a; border: 1px solid #666;
                               border-radius: 3px; padding: 4px 8px; }
        QStatusBar { background-color: #2b2b2b; border-top: 1px solid #555; }
        QSlider::groove:horizontal { background: #555; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #4a90d9; width: 14px; margin: -4px 0;
                                     border-radius: 7px; }
        QCheckBox { spacing: 6px; }
        QCheckBox::indicator { width: 16px; height: 16px; }
        QSplitter::handle { background-color: #555; width: 3px; }
        QMenuBar { background-color: #353535; }
        QMenuBar::item:selected { background-color: #4a4a4a; }
        QMenu { background-color: #3a3a3a; }
        QMenu::item:selected { background-color: #4a90d9; }
    """)

    window = MainWindow()
    window.show()

    logger.info("Application started")
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
