
import gc
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import QLabel, QMainWindow





def _to_qimage_rgb(img: np.ndarray) -> Optional[QImage]:
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
        return None
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, c = img.shape
    if c == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    if c == 4:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
    return None



class ProjectDisplay(QMainWindow):
    """
    Fullscreen window pinned to a target screen.
    - update_image(np.ndarray BGR/BGRA) scales to fill while keeping AR
    - show_image_fullscreen_on_second_monitor(image, H) applies homography (if 3x3)
    - show_solid_fullscreen((r,g,b)) paints absolute white (or any color) at projector res
    """

    def __init__(self, screen, parent=None):
        super().__init__(parent)
        self.screen = screen


        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(False)  
        self.setCentralWidget(self.label)
        self._last_target_size = None



        geom: QRect = screen.geometry()
        self.move(geom.topLeft())
        self.resize(geom.size())

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(0, 0, 0))
        self.setPalette(pal)
        self.setAutoFillBackground(True)


        self.showFullScreen()
        self.raise_()
        self.activateWindow()

    def update_image(self, image_bgr_or_bgra: np.ndarray):
        try:
            qimg = _to_qimage_rgb(image_bgr_or_bgra)
            if qimg is None:
                print("update_image: invalid image input"); return
            pm = QPixmap.fromImage(qimg)

            target = self.size()
            if self._last_target_size != target:
                self._last_target_size = target
            scaled = pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled)

            if not self.isVisible():
                self.showFullScreen()
        except Exception as e:
            print(f"update_image failed: {e}")

    def _proj_size(self):
        g = self.screen.geometry()
        return g.width(), g.height()

    def show_image_fullscreen_on_second_monitor(self, image_bgr: np.ndarray, homography_matrix=None):
        try:
            img = image_bgr
            if isinstance(homography_matrix, np.ndarray) and homography_matrix.shape == (3, 3):
                W, H = self._proj_size()
                img = cv2.warpPerspective(
                    img, homography_matrix, (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
                )
            self.update_image(img)
        except Exception as e:
            print(f"show_image_fullscreen_on_second_monitor error: {e}")

    def show_solid_fullscreen(self, color=(255, 255, 255)):
        try:
            W, H = self._proj_size()
            qimg = QImage(W, H, QImage.Format_RGB32)
            qimg.fill(QColor(*color))
            self.label.setPixmap(QPixmap.fromImage(qimg))
        except Exception as e:
            print(f"show_solid_fullscreen error: {e}")

    def closeEvent(self, event):
        try:
            self.label.clear()
            self.label.setPixmap(QPixmap())
            print("ProjectDisplay resources cleaned up.")
        except Exception as e:
            print(f"Error during ProjectDisplay cleanup: {e}")
        super().closeEvent(event)

