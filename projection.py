# projection.py
import cv2
import numpy as np

from PyQt5.QtWidgets import  QLabel, QMainWindow
from PyQt5.QtGui import QPixmap, QImage

app = None
image_window = None  # Store the window globally

class ProjectDisplay(QMainWindow):
    def __init__(self, screen):
        super().__init__()

        self.screen = screen
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        self.move(screen.geometry().x(), screen.geometry().y())
        self.resize(screen.geometry().width(), screen.geometry().height())
        self.showFullScreen()

    def update_image(self, image):
        """Updates the displayed image."""
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimage))

    def show_image_fullscreen_on_second_monitor(self, image, homography_matrix=None):
        """Displays an image on the second monitor using PyQt without threading issues."""

        if homography_matrix is None:
            homography_matrix = np.eye(3)

        # Apply Homography Transformation
        image_transformed = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))
        image_transformed = cv2.cvtColor(image_transformed, cv2.COLOR_BGR2RGB)  # Convert to RGB

        image_transformed = cv2.flip(image_transformed, 1)

        # Directly update the image (since we're in the GUI thread)
        self.update_image(image_transformed)



  