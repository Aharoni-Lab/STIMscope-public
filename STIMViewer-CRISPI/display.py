# display.py
import math

try:
    from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QWidget
    from PyQt5.QtGui import QImage, QPainter, QTransform
    from PyQt5.QtCore import QRectF, Qt
    from PyQt5.QtCore import pyqtSlot
except ImportError:
    from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QWidget
    from PyQt5.QtGui import QImage, QPainter
    from PyQt5.QtCore import QRectF
    from PyQt5.QtCore import pyqtSlot


class Display(QGraphicsView):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.__scene = CustomGraphicsScene(self)
        self.setScene(self.__scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def set_zoom(self, zoom_factor):
        """Scales the display while keeping aspect ratio."""
        self.setTransform(QTransform().scale(zoom_factor, zoom_factor))


    def on_image_received(self, image: QImage):
        self.__scene.set_image(image)
        self.update()

    def on_mask_received(self, mask, _=None):
        self.__scene.set_mask(mask)
    
class CustomGraphicsScene(QGraphicsScene):
    def __init__(self, parent: Display = None):
        super().__init__(parent)
        self.__parent = parent
        self.__image = QImage()
        # Add new layer for mask
        self.__mask = QImage()

    def set_image(self, image: QImage):
        self.__image = image
        self.update()
    
    def set_mask(self, mask: QImage):
        self.__mask = mask
        self.update()

    def drawBackground(self, painter: QPainter, rect: QRectF):
        # Display size
        display_width = self.__parent.width()
        display_height = self.__parent.height()

        # Image size
        image_width = self.__image.width()
        image_height = self.__image.height()

        # Return if we don't have an image yet
        if image_width == 0 or image_height == 0:
            return

        # Calculate aspect ratio of display
        ratio1 = display_width / display_height
        # Calculate aspect ratio of image
        ratio2 = image_width / image_height

        if ratio1 > ratio2:
            # The height must fit to the display height. So h remains and w must be scaled down
            image_width = display_height * ratio2
            image_height = display_height
        else:
            # The image width must fit to the display width. So w remains and h must be scaled down
            image_width = display_width
            image_height = display_width / ratio2

        image_pos_x = -1.0 * (image_width / 2.0)
        image_pos_y = -1.0 * (image_height / 2.0)

        # Remove digits after point
        image_pos_x = math.trunc(image_pos_x)
        image_pos_y = math.trunc(image_pos_y)

        rect = QRectF(image_pos_x, image_pos_y, image_width, image_height)
        painter.drawImage(rect, self.__image)

        # then draw the mask on top, with a translucent red tint
        if not self.__mask.isNull():
            mask_scaled = self.__mask.scaled(
                int(rect.width()), int(rect.height()),
                Qt.KeepAspectRatio, Qt.FastTransformation)
            painter.setOpacity(0.3)
            painter.drawImage(rect.topLeft(), mask_scaled)
            painter.setOpacity(1.0)