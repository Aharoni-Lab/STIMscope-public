
import os
from PyQt5 import QtWidgets, QtGui, QtCore

def _env_true(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

class Display(QtWidgets.QGraphicsView):


    def __init__(self, parent=None):
        super().__init__(parent)


        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
        self.setScene(self._scene)

        self._img_item = QtWidgets.QGraphicsPixmapItem()
        self._img_item.setZValue(0)
        self._img_item.setTransformationMode(QtCore.Qt.FastTransformation)
        self._img_item.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        self._scene.addItem(self._img_item)

        self._mask_item = QtWidgets.QGraphicsPixmapItem()
        self._mask_item.setOpacity(0.30)
        self._mask_item.setVisible(False)
        self._mask_item.setZValue(1)
        self._mask_item.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        self._scene.addItem(self._mask_item)


        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, on=True)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.SmartViewportUpdate)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontSavePainterState, True)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontAdjustForAntialiasing, True)

        if _env_true("STIM_GL_VIEWPORT", False):
            try:
                if QtWidgets.QApplication.instance() is not None:
                    from PyQt5.QtWidgets import QOpenGLWidget
                    self.setViewport(QOpenGLWidget())
                    self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
            except Exception:
                pass

        self._zoom = 1.0            
        self._have_image = False
        self._last_img_w = 0
        self._last_img_h = 0

        self._last_eff_scale = None
        self._nudged_for_scrollbars = False 





    @QtCore.pyqtSlot(QtGui.QImage)
    def on_image_received(self, qimg: QtGui.QImage):
        if not isinstance(qimg, QtGui.QImage) or qimg.isNull():
            return

        try:
            pm = QtGui.QPixmap.fromImage(qimg)
        except Exception:
            try:
                pm = QtGui.QPixmap.fromImage(qimg.convertToFormat(QtGui.QImage.Format_RGB888))
            except Exception:
                return

        if pm.isNull():
            return

        self._img_item.setPixmap(pm)
        self._have_image = True

        br = self._img_item.boundingRect()
        self._scene.setSceneRect(br)

        size_changed = False
        if br.isValid():
            w, h = int(br.width()), int(br.height())
            size_changed = (w != self._last_img_w) or (h != self._last_img_h)
            self._last_img_w, self._last_img_h = w, h
        else:
            return

        if self._mask_item.isVisible():
            self._mask_item.setPos(self._img_item.pos())

        if size_changed:
            self._apply_zoom_fit(center=True)
        else:
            self._apply_zoom_fit(center=False)

        if not getattr(self, "_nudged_for_scrollbars", False):
            self._nudged_for_scrollbars = True
            self.set_zoom(self._zoom * 1.001)

    def setImage(self, qimg: QtGui.QImage):
        self.on_image_received(qimg)


    @QtCore.pyqtSlot(QtGui.QImage)
    def on_mask_received(self, mask: QtGui.QImage):
       
        if isinstance(mask, QtGui.QImage) and not mask.isNull():
            try:
                pm = QtGui.QPixmap.fromImage(mask)
            except Exception:
                try:
                    pm = QtGui.QPixmap.fromImage(mask.convertToFormat(QtGui.QImage.Format_ARGB32))
                except Exception:
                    return
            self._mask_item.setPixmap(pm)
            self._mask_item.setVisible(True)
            self._mask_item.setPos(self._img_item.pos())
        else:
            self._mask_item.setVisible(False)
            self._mask_item.setPixmap(QtGui.QPixmap())

    def set_zoom(self, zoom_factor: float):
        
        try:
            z = float(zoom_factor)
        except Exception:
            return
        z = max(0.1, min(10.0, z))
        self._zoom = z
        self._apply_zoom_fit(center=False)



    def _fit_scale(self) -> float:
       
        if not self._have_image or self._last_img_w <= 0 or self._last_img_h <= 0:
            return 1.0
        vw = max(1, self.viewport().width())
        vh = max(1, self.viewport().height())
        sx = vw / float(self._last_img_w)
        sy = vh / float(self._last_img_h)
        return min(sx, sy)

    def _apply_zoom_fit(self, center: bool):
        base = self._fit_scale()
        eff  = base * self._zoom
        if self._last_eff_scale == eff and not center:
            return  
        self._last_eff_scale = eff

        t = QtGui.QTransform()
        t.scale(eff, eff)
        self.setTransform(t, combine=False)

        if center and self._have_image:
            self.centerOn(self._img_item)

    def wheelEvent(self, ev: QtGui.QWheelEvent):
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            step = ev.angleDelta().y() / 120.0
            self.set_zoom(self._zoom * (1.1 ** step))
            ev.accept()
            return
        super().wheelEvent(ev)

    def resizeEvent(self, ev: QtGui.QResizeEvent):
        super().resizeEvent(ev)
        self._apply_zoom_fit(center=False)
