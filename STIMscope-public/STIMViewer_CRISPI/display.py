
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
        # Avoid device-coordinate caching which can explode memory when zooming
        self._img_item.setCacheMode(QtWidgets.QGraphicsItem.NoCache)
        self._scene.addItem(self._img_item)

        self._mask_item = QtWidgets.QGraphicsPixmapItem()
        self._mask_item.setOpacity(0.30)
        self._mask_item.setVisible(False)
        self._mask_item.setZValue(1)
        self._mask_item.setCacheMode(QtWidgets.QGraphicsItem.NoCache)
        self._scene.addItem(self._mask_item)


        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        # Disable smooth scaling to reduce GPU/CPU load when zooming
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, on=False)
        # Minimize repaint work while zooming/panning
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.MinimalViewportUpdate)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)  # Handle dragging manually
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontSavePainterState, True)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontClipPainter, True)

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
        
        # Mouse drag panning state
        self._panning = False
        self._last_pan_pos = QtCore.QPoint()
        
        # Set default cursor for panning indication
        self.setCursor(QtCore.Qt.OpenHandCursor) 

        # Zoom indicator/reset button (overlay on viewport)
        try:
            self._zoom_btn = QtWidgets.QToolButton(self.viewport())
            self._zoom_btn.setText("100%")
            self._zoom_btn.setToolTip("Click to reset to 100% (Fit to window)")
            self._zoom_btn.setAutoRaise(True)
            self._zoom_btn.setCursor(QtCore.Qt.PointingHandCursor)
            self._zoom_btn.clicked.connect(self._reset_zoom_to_100)
            self._zoom_btn.move(8, 8)
            self._zoom_btn.setStyleSheet("QToolButton{background: rgba(0,0,0,120); color: white; padding: 2px 6px; border-radius: 3px;}")
            self._zoom_btn.show()
        except Exception:
            self._zoom_btn = None

        # Debounce rapid zoom events to avoid excessive transforms/repaints
        self._pending_zoom: float = None  # type: ignore
        self._zoom_timer = QtCore.QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(16)  # ~60 Hz
        self._zoom_timer.timeout.connect(self._flush_zoom)





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
            # Avoid fighting manual zoom with auto fit on every frame
            if not getattr(self, "_user_zoomed", False):
                self._apply_zoom_fit(center=False)

        if not getattr(self, "_nudged_for_scrollbars", False):
            self._nudged_for_scrollbars = True
            self.set_zoom(self._zoom * 1.001)
        self._update_zoom_indicator()

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
        # Mark that user adjusted zoom; prevents auto-fit thrash
        try:
            self._user_zoomed = True
        except Exception:
            self._user_zoomed = True
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
        # Guard against extreme transforms causing huge pixmap allocs
        base = self._fit_scale()
        eff  = max(0.05, min(20.0, base * self._zoom))
        # Skip tiny changes to reduce churn
        if self._last_eff_scale is not None and abs(self._last_eff_scale - eff) < 1e-3 and not center:
            return  
        self._last_eff_scale = eff

        try:
            t = QtGui.QTransform()
            t.scale(eff, eff)
            self.setTransform(t, combine=False)
        except Exception:
            # Fallback to identity transform if scale overflows
            self.setTransform(QtGui.QTransform(), combine=False)

        if center and self._have_image:
            self.centerOn(self._img_item)
        self._update_zoom_indicator()

    def wheelEvent(self, ev: QtGui.QWheelEvent):
        # Mouse wheel zoom (no Ctrl key needed) with guards to prevent spikes
        if not self._have_image or self._last_img_w <= 0 or self._last_img_h <= 0:
            ev.ignore()
            return
        try:
            step = ev.angleDelta().y() / 120.0
            # Cap per-event zoom factor to avoid extreme jumps
            factor = 1.1 ** max(-3.0, min(3.0, step))
            # Schedule zoom apply (debounced) to avoid thrashing during rapid wheel events
            try:
                z = float(self._zoom) * float(factor)
            except Exception:
                z = self._zoom
            z = max(0.1, min(10.0, z))
            self._pending_zoom = z
            try:
                self._user_zoomed = True
            except Exception:
                self._user_zoomed = True
            # Restart timer; only last zoom in a burst is applied
            self._zoom_timer.start(16)
            ev.accept()
        except Exception:
            ev.ignore()

    def _flush_zoom(self):
        try:
            if self._pending_zoom is None:
                return
            # Apply once per burst
            self.set_zoom(self._pending_zoom)
        finally:
            self._pending_zoom = None

    def _update_zoom_indicator(self):
        try:
            if self._zoom_btn is None:
                return
            # 100% corresponds to 'fit-to-window' scale
            pct = int(round(self._zoom * 100.0))
            self._zoom_btn.setText(f"{pct}%")
            # keep in corner
            self._zoom_btn.move(8, 8)
        except Exception:
            pass

    def _reset_zoom_to_100(self):
        try:
            # Reset zoom to 1.0 → fit-to-window (100%)
            self._zoom = 1.0
            # Clear manual flag on reset so auto-fit is allowed
            try:
                self._user_zoomed = False
            except Exception:
                pass
            self._apply_zoom_fit(center=False)
        except Exception:
            pass

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton:
            # Start panning
            self._panning = True
            self._last_pan_pos = ev.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self._panning:
            # Pan the view
            delta = ev.pos() - self._last_pan_pos
            self._last_pan_pos = ev.pos()
            
            # Convert mouse movement to scroll bar movement
            h_scroll = self.horizontalScrollBar()
            v_scroll = self.verticalScrollBar()
            
            h_scroll.setValue(h_scroll.value() - delta.x())
            v_scroll.setValue(v_scroll.value() - delta.y())
            
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton and self._panning:
            # Stop panning
            self._panning = False
            self.setCursor(QtCore.Qt.OpenHandCursor)
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)

    def resizeEvent(self, ev: QtGui.QResizeEvent):
        super().resizeEvent(ev)
        # Avoid heavy rescale on every resize step if user manually zoomed;
        # next image or reset button will re-fit if needed
        if not getattr(self, "_user_zoomed", False):
            self._apply_zoom_fit(center=False)
        self._update_zoom_indicator()
