from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter


class InteractiveGraphicsView(QGraphicsView):
    """
    A professional, CAD-style QGraphicsView with:
      - Smooth cursor-centered mouse-wheel zooming (clamped).
      - Middle-mouse-button panning with hand cursor feedback.
      - Anti-aliased rendering.
    """

    ZOOM_MIN = 0.05   # 5% of original size
    ZOOM_MAX = 20.0    # 2000% of original size
    ZOOM_FACTOR = 1.15 # 15% step per wheel notch

    def __init__(self, parent=None):
        super().__init__(parent)

        # Rendering quality
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )

        # Disable the default scroll-bar driven scrolling so we
        # control navigation entirely through the custom events.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        # Internal state for panning
        self._panning = False
        self._pan_start: QPointF = QPointF()

    # ------------------------------------------------------------------
    # Zoom  (Mouse Wheel — anchored to cursor position)
    # ------------------------------------------------------------------
    def wheelEvent(self, event):
        # Current zoom level (assumes uniform scaling)
        current_scale = self.transform().m11()

        # Determine zoom direction
        if event.angleDelta().y() > 0:
            factor = self.ZOOM_FACTOR
        else:
            factor = 1.0 / self.ZOOM_FACTOR

        # Clamp: prevent zooming beyond limits
        target_scale = current_scale * factor
        if target_scale < self.ZOOM_MIN:
            factor = self.ZOOM_MIN / current_scale
        elif target_scale > self.ZOOM_MAX:
            factor = self.ZOOM_MAX / current_scale

        # Anchor zoom to the exact mouse-cursor position in scene coords
        cursor_scene_pos = self.mapToScene(event.position().toPoint())

        # Apply the scale
        self.scale(factor, factor)

        # After scaling, find where the old cursor scene-point now maps
        # on the viewport, and translate the view so it stays under the cursor.
        new_cursor_scene_pos = self.mapToScene(event.position().toPoint())
        delta = new_cursor_scene_pos - cursor_scene_pos
        self.translate(delta.x(), delta.y())

    # ------------------------------------------------------------------
    # Pan  (Middle Mouse Button drag)
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            # Delta in viewport pixels → translate in scene coordinates
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.translate(
                delta.x() / self.transform().m11(),
                delta.y() / self.transform().m22(),
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)
