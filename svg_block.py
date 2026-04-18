"""
╔══════════════════════════════════════════════════════════════╗
║  SVG Block — Importable, movable, resizable vector element  ║
║  for the Plotter Studio canvas.                              ║
║                                                              ║
║  Uses svgelements to parse SVG paths into polylines that     ║
║  can be rendered on canvas and exported to G-Code.           ║
╚══════════════════════════════════════════════════════════════╝
"""
import os
import math
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPathItem
from PyQt6.QtGui import (
    QPen, QColor, QBrush, QPainterPath, QPainter
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QObject

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SVG_STROKE_COLOR  = QColor("#1a1a2e")
SVG_BORDER_NORMAL = QColor("#44aa88")
SVG_BORDER_SELECT = QColor("#22ddaa")
DEFAULT_FLATTEN_STEP = 0.5  # mm — resolution for Bézier linearisation


class SvgBlockSignals(QObject):
    """Qt signals for SvgBlock (QGraphicsItem can't be a QObject)."""
    delete_requested = pyqtSignal(object)


def parse_svg_to_polylines(svg_path: str, step: float = DEFAULT_FLATTEN_STEP):
    """
    Parse an SVG file and return:
        strokes : list[list[tuple[float, float]]]
        raw_bbox: (min_x, min_y, max_x, max_y)

    All coordinates are in millimetres (SVG user-units treated as mm).
    Bézier curves are linearised to segments of ≤ *step* mm.
    """
    from svgelements import SVG, Path, Shape, Circle, Ellipse, Rect, Line, Polyline, Polygon

    svg = SVG.parse(svg_path)
    strokes = []

    for element in svg.elements():
        # Skip non-drawable elements
        if not isinstance(element, (Path, Shape)):
            continue
        # Convert every shape to a Path first, then linearise
        try:
            if isinstance(element, Path):
                path = element
            else:
                path = Path(element)
        except Exception:
            continue

        # Ensure the path has segments
        try:
            d_str = path.d()
            if not d_str or not d_str.strip():
                continue
        except Exception:
            continue

        pts = []
        try:
            for seg in path.segments():
                # .point() returns complex: (real=x, imag=y)
                try:
                    length = seg.length()
                except Exception:
                    length = 0

                if length < 0.001:
                    p = seg.point(0.0)
                    pts.append((p.real, p.imag))
                    continue

                n_steps = max(2, int(math.ceil(length / step)))
                for i in range(n_steps + 1):
                    t = i / n_steps
                    p = seg.point(t)
                    pts.append((p.real, p.imag))
        except Exception:
            continue

        if len(pts) >= 2:
            # Deduplicate consecutive identical points
            cleaned = [pts[0]]
            for pt in pts[1:]:
                if abs(pt[0] - cleaned[-1][0]) > 0.001 or abs(pt[1] - cleaned[-1][1]) > 0.001:
                    cleaned.append(pt)
            if len(cleaned) >= 2:
                strokes.append(cleaned)

    # Compute bounding box
    if strokes:
        all_x = [x for s in strokes for x, y in s]
        all_y = [y for s in strokes for x, y in s]
        raw_bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
    else:
        raw_bbox = (0, 0, 1, 1)

    return strokes, raw_bbox


class SvgBlock(QGraphicsItem):
    """
    A draggable, resizable vector-graphics block for the Plotter Studio canvas.

    Key features:
      • Parses SVG into polyline strokes at construction time
      • Stores the *original* strokes and scales them on the fly during resize
      • Exports strokes in scene coordinates via get_strokes_in_scene()
      • Supports 4-corner + 4-edge resize handles with aspect-ratio lock
    """
    HANDLE_SIZE = 3.0  # radius of resize handles (mm)

    def __init__(self, svg_path: str, target_width: float = 60.0):
        super().__init__()
        self.signals = SvgBlockSignals()

        self.svg_path = svg_path
        self.svg_name = os.path.basename(svg_path)

        # Parse SVG
        original_strokes, raw_bbox = parse_svg_to_polylines(svg_path)
        self._original_strokes = original_strokes
        self._raw_bbox = raw_bbox

        # Compute original dimensions
        raw_w = max(raw_bbox[2] - raw_bbox[0], 0.01)
        raw_h = max(raw_bbox[3] - raw_bbox[1], 0.01)
        self._aspect_ratio = raw_h / raw_w

        # Target display size (mm on canvas)
        self._display_width = target_width
        self._display_height = target_width * self._aspect_ratio

        # Build scaled strokes
        self._strokes: list = []
        self._bbox: tuple = (0, 0, 10, 10)
        self._rescale_strokes()

        # Interaction state
        self._resizing = False
        self._resize_corner = None  # which corner/edge is being dragged
        self._drag_start_pos = QPointF()
        self._drag_start_size = (0.0, 0.0)
        self._lock_aspect = True

        # Flags
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    # ------------------------------------------------------------------
    # Rescale original strokes to current display dimensions
    # ------------------------------------------------------------------
    def _rescale_strokes(self):
        """Rebuild self._strokes by scaling original strokes to display size."""
        raw_x0, raw_y0, raw_x1, raw_y1 = self._raw_bbox
        raw_w = max(raw_x1 - raw_x0, 0.01)
        raw_h = max(raw_y1 - raw_y0, 0.01)

        sx = self._display_width / raw_w
        sy = self._display_height / raw_h

        self._strokes = []
        for stroke in self._original_strokes:
            self._strokes.append([
                ((x - raw_x0) * sx, (y - raw_y0) * sy)
                for x, y in stroke
            ])

        self._bbox = (0, 0, self._display_width, self._display_height)
        self._rebuild_path()

    def _rebuild_path(self):
        """Build QPainterPath child items from current strokes."""
        for child in list(self.childItems()):
            if isinstance(child, QGraphicsPathItem):
                child.setParentItem(None)
                if self.scene():
                    self.scene().removeItem(child)

        if not self._strokes:
            self.update()
            return

        path = QPainterPath()
        for stroke in self._strokes:
            if len(stroke) < 2:
                continue
            path.moveTo(stroke[0][0], stroke[0][1])
            for px, py in stroke[1:]:
                path.lineTo(px, py)

        path_item = QGraphicsPathItem(path, self)
        path_item.setPen(QPen(
            SVG_STROKE_COLOR, 0.4,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        ))

        self.prepareGeometryChange()
        self.update()

    # ------------------------------------------------------------------
    # QGraphicsItem interface
    # ------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        pad = 4.0
        return QRectF(-pad, -pad,
                      self._display_width + pad * 2,
                      self._display_height + pad * 2)

    def paint(self, painter: QPainter, option, widget=None):
        """Draw selection border and resize handles."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self._display_width, self._display_height)

        is_selected = self.isSelected()
        if is_selected:
            pen = QPen(SVG_BORDER_SELECT, 0.6, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(34, 221, 170, 18)))
        else:
            pen = QPen(SVG_BORDER_NORMAL, 0.3, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.drawRect(rect)

        # Draw resize handles when selected
        if is_selected:
            dot_pen = QPen(SVG_BORDER_SELECT, 0.5)
            painter.setPen(dot_pen)
            painter.setBrush(QBrush(SVG_BORDER_SELECT))
            r = 1.5
            w, h = self._display_width, self._display_height
            for cx, cy in [
                (0, 0), (w, 0), (0, h), (w, h),        # corners
                (w / 2, 0), (w / 2, h),                  # top/bottom mid
                (0, h / 2), (w, h / 2),                  # left/right mid
            ]:
                painter.drawEllipse(QPointF(cx, cy), r, r)

        # Label
        if is_selected:
            painter.setPen(QPen(QColor("#88ccaa"), 0.0))
            font = painter.font()
            font.setPointSizeF(max(2.5, self._display_width * 0.04))
            painter.setFont(font)
            name_rect = QRectF(0, self._display_height + 1, self._display_width, 6)
            painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft, f"📐 {self.svg_name}")

    # ------------------------------------------------------------------
    # Hover / resize detection
    # ------------------------------------------------------------------
    def _get_handle_at(self, pos) -> str | None:
        """Return the handle name at the given local position, or None."""
        w, h = self._display_width, self._display_height
        hr = self.HANDLE_SIZE

        handles = {
            'tl': (0, 0), 'tr': (w, 0), 'bl': (0, h), 'br': (w, h),
            'tc': (w/2, 0), 'bc': (w/2, h),
            'ml': (0, h/2), 'mr': (w, h/2),
        }

        for name, (hx, hy) in handles.items():
            if abs(pos.x() - hx) < hr and abs(pos.y() - hy) < hr:
                return name
        return None

    def hoverMoveEvent(self, event):
        handle = self._get_handle_at(event.pos())
        if handle in ('tl', 'br'):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in ('tr', 'bl'):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif handle in ('tc', 'bc'):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif handle in ('ml', 'mr'):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverLeaveEvent(event)

    # ------------------------------------------------------------------
    # Mouse events for resizing
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        self._drag_start_pos = event.scenePos()
        self._drag_start_block_pos = self.scenePos()
        handle = self._get_handle_at(event.pos())
        if handle and self.isSelected():
            self._resizing = True
            self._resize_corner = handle
            self._drag_start_size = (self._display_width, self._display_height)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            dx = event.scenePos().x() - self._drag_start_pos.x()
            dy = event.scenePos().y() - self._drag_start_pos.y()
            old_w, old_h = self._drag_start_size
            corner = self._resize_corner

            new_w, new_h = old_w, old_h

            if corner in ('br', 'mr', 'tr'):
                new_w = max(10.0, old_w + dx)
            elif corner in ('bl', 'ml', 'tl'):
                new_w = max(10.0, old_w - dx)

            if corner in ('br', 'bc', 'bl'):
                new_h = max(10.0, old_h + dy)
            elif corner in ('tr', 'tc', 'tl'):
                new_h = max(10.0, old_h - dy)

            # Aspect-ratio lock for corner drags
            if self._lock_aspect and corner in ('tl', 'tr', 'bl', 'br'):
                # Use whichever dimension changed more
                scale_w = new_w / old_w
                scale_h = new_h / old_h
                scale = (scale_w + scale_h) / 2
                new_w = max(10.0, old_w * scale)
                new_h = max(10.0, old_h * scale)

            self.prepareGeometryChange()
            self._display_width = new_w
            self._display_height = new_h

            # Reposition for top-left / left / top drags
            if corner in ('tl', 'ml', 'bl'):
                move_dx = old_w - new_w
                self.setX(self._drag_start_block_pos.x() + move_dx)
            if corner in ('tl', 'tc', 'tr'):
                move_dy = old_h - new_h
                self.setY(self._drag_start_block_pos.y() + move_dy)

            self._rescale_strokes()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self._resize_corner = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Context menu (right-click)
    # ------------------------------------------------------------------
    def contextMenuEvent(self, event):
        from PyQt6.QtWidgets import QMenu
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu { background-color: #2d2d30; color: #e0e0e0; border: 1px solid #555; }
            QMenu::item:selected { background-color: #094771; }
        """)

        toggle_lock = menu.addAction(
            "🔒 Unlock Aspect Ratio" if self._lock_aspect else "🔓 Lock Aspect Ratio"
        )
        toggle_lock.triggered.connect(self._toggle_aspect_lock)

        menu.addSeparator()

        del_act = menu.addAction("🗑️ Delete SVG Block")
        del_act.triggered.connect(lambda: self.signals.delete_requested.emit(self))

        menu.exec(event.screenPos())

    def _toggle_aspect_lock(self):
        self._lock_aspect = not self._lock_aspect

    # ------------------------------------------------------------------
    # G-Code export interface (same as HandwritingBlock)
    # ------------------------------------------------------------------
    def get_strokes_in_scene(self) -> list:
        """Return strokes translated to scene coordinates."""
        origin = self.scenePos()
        return [
            [(px + origin.x(), py + origin.y()) for px, py in s]
            for s in self._strokes
        ]

    # ------------------------------------------------------------------
    # Serialisation helpers for session save/load
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        pos = self.scenePos()
        return {
            "type":     "svg",
            "svg_path": self.svg_path,
            "x":        pos.x(),
            "y":        pos.y(),
            "width":    self._display_width,
            "height":   self._display_height,
            "lock_aspect": self._lock_aspect,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SvgBlock':
        blk = cls(d["svg_path"], target_width=d.get("width", 60.0))
        blk._display_height = d.get("height", blk._display_height)
        blk._lock_aspect = d.get("lock_aspect", True)
        blk._rescale_strokes()
        return blk
