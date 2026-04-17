"""
╔══════════════════════════════════════════════════════════════════════╗
║                    PLOTTER STUDIO v2.0                              ║
║                                                                      ║
║  Canvas Layout Editor — Canva-style draggable handwriting blocks    ║
║  Each text block is independently synthesized, movable & regeneable ║
║                                                                      ║
║  IMPORTANT: PyTorch must be imported BEFORE PyQt6 on Windows to     ║
║  prevent C10.dll initialization conflicts (WinError 1114).          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# === CRITICAL IMPORT ORDER: PyTorch FIRST, then PyQt6 ===
import torch
from handwriting_synthesis.sampling import HandwritingSynthesizer as CoreSynthesizer
from handwriting_synthesis.stroke_reviewer import StrokeReviewer

import sys
import os
import math

import json
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtCore import QPointF
import xml.etree.ElementTree as ET

import re
import numpy as np

from PyQt6.QtGui import QAction, QUndoStack, QUndoCommand
from PyQt6.QtWidgets import (QMenuBar, QComboBox, QMenu, 
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGraphicsScene, QGraphicsRectItem,
    QGraphicsPathItem, QSlider, QSplitter, QGraphicsView, QMessageBox,
    QLineEdit, QFormLayout, QGroupBox, QFileDialog, QSizePolicy,
    QCheckBox, QGraphicsItem, QDialog, QPlainTextEdit, QDialogButtonBox
)
from PyQt6.QtGui import (
    QDoubleValidator, QIntValidator, QPen, QColor, QFont, QPainter,
    QBrush, QPainterPath, QWheelEvent, QMouseEvent, QAction
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QObject


# ==============================================================================
# 1. CONSTANTS
# ==============================================================================
A4_WIDTH_MM  = 210.0
A4_HEIGHT_MM = 297.0
DEFAULT_MODEL_PATH = "checkpoints/Epoch_46"

BLOCK_BORDER_NORMAL   = QColor("#aaaacc")
BLOCK_BORDER_SELECTED = QColor("#4488ff")
BLOCK_BORDER_WIDTH    = 0.4

# Punctuation to strip from words before neural synthesis
PUNCT_CHARS = set('.,;:!?"\'()[]{}\u201c\u201d\u2018\u2019\u2013\u2014')

# Color coding for regenerated chunks (indexed by retries_used, last entry = 3+)
# Each entry: (pen_color_hex, legend_text)
RETRY_COLORS = [
    ("#1a1a2e", "clean"),           # 0 retries  — near-black (normal)
    ("#b8860b", "1 retry"),         # 1 retry    — dark goldenrod / amber
    ("#c05000", "2 retries"),       # 2 retries  — burnt orange
    ("#a01020", "3+ retries"),      # 3+ retries — crimson red
]


# ==============================================================================
# 2. SYNTHESIS ENGINE
# ==============================================================================
class SynthesisEngine:
    """
    Manages the PyTorch handwriting model lifecycle.
    Caches the loaded model and only reloads when bias changes.
    """
    def __init__(self, model_path=DEFAULT_MODEL_PATH, bias=1.0):
        self.model_path = model_path
        self.bias = bias
        self.device = torch.device("cpu")
        self._synthesizer = None

    def _ensure_loaded(self):
        if self._synthesizer is None:
            print(f"[Engine] Loading model from '{self.model_path}' (bias={self.bias:.2f})...")
            self._synthesizer = CoreSynthesizer.load(self.model_path, self.device, self.bias)

    def set_bias(self, new_bias: float):
        # Neural network MDN equations collapse if bias goes > ~2.5. Clamp to mathematical bounds.
        new_bias = min(max(new_bias, 0.0), 2.0)
        if abs(new_bias - self.bias) > 0.001:
            self.bias = new_bias
            self._synthesizer = None

    def synthesize(self, text: str, scale: float = 0.02,
                   line_spacing: float = 15.0,
                   max_width: float = 170.0,
                   stochastic: bool = True) -> tuple:
        """LEGACY: Replaced by word-by-word synthesis in the layout engine."""
        return self.synthesize_word(text, scale, stochastic)

    def synthesize_word(self, text: str, scale: float = 0.02,
                       stochastic: bool = True) -> tuple:
        """
        Synthesize a single word (or small chunk) and normalize it to a horizontal baseline.
        Returns (strokes, bbox) where strokes are local to (0,0).
        """
        self._ensure_loaded()
        if not text.strip():
            return [], (0, 0, 0, 0)

        # Encode with a trailing space for more natural ending
        c = self._synthesizer._encode_text(text + " ")
        estimated_steps = max(300, len(text) * 65 + 100)

        with torch.no_grad():
            seq = self._synthesizer.model.sample_means(
                context=c, steps=estimated_steps, stochastic=stochastic
            )
        seq = seq.cpu()

        seq = self._synthesizer._undo_normalization(seq)
        output = seq.numpy()

        cursor_x, cursor_y = 0.0, 0.0
        strokes = []
        current_stroke = []
        previous_eos = 1.0

        for row in output:
            dx, dy, eos = float(row[0]), float(row[1]), float(row[2])
            cursor_x += dx * scale
            cursor_y += dy * scale
            
            is_jump = (previous_eos == 1.0)
            if is_jump:
                if current_stroke: strokes.append(current_stroke)
                current_stroke = [(cursor_x, cursor_y)]
            else:
                current_stroke.append((cursor_x, cursor_y))

            if eos == 1.0:
                if len(current_stroke) >= 2: strokes.append(current_stroke)
                current_stroke = []
            
            previous_eos = eos

        if current_stroke: strokes.append(current_stroke)

        # 2. Straighten internal slant (if long enough)
        if len(text) > 4:
            strokes = self._straighten_baseline(strokes)

        # NEW: Find true baseline and normalize so baseline is at y=0
        baseline = self._find_baseline(strokes)
        
        normalized_strokes = []
        for s in strokes:
            # Apply Douglas-Peucker simplification post-generation
            simplified = SynthesisEngine._dp_simplify_stroke(s, 0.15)
            normalized_strokes.append([(x, y - baseline) for x, y in simplified])
            
        # Pen-lift elision
        merged_strokes = []
        if normalized_strokes:
            current_merged = normalized_strokes[0]
            for s in normalized_strokes[1:]:
                if not s: continue
                dist = math.hypot(s[0][0] - current_merged[-1][0], s[0][1] - current_merged[-1][1])
                if dist < 0.3:
                    current_merged.extend(s)
                else:
                    merged_strokes.append(current_merged)
                    current_merged = s
            if current_merged:
                merged_strokes.append(current_merged)
            strokes = merged_strokes
        else:
            strokes = []

        if strokes:
            all_x = [x for s in strokes for x, y in s]
            all_y = [y for s in strokes for x, y in s]
            bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
        else:
            bbox = (0, 0, 0, 0)

        return strokes, bbox

    @staticmethod
    def _find_baseline(strokes: list) -> float:
        """Find the baseline of a word by analyzing local Y maxima (bottoms of letters)."""
        if not strokes:
            return 0.0

        bottom_points = []
        for stroke in strokes:
            if len(stroke) < 2:
                if stroke:
                    bottom_points.append(stroke[0][1])
                continue
            
            # endpoints
            bottom_points.append(stroke[0][1])
            bottom_points.append(stroke[-1][1])
            
            # local Y maxima (since Y points down, these correspond to bottoms of letters)
            for i in range(1, len(stroke) - 1):
                y_prev = stroke[i-1][1]
                y_curr = stroke[i][1]
                y_next = stroke[i+1][1]
                if y_curr > y_prev and y_curr > y_next:
                    bottom_points.append(y_curr)

        if not bottom_points:
            all_y = [y for s in strokes for x, y in s]
            return max(all_y) if all_y else 0.0

        # Use the 65th-percentile instead of median (50th).
        # This sits just above the true ink baseline, safely below descender
        # outliers while still capturing where the majority of letter-bottom
        # points cluster.  More stable across different stroke densities.
        bottom_points.sort()
        idx65 = int(len(bottom_points) * 0.65)
        idx65 = min(idx65, len(bottom_points) - 1)
        return bottom_points[idx65]

    @staticmethod
    def _dp_simplify_stroke(stroke: list, epsilon: float = 0.15) -> list:
        """Lightweight Douglas-Peucker simplification to remove redundant collinear points."""
        if len(stroke) < 3:
            return stroke
        
        start, end = stroke[0], stroke[-1]
        max_dist = 0.0
        max_idx = 0
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        line_len_sq = dx*dx + dy*dy
        
        if line_len_sq == 0.0:
            for i in range(1, len(stroke) - 1):
                pt = stroke[i]
                dist = math.hypot(pt[0] - start[0], pt[1] - start[1])
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
        else:
            for i in range(1, len(stroke) - 1):
                pt = stroke[i]
                t = max(0.0, min(1.0, ((pt[0] - start[0]) * dx + (pt[1] - start[1]) * dy) / line_len_sq))
                proj_x = start[0] + t * dx
                proj_y = start[1] + t * dy
                dist = math.hypot(pt[0] - proj_x, pt[1] - proj_y)
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
                    
        if max_dist > epsilon:
            left = SynthesisEngine._dp_simplify_stroke(stroke[:max_idx+1], epsilon)
            right = SynthesisEngine._dp_simplify_stroke(stroke[max_idx:], epsilon)
            return left[:-1] + right
        else:
            return [start, end]

    @staticmethod
    def _straighten_baseline(strokes: list) -> list:
        """
        Remove baseline slope from strokes using linear regression.
        Groups all points into horizontal "lines" based on Y-proximity,
        computes the slope of each line, and subtracts it.
        """
        if not strokes:
            return strokes

        # Collect all points with their stroke/point indices
        all_pts = []
        for si, stroke in enumerate(strokes):
            for pi, (x, y) in enumerate(stroke):
                all_pts.append((x, y, si, pi))

        if len(all_pts) < 3:
            return strokes

        # Sort by Y to cluster into horizontal lines
        all_pts.sort(key=lambda p: p[1])

        # Cluster points into lines based on Y-proximity
        lines = []
        current_line = [all_pts[0]]
        # Use median Y-range of all points to set the line threshold
        y_vals = [p[1] for p in all_pts]
        y_range = max(y_vals) - min(y_vals)
        num_estimated_lines = max(1, round(y_range / 8.0))  # ~8mm per line
        threshold = max(3.0, y_range / max(1, num_estimated_lines * 2))

        for pt in all_pts[1:]:
            if abs(pt[1] - current_line[-1][1]) < threshold:
                current_line.append(pt)
            else:
                lines.append(current_line)
                current_line = [pt]
        lines.append(current_line)

        # For each line, compute slope and apply correction
        corrections = {}  # (stroke_idx, point_idx) -> dy_correction
        for line_pts in lines:
            if len(line_pts) < 4:
                continue
            xs = np.array([p[0] for p in line_pts])
            ys = np.array([p[1] for p in line_pts])
            if (xs.max() - xs.min()) < 1.0:  # Skip near-vertical clusters
                continue
            # Linear regression: y = slope * x + intercept
            slope, intercept = np.polyfit(xs, ys, 1)
            # Only correct if there's meaningful slope
            if abs(slope) < 0.001:
                continue
            mean_x = xs.mean()
            for pt in line_pts:
                correction = slope * (pt[0] - mean_x)
                corrections[(pt[2], pt[3])] = correction

        # Apply corrections
        new_strokes = []
        for si, stroke in enumerate(strokes):
            new_stroke = []
            for pi, (x, y) in enumerate(stroke):
                dy = corrections.get((si, pi), 0.0)
                new_stroke.append((x, y - dy))
            new_strokes.append(new_stroke)

        return new_strokes

    @staticmethod
    def make_punct_strokes(char: str, word_height: float) -> tuple:
        """Generate handwriting-like strokes for a punctuation character.
        Returns (strokes, width) where strokes are positioned at x=0."""
        h = max(word_height, 1.0)
        r = h * 0.04  # dot radius proportional to word height

        if char == '.':
            cx = r
            cy = -r
            return (
                [[(cx - r, cy), (cx, cy - r), (cx + r, cy),
                  (cx, cy + r), (cx - r, cy)]],
                r * 3,
            )
        elif char == ',':
            cx = r
            cy = -r
            return ([[(cx, cy), (cx - r * 0.6, cy + h * 0.14)]], r * 3)
        elif char == ';':
            cx = r
            return (
                [[(cx - r * 0.4, -h * 0.4), (cx + r * 0.4, -h * 0.4)],
                 [(cx, -r), (cx - r * 0.6, h * 0.14)]],
                r * 3,
            )
        elif char == ':':
            cx = r
            return (
                [[(cx - r * 0.4, -h * 0.4), (cx + r * 0.4, -h * 0.4)],
                 [(cx - r * 0.4, -r), (cx + r * 0.4, -r)]],
                r * 3,
            )
        elif char == '!':
            cx = r
            return (
                [[(cx, -h * 0.7), (cx, -h * 0.2)],
                 [(cx - r * 0.3, -r), (cx + r * 0.3, -r)]],
                r * 3,
            )
        elif char == '?':
            cx = r * 2
            return (
                [[(cx - h * 0.06, -h * 0.6), (cx, -h * 0.7),
                  (cx + h * 0.06, -h * 0.6), (cx + h * 0.06, -h * 0.4),
                  (cx, -h * 0.25), (cx, -h * 0.15)],
                 [(cx - r * 0.3, -r), (cx + r * 0.3, -r)]],
                h * 0.15,
            )
        elif char in ('"', '\u201c'):
            return (
                [[(r, -h * 0.8), (r * 2, -h * 0.6)],
                 [(r * 3.5, -h * 0.8), (r * 4.5, -h * 0.6)]],
                r * 6,
            )
        elif char in ('\u201d',):
            return (
                [[(r, -h * 0.6), (r * 2, -h * 0.8)],
                 [(r * 3.5, -h * 0.6), (r * 4.5, -h * 0.8)]],
                r * 6,
            )
        elif char in ("'", '\u2018'):
            return ([[(r, -h * 0.8), (r * 2, -h * 0.6)]], r * 3)
        elif char in ('\u2019',):
            return ([[(r, -h * 0.6), (r * 2, -h * 0.8)]], r * 3)
        elif char == '(':
            return (
                [[(h * 0.08, -h * 0.75), (0, -h * 0.4), (h * 0.08, h * 0.05)]],
                h * 0.1,
            )
        elif char == ')':
            return (
                [[(0, -h * 0.75), (h * 0.08, -h * 0.4), (0, h * 0.05)]],
                h * 0.1,
            )
        elif char in ('-', '\u2013', '\u2014'):
            w = h * 0.12 if char == '-' else h * 0.22
            return ([[(0, -h * 0.4), (w, -h * 0.4)]], w + r)
        else:
            return ([[(r, -h * 0.4), (r * 2, -h * 0.4)]], r * 3)

    def compile_gcode(self, strokes: list,
                      pen_up_cmd: str = "M3 S0", pen_down_cmd: str = "M3 S90",
                      draw_speed: int = 1500, rapid_speed: int = 3000,
                      flip_axes: bool = False,
                      origin_x: float = 0.0,
                      origin_y: float = A4_HEIGHT_MM) -> list:
        """
        Convert strokes into Grbl-compatible G-code.
        Strokes are in scene mm with Y increasing downward.
        G-code uses custom origin, Y increasing upward.
        """
        gcode = [
            "G90 ; Absolute Positioning",
            "G21 ; Millimeters",
            f"{pen_up_cmd} ; Initialize Pen Up",
        ]
        pen_is_down = False

        for stroke in strokes:
            if len(stroke) < 2:
                continue
            if pen_is_down:
                gcode.append(f"{pen_up_cmd} ; Pen Up")
                pen_is_down = False

            sx, sy = stroke[0]
            cnc_x = sx - origin_x
            cnc_y = origin_y - sy
            if flip_axes:
                cnc_x, cnc_y = cnc_y, cnc_x
            gcode.append(f"G0 X{cnc_x:.3f} Y{cnc_y:.3f} F{rapid_speed}")
            gcode.append(f"{pen_down_cmd} ; Pen Down")
            pen_is_down = True

            for px, py in stroke[1:]:
                cnc_px = px - origin_x
                cnc_py = origin_y - py
                if flip_axes:
                    cnc_px, cnc_py = cnc_py, cnc_px
                gcode.append(f"G1 X{cnc_px:.3f} Y{cnc_py:.3f} F{draw_speed}")

        if pen_is_down:
            gcode.append(f"{pen_up_cmd} ; Pen Up")
        gcode.append(f"G0 X0 Y0 F{rapid_speed} ; Return Home")
        return gcode


class OriginCrosshair(QGraphicsItem):
    def boundingRect(self):
        return QRectF(-10, -10, 20, 20)
    def paint(self, painter, option, widget):
        pen = QPen(QColor("#f05050"), 1.0)
        painter.setPen(pen)
        painter.drawLine(-8, 0, 8, 0)
        painter.drawLine(0, -8, 0, 8)
        painter.drawEllipse(QPointF(0,0), 3, 3)

class A4PreviewView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setStyleSheet("background-color: #1e1e1e; border: none;")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.show_rulers = True
        self.show_grid = False
        self._panning = False
        self._pan_start = QPointF()

    def set_rulers(self, visible):
        self.show_rulers = visible
        self.viewport().update()

    def set_grid(self, visible):
        self.show_grid = visible
        self.viewport().update()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if hasattr(self, 'show_grid') and self.show_grid:
            self._draw_grid(painter, rect)
        if hasattr(self, 'show_rulers') and self.show_rulers:
            self._draw_rulers(painter, rect)

    def _draw_grid(self, painter, rect):
        painter.save()
        pen = QPen(QColor(150, 150, 150, 40), 1, Qt.PenStyle.DotLine)
        painter.setPen(pen)
        left = int(math.floor(rect.left() / 5.0)) * 5
        right = int(math.ceil(rect.right()))
        top = int(math.floor(rect.top() / 5.0)) * 5
        bottom = int(math.ceil(rect.bottom()))
        for x in range(left, right, 5):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(top, bottom, 5):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)
        painter.restore()

    def _draw_rulers(self, painter, rect):
        painter.save()
        view_rect = self.viewport().rect()
        scene_top_left = self.mapToScene(view_rect.topLeft())
        scene_bottom_right = self.mapToScene(view_rect.bottomRight())
        tf = self.transform()
        scale_x = tf.m11()
        if scale_x < 0.001: scale_x = 1.0
        ruler_thickness = 20 / scale_x
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(30, 30, 30, 240))
        painter.drawRect(QRectF(scene_top_left.x(), scene_top_left.y(), scene_bottom_right.x() - scene_top_left.x(), ruler_thickness))
        painter.drawRect(QRectF(scene_top_left.x(), scene_top_left.y(), ruler_thickness, scene_bottom_right.y() - scene_top_left.y()))
        painter.setPen(QPen(QColor(200, 200, 200), 1.0 / scale_x))
        font = QFont("Arial", max(int(8/scale_x), 1))
        painter.setFont(font)
        step = 10 if scale_x > 1.5 else (50 if scale_x > 0.5 else 100)
        start_x = int(scene_top_left.x() / step) * step
        for x in range(start_x, int(scene_bottom_right.x()), step):
            if x < scene_top_left.x() + ruler_thickness: continue
            painter.drawLine(QPointF(x, scene_top_left.y() + ruler_thickness - 5/scale_x), QPointF(x, scene_top_left.y() + ruler_thickness))
            txt_rect = QRectF(x + 2/scale_x, scene_top_left.y(), 40/scale_x, ruler_thickness - 2/scale_x)
            painter.drawText(txt_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, str(x))
        start_y = int(scene_top_left.y() / step) * step
        for y in range(start_y, int(scene_bottom_right.y()), step):
            if y < scene_top_left.y() + ruler_thickness: continue
            painter.drawLine(QPointF(scene_top_left.x() + ruler_thickness - 5/scale_x, y), QPointF(scene_top_left.x() + ruler_thickness, y))
            painter.save()
            painter.translate(scene_top_left.x() + 4/scale_x, y - 2/scale_x)
            painter.rotate(-90)
            painter.drawText(QRectF(0, 0, 40/scale_x, ruler_thickness), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, str(y))
            painter.restore()
        painter.restore()

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            zoom_in = event.angleDelta().y() > 0
            if zoom_in:
                self.scale(1.15, 1.15)
            else:
                self.scale(1.0 / 1.15, 1.0 / 1.15)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

# ==============================================================================
# 3. ADD BLOCK DIALOG
# ==============================================================================
class AddBlockDialog(QDialog):
    """Simple modal dialog for entering text for a new handwriting block."""
    def __init__(self, parent=None, initial_text=""):
        super().__init__(parent)
        self.setWindowTitle("Add / Edit Text Block")
        self.setMinimumWidth(420)
        self.setStyleSheet("""
            QDialog { background-color: #252526; }
            QLabel { color: #cccccc; font-size: 12px; }
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: 'Segoe UI', Arial;
                font-size: 14px;
                padding: 6px;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 20px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1177bb; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        layout.addWidget(QLabel("Enter text for this handwriting block:"))

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlainText(initial_text)
        self.text_edit.setMinimumHeight(120)
        layout.addWidget(self.text_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_text(self) -> str:
        return self.text_edit.toPlainText().strip()


# ==============================================================================
# 4. HANDWRITING BLOCK — draggable canvas item
# ==============================================================================
from PyQt6.QtCore import QThread


class SynthesisWorker(QThread):
    progress = pyqtSignal(int, str)
    chunk_ready = pyqtSignal(str)
    log_msg = pyqtSignal(str)
    finished = pyqtSignal(object, list)
    error = pyqtSignal(str)

    def __init__(self, block, engine, tokens, scale_param, stochastic, ai_review, review_threshold, review_retries):
        super().__init__()
        self.block = block
        self.engine = engine
        self.tokens = tokens
        self.scale_param = scale_param
        self.stochastic = stochastic
        self.ai_review = ai_review
        self.review_threshold = review_threshold
        self.review_retries = review_retries

    def run(self):
        try:
            stroke_data = []
            for i, token in enumerate(self.tokens):
                if i % 5 == 0:
                    self.chunk_ready.emit(f"Synthesizing: {i}/{len(self.tokens)} words...")
                
                if token == '\n':
                    stroke_data.append(('\n', [], (0, 0, 0, 0), 0))
                    continue

                leading, core, trailing = HandwritingBlock._split_punctuation(token)

                if not core:
                    all_p = []
                    cx = 0.0
                    fallback_h = 5.0
                    for ch in token:
                        ps, pw = self.engine.make_punct_strokes(ch, fallback_h)
                        for s in ps:
                            all_p.append([(x + cx, y) for x, y in s])
                        cx += pw
                    if all_p:
                        ax = [x for s in all_p for x, y in s]
                        ay = [y for s in all_p for x, y in s]
                        stroke_data.append((token, all_p, (min(ax), min(ay), max(ax), max(ay)), 0))
                    else:
                        stroke_data.append((token, [], (0, 0, 0, 0), 0))
                    continue

                strokes, bbox = self.engine.synthesize_word(
                    text=core,
                    scale=self.scale_param,
                    stochastic=self.stochastic,
                )

                if not strokes:
                    stroke_data.append((token, strokes, bbox, 0))
                    continue

                word_h = bbox[3] - bbox[1]

                if leading:
                    total_shift = 0.0
                    punct_parts = []
                    for ch in leading:
                        ps, pw = self.engine.make_punct_strokes(ch, word_h)
                        for s in ps:
                            punct_parts.append([(x + total_shift, y) for x, y in s])
                        total_shift += pw
                    strokes = [[(x + total_shift, y) for x, y in s] for s in strokes]
                    strokes = punct_parts + strokes

                if trailing:
                    ax = [x for s in strokes for x, y in s]
                    right_edge = max(ax) if ax else 0
                    gap = word_h * 0.02
                    cx = right_edge + gap
                    for ch in trailing:
                        ps, pw = self.engine.make_punct_strokes(ch, word_h)
                        for s in ps:
                            strokes.append([(x + cx, y) for x, y in s])
                        cx += pw

                ax = [x for s in strokes for x, y in s]
                ay = [y for s in strokes for x, y in s]
                if ax and ay:
                    bbox = (min(ax), min(ay), max(ax), max(ay))

                stroke_data.append((token, strokes, bbox, 0))

            if self.ai_review and stroke_data:
                reviewer = StrokeReviewer(
                    threshold=self.review_threshold,
                    max_retries=self.review_retries,
                )
                stroke_data = reviewer.review_and_improve(
                    engine=self.engine,
                    stroke_data=stroke_data,
                    scale=self.scale_param,
                    stochastic=self.stochastic,
                    status_cb=self.chunk_ready.emit,
                    log_cb=self.log_msg.emit,
                )

            self.chunk_ready.emit(f"✓ Block rendered: {len(self.tokens)} tokens.")
            self.finished.emit(self.block, stroke_data)
        except Exception as e:
            self.error.emit(str(e))



class AddBlockCommand(QUndoCommand):
    def __init__(self, scene, block, blocks_list, description="Add Block"):
        super().__init__(description)
        self.scene = scene
        self.block = block
        self.blocks = blocks_list

    def redo(self):
        if self.block not in self.blocks:
            self.scene.addItem(self.block)
            self.blocks.append(self.block)

    def undo(self):
        if self.block in self.blocks:
            self.scene.removeItem(self.block)
            self.blocks.remove(self.block)

class MoveBlockCommand(QUndoCommand):
    def __init__(self, block, old_pos, new_pos, description="Move Block"):
        super().__init__(description)
        self.block = block
        self.old_pos = old_pos
        self.new_pos = new_pos

    def redo(self):
        self.block.setPos(self.new_pos)

    def undo(self):
        self.block.setPos(self.old_pos)


class DeleteBlockCommand(QUndoCommand):
    def __init__(self, scene, block, blocks_list):
        super().__init__("Delete Block")
        self.scene = scene
        self.block = block
        self.blocks = blocks_list

    def redo(self):
        if self.block in self.blocks:
            self.scene.removeItem(self.block)
            self.blocks.remove(self.block)

    def undo(self):
        if self.block not in self.blocks:
            self.scene.addItem(self.block)
            self.blocks.append(self.block)

class BlockSignals(QObject):
    """Qt signals for HandwritingBlock (QGraphicsItem can't be a QObject directly)."""
    regenerate_requested = pyqtSignal(object)
    delete_requested = pyqtSignal(object)
    edit_requested = pyqtSignal(object)


class HandwritingBlock(QGraphicsItem):
    """
    A self-contained, draggable block of synthesized handwriting on the canvas.
    Stores its own source text and synthesis params so it can be regenerated.
    """
    def __init__(self, source_text: str, scale: float, line_spacing: float,
                 block_width: float = 170.0, word_spacing: float = 5.0,
                 stochastic: bool = True,
                 ai_review: bool = True,
                 review_threshold: float = 0.60,
                 review_retries: int = 3):
        super().__init__()
        self.signals = BlockSignals()

        self.source_text = source_text
        self.scale_param = scale
        self.line_spacing = line_spacing
        self.block_width = block_width
        self.word_spacing = word_spacing
        self.stochastic = stochastic

        # AI Review params
        self.ai_review = ai_review
        self.review_threshold = review_threshold
        self.review_retries = review_retries

        # Synthesized word data: list of (text, strokes, bbox)
        self._stroke_data: list = []
        
        # Final flattened strokes for rendering/G-code
        self._strokes: list = []
        self._bbox: tuple = (0, 0, 10, 5)
        self._path_item: QGraphicsPathItem | None = None

        # Interaction state
        self._selected = False

        # Flags
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setAcceptHoverEvents(True)
        self._resizing = False
        self._drag_start_x = 0
        self._drag_start_w = 0
        self.setZValue(10)

    def hoverMoveEvent(self, event):
        if event.pos().x() > self.block_width - 8:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverLeaveEvent(event)



    def mousePressEvent(self, event):
        self._drag_start_pos = self.scenePos()
        if event.pos().x() > self.block_width - 8:
            self._resizing = True
            self._drag_start_x = event.scenePos().x()
            self._drag_start_w = self.block_width
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            dx = event.scenePos().x() - self._drag_start_x
            self.block_width = max(30.0, self._drag_start_w + dx)
            self.re_layout()
            self.update()
            
            scene = self.scene()
            if scene:
                views = scene.views()
                if views and hasattr(views[0].parent(), 'slider_blk_width'):
                    views[0].parent().slider_blk_width.set_value(self.block_width)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            event.accept()
        else:
            old_pos = getattr(self, '_drag_start_pos', None)
            new_pos = self.scenePos()
            scene = self.scene()
            if scene and scene.views() and getattr(scene.views()[0], 'show_grid', False):
                snap_x = round(new_pos.x() / 5.0) * 5.0
                snap_y = round(new_pos.y() / 5.0) * 5.0
                new_pos = QPointF(snap_x, snap_y)
                self.setPos(new_pos)

            super().mouseReleaseEvent(event)
            
            if old_pos is not None and old_pos != new_pos:
                scene = self.scene()
                if scene and scene.views():
                    win = scene.views()[0].window()
                    if hasattr(win, 'undo_stack'):
                        win.undo_stack.push(MoveBlockCommand(self, old_pos, new_pos))


    def contextMenuEvent(self, event):
        menu = QMenu()
        act_edit = menu.addAction("Edit Text")
        act_regen = menu.addAction("Regenerate")
        act_prop = menu.addAction("Block Properties...")
        menu.addSeparator()
        act_del = menu.addAction("Delete")

        action = menu.exec(event.screenPos())
        if action == act_edit:
            self.signals.edit_requested.emit(self)
        elif action == act_regen:
            self.signals.regenerate_requested.emit(self)
        elif action == act_del:
            self.signals.delete_requested.emit(self)
        elif action == act_prop:
            dlg = BlockPropertiesDialog(self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                props = dlg.get_props()
                need_regen = (
                    abs(props['scale'] - self.scale_param) > 0.0001
                )
                self.scale_param = props['scale']
                self.block_width = props['width']
                self.line_spacing = props['line_spacing']
                self.word_spacing = props['word_spacing']
                if need_regen:
                    self.signals.regenerate_requested.emit(self)
                else:
                    self.re_layout()
                    self.update()
        event.accept()


    # ---- Static helpers -------------------------------------------------------
    @staticmethod
    def _tokenize(text: str, min_chunk_chars: int = 8) -> list:
        """
        Smart phrase-level tokenizer.

        Groups words into phrases until the running character count reaches
        min_chunk_chars, ensuring the RNN always receives enough context to
        generate well-formed cursive strokes.

        Leftover handling:
          - If the leftover phrase has >= min_chunk_chars/2 chars  it is
            emitted as its own chunk (avoids giant merged tokens).
          - Only truly tiny remainders (< min_chunk_chars/2) are merged
            backward into the preceding token.

        Newlines are preserved as '\\n' sentinels for line-break layout.
        Maximum chunk growth is capped at min_chunk_chars * 3 to prevent
        runaway merges on very short-word text.
        """
        MAX_CHUNK_MULTIPLIER = 3

        tokens = []
        for line in text.split('\n'):
            words = [w for w in line.split() if w.strip()]
            if not words:
                tokens.append('\n')
                continue

            chunk_words: list[str] = []
            chunk_chars: int = 0

            for word in words:
                # Count only the alphabetic core for length estimate
                core = word.strip('.,;:!?\'"()[]{}–—“”‘’')
                core_len = max(1, len(core))

                chunk_words.append(word)
                chunk_chars += core_len

                # Flush when chunk is long enough or hitting hard cap
                if (chunk_chars >= min_chunk_chars or
                        chunk_chars >= min_chunk_chars * MAX_CHUNK_MULTIPLIER):
                    tokens.append(' '.join(chunk_words))
                    chunk_words = []
                    chunk_chars = 0

            # Flush remaining words
            if chunk_words:
                leftover = ' '.join(chunk_words)
                leftover_chars = chunk_chars
                tiny_threshold = max(4, min_chunk_chars // 2)

                if leftover_chars < tiny_threshold:
                    # Small remainder: merge backward to avoid a tiny tail chunk
                    for idx in range(len(tokens) - 1, -1, -1):
                        if tokens[idx] != '\n':
                            # Only merge if resulting chunk won't exceed cap
                            merged_len = len(tokens[idx].replace(' ', '')) + leftover_chars
                            if merged_len <= min_chunk_chars * MAX_CHUNK_MULTIPLIER:
                                tokens[idx] += ' ' + leftover
                                break
                    else:
                        tokens.append(leftover)
                else:
                    # Independently emit — it has enough chars to synthesize cleanly
                    tokens.append(leftover)

            tokens.append('\n')

        # Remove trailing newline sentinel
        while tokens and tokens[-1] == '\n':
            tokens.pop()
        return tokens

    @staticmethod
    def _split_punctuation(token: str) -> tuple:
        """Split leading/trailing punctuation from a word token.
        Returns (leading, core, trailing). Uses the module-level PUNCT_CHARS set."""
        leading = ""
        trailing = ""
        core = token
        while core and core[0] in PUNCT_CHARS:
            leading += core[0]
            core = core[1:]
        while core and core[-1] in PUNCT_CHARS:
            trailing = core[-1] + trailing
            core = core[:-1]
        return leading, core, trailing


    def get_tokens(self) -> list:
        lines = self.source_text.splitlines(keepends=True)
        tokens = []
        for line in lines:
            words = line.split()
            chunk = []
            for w in words:
                chunk.append(w)
                if len(" ".join(chunk)) >= 15:
                    tokens.append(" ".join(chunk))
                    chunk = []
            if chunk:
                if tokens and tokens[-1] != '\n':
                    tokens[-1] += " " + " ".join(chunk)
                else:
                    tokens.append(" ".join(chunk))
            if line.endswith('\n'):
                tokens.append('\n')
        return tokens

    def apply_stroke_data(self, stroke_data):
        self._stroke_data = stroke_data
        self.re_layout()
        self.update()
    def re_layout(self):
        """Typeset the already-synthesized words based on current spacing/width."""
        cursor_x = 0.0
        cursor_y = 0.0
        all_strokes: list = []
        all_retry_counts: list[int] = []  # parallel to all_strokes

        for item in self._stroke_data:
            text, strokes, bbox = item[0], item[1], item[2]
            n_retries = item[3] if len(item) > 3 else 0

            if text == '\n':
                cursor_x = 0.0
                cursor_y += self.line_spacing
                continue

            w = bbox[2] - bbox[0]

            # Simple word-wrap
            if cursor_x + w > self.block_width and cursor_x > 0:
                cursor_x = 0.0
                cursor_y += self.line_spacing

            # Offset strokes from local(0,0) to cursor(x,y)
            # Y=0 corresponds to the true baseline for the synthesized word
            move_x = cursor_x - bbox[0]
            move_y = cursor_y

            for s in strokes:
                all_strokes.append([(px + move_x, py + move_y) for px, py in s])
                all_retry_counts.append(n_retries)

            cursor_x += w + self.word_spacing

        self._strokes = all_strokes
        self._stroke_retry_counts = all_retry_counts

        if all_strokes:
            xs = [x for s in all_strokes for x, y in s]
            ys = [y for s in all_strokes for x, y in s]
            self._bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            self._bbox = (0, 0, 10, 5)

        self._rebuild_path()

    def _rebuild_path(self):
        """
        Build / rebuild QPainterPath child items from current strokes.
        Each distinct retry-count bucket gets its own QGraphicsPathItem
        so strokes can be drawn in different colours:
            0 retries  → normal near-black
            1 retry    → amber
            2 retries  → burnt orange
            3+ retries → crimson
        """
        # Remove ALL old path children (there may now be multiple)
        for child in list(self.childItems()):
            if isinstance(child, QGraphicsPathItem):
                child.setParentItem(None)
                if self.scene():
                    self.scene().removeItem(child)
        self._path_item = None

        if not self._strokes:
            self.update()
            return

        retry_counts = getattr(self, '_stroke_retry_counts', [])

        # Group strokes by retry count bucket
        from collections import defaultdict
        groups: dict[int, list] = defaultdict(list)
        for idx, stroke in enumerate(self._strokes):
            if len(stroke) < 2:
                continue
            n = retry_counts[idx] if idx < len(retry_counts) else 0
            groups[n].append(stroke)

        first_item = None
        for n_retries, stroke_group in sorted(groups.items()):
            path = QPainterPath()
            for stroke in stroke_group:
                path.moveTo(stroke[0][0], stroke[0][1])
                for px, py in stroke[1:]:
                    path.lineTo(px, py)

            color_idx = min(n_retries, len(RETRY_COLORS) - 1)
            hex_color, _ = RETRY_COLORS[color_idx]

            path_item = QGraphicsPathItem(path, self)
            path_item.setPen(QPen(
                QColor(hex_color), 0.4,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            ))
            if first_item is None:
                first_item = path_item

        self._path_item = first_item  # keep backward compat reference
        self.prepareGeometryChange()
        self.update()

    # ---- QGraphicsItem interface ---------------------------------------------
    def boundingRect(self) -> QRectF:
        pad = 3.0
        min_x, min_y, max_x, max_y = self._bbox
        return QRectF(min_x - pad, min_y - pad,
                      (max_x - min_x) + pad * 2,
                      (max_y - min_y) + pad * 2)

    def paint(self, painter: QPainter, option, widget=None):
        """Draw selection/hover border only. Strokes are drawn by child path item."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.boundingRect()

        is_selected = self.isSelected()
        if is_selected:
            pen = QPen(BLOCK_BORDER_SELECTED, 0.6, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(68, 136, 255, 18)))
        else:
            pen = QPen(BLOCK_BORDER_NORMAL, 0.3, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.drawRect(rect)

        # Draw resize-corner hint dots when selected
        if is_selected:
            dot_pen = QPen(BLOCK_BORDER_SELECTED, 0.5)
            painter.setPen(dot_pen)
            painter.setBrush(QBrush(BLOCK_BORDER_SELECTED))
            dot_r = 1.5
            for cx, cy in [
                (rect.left(), rect.top()), (rect.right(), rect.top()),
                (rect.left(), rect.bottom()), (rect.right(), rect.bottom()),
            ]:
                painter.drawEllipse(QPointF(cx, cy), dot_r, dot_r)

    # ---- Events (definitive set) ---------------------------------------------

    def get_strokes_in_scene(self) -> list:
        """Return strokes translated to scene coordinates."""
        origin = self.scenePos()
        strokes = []
        for s in self._strokes:
            strokes.append([(px + origin.x(), py + origin.y()) for px, py in s])
        return strokes


from PyQt6.QtWidgets import QDockWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QTextEdit, QApplication, QMainWindow, QSplitter, QGroupBox, QFormLayout, QLineEdit, QFileDialog, QCheckBox, QMenu, QSlider, QMenuBar
from PyQt6.QtGui import QTextCursor, QIntValidator, QDoubleValidator, QAction, QUndoStack
from PyQt6.QtCore import Qt
import sys
import re

class DecimalSlider(QWidget):
    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, decimals: int = 3, suffix: str = "", parent=None):
        super().__init__(parent)
        self.multiplier = 10 ** decimals
        self.decimals = decimals
        self.suffix = suffix

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 8)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        self.label = QLabel(label)
        self.label.setStyleSheet("font-weight: bold; font-size: 13px; color: #c8c8c8;")
        top_row.addWidget(self.label)
        top_row.addStretch()

        self.input = QLineEdit(f"{default:.{decimals}f}")
        self.input.setFixedWidth(60)
        self.input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input.setValidator(QDoubleValidator(min_val, max_val, decimals))
        self.input.setStyleSheet('''
            QLineEdit {
                background-color: #3c3f41; border: 1px solid #555;
                border-radius: 3px; padding: 2px 4px;
                color: #e0e0e0; font-size: 11px; font-weight: bold;
            }
        ''')
        top_row.addWidget(self.input)
        layout.addLayout(top_row)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(min_val * self.multiplier),
                             int(max_val * self.multiplier))
        self.slider.setValue(int(default * self.multiplier))
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._slider_changed)
        self.input.editingFinished.connect(self._input_changed)

    def _slider_changed(self, v):
        self.input.setText(f"{v / self.multiplier:.{self.decimals}f}")

    def _input_changed(self):
        try:
            val = float(self.input.text())
            self.slider.setValue(int(val * self.multiplier))
        except ValueError:
            pass

    def value(self) -> float:
        return self.slider.value() / self.multiplier

    def set_value(self, val: float):
        """Programmatically set the slider value (used by block resize handle)."""
        clamped = int(max(self.slider.minimum(), min(self.slider.maximum(),
                         round(val * self.multiplier))))
        self.slider.setValue(clamped)

class BlockPropertiesDialog(QDialog):
    """Per-block style override dialog (right-click → Block Properties)."""
    def __init__(self, block, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Block Properties")
        self.setMinimumWidth(360)
        self.setStyleSheet("""
            QDialog { background-color: #252526; }
            QLabel  { color: #cccccc; font-size: 12px; }
            QPushButton {
                background-color: #0e639c; color: white; border: none;
                padding: 7px 18px; font-weight: bold; border-radius: 4px;
            }
            QPushButton:hover { background-color: #1177bb; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.addWidget(QLabel(f"Properties for block: \"{block.source_text[:40]}...\""
                                if len(block.source_text) > 40
                                else f"Properties for block: \"{block.source_text}\""))

        self.slider_scale = DecimalSlider(
            "Text Scale", 0.005, 0.100, block.scale_param, decimals=3)
        self.slider_width = DecimalSlider(
            "Block Width (mm)", 30.0, 300.0, block.block_width, decimals=1)
        self.slider_spacing = DecimalSlider(
            "Line Spacing (mm)", 2.0, 60.0, block.line_spacing, decimals=2)
        self.slider_word = DecimalSlider(
            "Word Spacing (mm)", 1.0, 20.0, block.word_spacing, decimals=1)

        for sldr in (self.slider_scale, self.slider_width,
                     self.slider_spacing, self.slider_word):
            layout.addWidget(sldr)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_props(self) -> dict:
        return {
            'scale':        self.slider_scale.value(),
            'width':        self.slider_width.value(),
            'line_spacing': self.slider_spacing.value(),
            'word_spacing': self.slider_word.value(),
        }


class ReviewPanel(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("🔬 AI Debug Log", parent)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)

        toolbar = QHBoxLayout()
        self._total_regen_label = QLabel("Regenerations: 0")
        self._total_regen_label.setStyleSheet("color: #ccc; font-weight: bold; margin-right: 15px;")
        
        self._avg_retries_label = QLabel("Avg Retries: 0.0")
        self._avg_retries_label.setStyleSheet("color: #ccc; font-weight: bold; margin-right: 15px;")

        legend_row = QHBoxLayout()
        legend_row.setSpacing(8)
        
        lbl_key = QLabel("Stroke color key:")
        lbl_key.setStyleSheet("color: #888; font-size: 11px;")
        legend_row.addWidget(lbl_key)

        _SWATCH_TMPL = (
            "QLabel {{ background-color:{bg}; color:{fg}; border-radius:4px; "
            "font-size:11px; font-weight: bold; padding:3px 8px; font-family:'Segoe UI',sans-serif; }}"
        )
        for hex_color, legend_text in RETRY_COLORS:
            fg = "#ffffff" if legend_text != "1 retry" else "#111111"
            sw = QLabel(f"■ {legend_text}" if legend_text == "clean" else legend_text)
            sw.setStyleSheet(_SWATCH_TMPL.format(bg=hex_color, fg=fg))
            legend_row.addWidget(sw)
        legend_row.addStretch()

        copy_btn = QPushButton("📋 Copy Log")
        copy_btn.setStyleSheet("background-color: #3e3e42; color: #d4d4d4; border-radius: 3px; padding: 4px 12px;")
        copy_btn.clicked.connect(self.copy_to_clipboard)

        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("background-color: #3e3e42; color: #d4d4d4; border-radius: 3px; padding: 4px 15px;")
        clear_btn.clicked.connect(self.clear)

        toolbar.addWidget(self._total_regen_label)
        toolbar.addWidget(self._avg_retries_label)
        toolbar.addLayout(legend_row)
        toolbar.addStretch()
        toolbar.addWidget(copy_btn)
        toolbar.addWidget(clear_btn)
        layout.addLayout(toolbar)

        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setStyleSheet(
            "QTextEdit { background-color: #0d1117; color: #c9d1d9; font-family: 'Consolas', monospace; font-size: 13px; border: 1px solid #21262d; border-radius: 4px; padding: 4px; }"
        )
        layout.addWidget(self._log_edit)

        self._regen_count = 0
        self._total_chunks = 0
        self.setWidget(container)

    def append(self, msg: str):
        if re.search(r"\[\d+/\d+\].*init score=", msg):
            self._regen_count += 1
            if self._total_chunks == 0:
                self._total_chunks = 1
            self._total_regen_label.setText(f"Regenerations: {self._regen_count}")
            avg = self._regen_count / self._total_chunks
            self._avg_retries_label.setText(f"Avg Retries: {avg:.2f}")

        if "✓" in msg:
            self._total_chunks += 1
            if self._total_chunks > 0:
                avg = self._regen_count / self._total_chunks
                self._avg_retries_label.setText(f"Avg Retries: {avg:.2f}")

        html_msg = msg.replace("<", "&lt;").replace(">", "&gt;")
        def rep_tag(match):
            tag = match.group(1)
            color = "#8b949e"
            if "REJECT" in tag or "FAIL" in tag or "✗" in tag:
                color = "#f85149"
            elif "OK" in tag or "PASS" in tag or "✓" in tag:
                color = "#3fb950"
            elif ">>" in tag:
                color = "#58a6ff"
            elif "WARN" in tag:
                color = "#d29922"
            return f'<span style="color:{color}; font-weight:bold;">[{tag}]</span>'
            
        html_msg = re.sub(r"\[(.*?)\]", rep_tag, html_msg)
        self._log_edit.append(html_msg)
        cursor = self._log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_edit.setTextCursor(cursor)
        self._log_edit.ensureCursorVisible()

    def clear(self):
        self._regen_count = 0
        self._total_chunks = 0
        self._total_regen_label.setText("Regenerations: 0")
        self._avg_retries_label.setText("Avg Retries: 0.0")
        self._log_edit.clear()

    def copy_to_clipboard(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._log_edit.toPlainText())


# ---------------------------------------------------------------------------
# Style presets
# Keys: bias, scale, line_spacing, word_spacing, min_chunk, threshold, retries
#
# Design notes:
#   - Neat uses bias=0.75 (not 0.50).  bias<0.75 makes stochastic sampling
#     unreliable: the MDN stays flat so sampling occasionally hits degenerate
#     mixture components.  0.75 is the sweet spot for clean + stable strokes.
#   - threshold decreases as style gets looser because legitimate Casual/Loose
#     strokes have lower X-monotonicity and wider arcs (which the scorer
#     would incorrectly penalise at Normal thresholds).
#   - retries decrease as style gets looser because the scorer is more lenient.
# ---------------------------------------------------------------------------
STYLE_PRESETS: dict[str, dict] = {
    "✏️ Neat":    dict(bias=0.75, scale=0.012, line_spacing=8.0,  word_spacing=3.5, min_chunk=12, threshold=0.65, retries=6),
    "📝 Normal":  dict(bias=1.00, scale=0.015, line_spacing=10.0, word_spacing=5.0, min_chunk=8,  threshold=0.60, retries=4),
    "🗒️ Casual":  dict(bias=1.40, scale=0.017, line_spacing=11.0, word_spacing=5.5, min_chunk=7,  threshold=0.50, retries=3),
    "✍️ Loose":   dict(bias=1.80, scale=0.019, line_spacing=12.0, word_spacing=6.5, min_chunk=6,  threshold=0.45, retries=3),
    "🌀 Sloppy":  dict(bias=2.00, scale=0.022, line_spacing=14.0, word_spacing=7.0, min_chunk=5,  threshold=0.40, retries=2),
}


class PlotterStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plotter Studio v2.0 — Canvas Layout Editor")
        self.resize(1600, 950)

        self.engine = SynthesisEngine()
        self._blocks = []
        self.undo_stack = QUndoStack(self)

        self._apply_global_style()
        self._build_menu()
        self._build_ui()

    def _apply_preset(self, name: str):
        """Apply a named style preset to all sidebar sliders and AI review settings."""
        p = STYLE_PRESETS.get(name)
        if not p:
            return
        self.slider_bias.set_value(p['bias'])
        self.slider_scale.set_value(p['scale'])
        self.slider_spacing.set_value(p['line_spacing'])
        self.slider_word.set_value(p['word_spacing'])
        self.slider_min_chunk.set_value(float(p['min_chunk']))
        # Also tune the AI reviewer for this style
        if 'threshold' in p:
            self.slider_quality.set_value(p['threshold'])
        if 'retries' in p:
            self.spin_max_retries.setValue(p['retries'])
        self.statusBar().showMessage(
            f"Preset '{name}' applied — bias={p['bias']}, threshold={p.get('threshold','?')}, "
            f"retries={p.get('retries','?')}, min_chunk={p['min_chunk']}",
            5000
        )

    def _apply_global_style(self):
        self.setStyleSheet('''
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI', 'Roboto', sans-serif; }
            QGroupBox { border: 1px solid #3e3e42; border-radius: 6px; margin-top: 10px; padding-top: 14px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; background-color: #0e639c; }
            QPushButton:hover { background-color: #1177bb; }
            QSplitter::handle { background-color: #3e3e42; }
            QDockWidget::title { background: #252526; padding-left: 10px; padding-top: 4px; color: #e0e0e0; font-weight: bold; }
        ''')

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        add_act = QAction("Add Text Block...", self)
        add_act.setShortcut("Ctrl+N")
        add_act.triggered.connect(self._on_add_block)
        file_menu.addAction(add_act)
        file_menu.addSeparator()

        open_act = QAction("Open Session...", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._on_open_session)
        file_menu.addAction(open_act)

        save_act = QAction("Save Session...", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._on_save_session)
        file_menu.addAction(save_act)

        file_menu.addSeparator()

        undo_act = QAction("Undo", self)
        undo_act.setShortcut("Ctrl+Z")
        undo_act.triggered.connect(self.undo_stack.undo)
        file_menu.addAction(undo_act)

        redo_act = QAction("Redo", self)
        redo_act.setShortcut("Ctrl+Y")
        redo_act.triggered.connect(self.undo_stack.redo)
        file_menu.addAction(redo_act)

        export_menu = menu.addMenu("Export")
        export_gc = QAction("Export G-Code...", self)
        export_gc.setShortcut("Ctrl+E")
        export_gc.triggered.connect(self._on_export)
        export_menu.addAction(export_gc)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.setSizes([310, 1290])
        root_layout.addWidget(splitter)
        
        self.review_panel = ReviewPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.review_panel)

    def _build_control_panel(self):
        from PyQt6.QtWidgets import QSpinBox, QScrollArea
        panel = QWidget()
        panel.setFixedWidth(310)
        panel.setStyleSheet("background-color: #252526; border-right: 1px solid #3e3e42;")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #252526; }")

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(15, 10, 15, 15)
        layout.setSpacing(6)

        # --- Plotter Studio title ---
        title = QLabel("Plotter Studio v2.0")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff; padding-bottom: 2px;")
        layout.addWidget(title)

        # --- Style Presets -----------------------------------------
        preset_hdr = QLabel("Style Preset")
        preset_hdr.setStyleSheet("font-size: 11px; font-weight: bold; color: #888; padding-top: 6px;")
        layout.addWidget(preset_hdr)

        preset_combo_row = QHBoxLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.setStyleSheet("""
            QComboBox {
                background-color: #3c3f41; color: #e0e0e0; border: 1px solid #555;
                border-radius: 4px; padding: 5px 8px; font-size: 13px; font-weight: bold;
            }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #2d2d30; color: #e0e0e0;
                selection-background-color: #094771;
            }
        """)
        for name in STYLE_PRESETS:
            self.combo_preset.addItem(name)
        self.combo_preset.setCurrentText("📝 Normal")
        self.combo_preset.currentTextChanged.connect(self._apply_preset)
        preset_combo_row.addWidget(self.combo_preset, 1)

        btn_add_block = QPushButton("＋ Add Block")
        btn_add_block.setFixedHeight(36)
        btn_add_block.setStyleSheet("""
            QPushButton {
                background-color: #0e639c; color: white; font-weight: bold;
                border-radius: 5px; padding: 4px 10px; font-size: 12px;
            }
            QPushButton:hover { background-color: #1177bb; }
        """)
        btn_add_block.clicked.connect(self._on_add_block)
        preset_combo_row.addWidget(btn_add_block)
        layout.addLayout(preset_combo_row)

        # --- Synthesis Parameters header ---
        synth_hdr = QLabel("Synthesis Parameters")
        synth_hdr.setStyleSheet("font-size: 11px; font-weight: bold; color: #888; padding-top: 6px;")
        layout.addWidget(synth_hdr)

        # Sliders
        self.slider_bias = DecimalSlider("Bias / Sloppiness", 0.10, 2.00, 1.00, decimals=2)
        self.slider_scale = DecimalSlider("Text Scale", 0.005, 0.100, 0.015, decimals=3)
        self.slider_spacing = DecimalSlider("Line Spacing (mm)", 2.00, 60.00, 10.00, decimals=2)
        self.slider_word = DecimalSlider("Word Spacing (mm)", 1.0, 20.0, 5.0, decimals=1)
        self.slider_blk_width = DecimalSlider("Block Width (mm)", 30.0, 300.0, 190.0, decimals=1)
        self.slider_min_chunk = DecimalSlider("Min Phrase (chars)", 4.0, 30.0, 8.0, decimals=0)

        layout.addWidget(self.slider_bias)
        layout.addWidget(self.slider_scale)
        layout.addWidget(self.slider_spacing)
        layout.addWidget(self.slider_word)
        layout.addWidget(self.slider_blk_width)
        layout.addWidget(self.slider_min_chunk)

        # --- Hardware / G-Code ---
        hw_group = QGroupBox("Hardware / G-Code")
        hw_layout = QFormLayout(hw_group)
        hw_layout.setSpacing(8)

        field_style = '''
            QLineEdit {
                background-color: #3c3f41; border: 1px solid #555;
                border-radius: 3px; padding: 4px 8px; color: #e0e0e0;
                font-family: 'Consolas', 'Courier New', monospace; font-size: 12px;
            }
        '''
        self.input_draw_speed = QLineEdit("1500")
        self.input_draw_speed.setStyleSheet(field_style)
        self.input_move_speed = QLineEdit("3000")
        self.input_move_speed.setStyleSheet(field_style)
        self.input_pen_down = QLineEdit("M3 S90")
        self.input_pen_down.setStyleSheet(field_style)
        self.input_pen_up = QLineEdit("M3 S0")
        self.input_pen_up.setStyleSheet(field_style)

        for text, widget in [
            ("Draw Speed (F):", self.input_draw_speed),
            ("Move Speed (F):", self.input_move_speed),
            ("Pen Down Cmd:",   self.input_pen_down),
            ("Pen Up Cmd:",     self.input_pen_up),
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #aaa; font-size: 11px;")
            hw_layout.addRow(lbl, widget)

        self.chk_flip_axes = QCheckBox("Flip X/Y Axes")
        self.chk_flip_axes.setStyleSheet("color: #e0e0e0; font-size: 12px; margin-top: 4px;")
        hw_layout.addRow(self.chk_flip_axes)
        layout.addWidget(hw_group)

        # --- AI Review ---
        ai_group = QGroupBox("AI Review")
        ai_layout = QVBoxLayout(ai_group)
        ai_layout.setSpacing(6)

        self.chk_ai_review = QCheckBox("Enable AI Review")
        self.chk_ai_review.setChecked(True)
        self.chk_ai_review.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 12px;")
        ai_layout.addWidget(self.chk_ai_review)

        ai_desc = QLabel("Auto-detects and retries bad chunks\n(dots, collapses, runaway sequences)")
        ai_desc.setStyleSheet("color: #888; font-size: 10px;")
        ai_desc.setWordWrap(True)
        ai_layout.addWidget(ai_desc)

        self.slider_quality = DecimalSlider("Quality Threshold", 0.00, 1.00, 0.60, decimals=2)
        ai_layout.addWidget(self.slider_quality)

        retries_row = QHBoxLayout()
        retries_lbl = QLabel("Max Retries per Chunk")
        retries_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #c8c8c8;")
        retries_row.addWidget(retries_lbl)
        retries_row.addStretch()
        self.spin_max_retries = QSpinBox()
        self.spin_max_retries.setRange(1, 20)
        self.spin_max_retries.setValue(4)
        self.spin_max_retries.setFixedWidth(60)
        self.spin_max_retries.setStyleSheet('''
            QSpinBox {
                background-color: #3c3f41; border: 1px solid #555;
                border-radius: 3px; padding: 2px 4px;
                color: #e0e0e0; font-size: 11px; font-weight: bold;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #555; border: none; width: 16px;
            }
        ''')
        retries_row.addWidget(self.spin_max_retries)
        ai_layout.addLayout(retries_row)

        layout.addWidget(ai_group)

        layout.addStretch(1)
        scroll.setWidget(inner)

        wrapper = QVBoxLayout(panel)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(scroll)
        return panel

    def _build_canvas_panel(self):
        from PyQt6.QtWidgets import QGraphicsScene, QGraphicsRectItem
        from PyQt6.QtGui import QBrush, QColor, QPen
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        top_hint = QLabel("🖱️ Drag blocks to reposition · Right-click block → Regenerate / Edit / Delete · Scroll to zoom · Middle-drag to pan")
        top_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_hint.setStyleSheet("background-color: #1e1e1e; padding: 6px; font-size: 11px; color: #a0a0a0;")
        layout.addWidget(top_hint)

        self.scene = QGraphicsScene()
        self.scene.setSceneRect(-20, -20, 250, 330)
        paper = QGraphicsRectItem(0, 0, 210, 297)
        paper.setBrush(QBrush(QColor("#ffffff")))
        self.scene.addItem(paper)

        if 'A4PreviewView' in globals():
            self.view = A4PreviewView(self.scene)
        else:
            self.view = InteractiveGraphicsView()
            self.view.setScene(self.scene)
        
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        layout.addWidget(self.view)
        
        self.btn_debug_log = QPushButton("🔬 Debug Log")
        self.btn_debug_log.setCheckable(True)
        self.btn_debug_log.setChecked(True)
        self.btn_debug_log.setStyleSheet('''
            QPushButton {
                background-color: #1e3a5f;
                color: #e0e0e0;
                font-weight: bold;
                border: 1px solid #3e6bb3;
                border-radius: 4px;
                padding: 6px;
                margin: 0px 4px 4px 4px;
            }
            QPushButton:hover {
                background-color: #2e5a8f;
            }
            QPushButton:checked {
                background-color: #1a2a44;
                border: 1px solid #2e4b78;
            }
        ''')
        self.btn_debug_log.toggled.connect(self._on_toggle_review_panel)
        layout.addWidget(self.btn_debug_log)
        
        return panel

    def _on_toggle_review_panel(self, checked):
        if hasattr(self, 'review_panel'):
            self.review_panel.setVisible(checked)

    def _on_add_block(self):
        dlg = AddBlockDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        text = dlg.get_text()
        if not text: return
        self._create_and_place_block(text)

    def _create_and_place_block(self, text):
        scale = max(0.001, self.slider_scale.value())
        stochastic = True

        block = HandwritingBlock(
            text, scale,
            line_spacing=self.slider_spacing.value(),
            block_width=self.slider_blk_width.value(),
            word_spacing=self.slider_word.value(),
        )
        block.signals.regenerate_requested.connect(self._on_regenerate_block)
        block.signals.delete_requested.connect(self._on_delete_block)
        block.signals.edit_requested.connect(self._on_edit_block)

        block.setPos(20, 20 if not self._blocks else self._blocks[-1].scenePos().y() + 40)
        self.undo_stack.push(AddBlockCommand(self.scene, block, self._blocks))
        self._run_synthesis_worker(block, stochastic)

    def _on_edit_block(self, block):
        dlg = AddBlockDialog(self, initial_text=block.source_text)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_text = dlg.get_text()
        if not new_text:
            return
        block.source_text = new_text
        self._run_synthesis_worker(block, stochastic=True)

    def _on_regenerate_block(self, block):
        stochastic = True
        self._run_synthesis_worker(block, stochastic)

    def _on_delete_block(self, block):
        self.undo_stack.push(DeleteBlockCommand(self.scene, block, self._blocks))

    def _run_synthesis_worker(self, block, stochastic):
        ai_review = self.chk_ai_review.isChecked()
        # Read actual UI values
        review_threshold = self.slider_quality.value()
        review_retries = self.spin_max_retries.value()
        min_chunk = max(4, int(self.slider_min_chunk.value()))

        # Update engine bias when slider changes
        self.engine.set_bias(self.slider_bias.value())

        tokens = HandwritingBlock._tokenize(block.source_text, min_chunk_chars=min_chunk)
        self._worker = SynthesisWorker(
            block, self.engine, tokens, block.scale_param,
            stochastic, ai_review, review_threshold, review_retries
        )
        self._worker.log_msg.connect(self.review_panel.append)
        self._worker.chunk_ready.connect(self._on_worker_status)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_status(self, msg: str):
        self.statusBar().showMessage(msg, 4000)

    def _on_worker_error(self, msg: str):
        QMessageBox.critical(self, "Synthesis Error", msg)
        self.statusBar().showMessage(f"Error: {msg}", 8000)

    def _on_worker_finished(self, block, stroke_data):
        block.apply_stroke_data(stroke_data)
        block.update()
        self.statusBar().showMessage("Synthesis complete.", 3000)

    def _on_export(self):
        all_strokes = []
        for blk in self._blocks:
            all_strokes.extend(blk.get_strokes_in_scene())

        if not all_strokes:
            QMessageBox.warning(self, "Export", "No strokes to export. Add a block first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export G-Code", "output.gcode",
            "G-Code (*.gcode *.nc);;All Files (*)"
        )
        if not path:
            return

        try:
            draw_speed  = int(self.input_draw_speed.text() or "1500")
        except ValueError:
            draw_speed = 1500
        try:
            move_speed  = int(self.input_move_speed.text() or "3000")
        except ValueError:
            move_speed = 3000

        pen_down = self.input_pen_down.text() or "M3 S90"
        pen_up   = self.input_pen_up.text()   or "M3 S0"
        flip     = self.chk_flip_axes.isChecked()

        lines = self.engine.compile_gcode(
            all_strokes,
            pen_up_cmd=pen_up, pen_down_cmd=pen_down,
            draw_speed=draw_speed, rapid_speed=move_speed,
            flip_axes=flip,
        )

        with open(path, "w") as f:
            f.write("\n".join(lines))

        QMessageBox.information(self, "Export G-Code",
                                f"{len(lines)} lines written to:\n{path}")
        self.statusBar().showMessage(f"Exported {len(lines)} G-code lines.", 5000)

    # ------------------------------------------------------------------
    # Session save / load
    # ------------------------------------------------------------------
    def _on_save_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "session.pstudio",
            "Plotter Studio Session (*.pstudio);;All Files (*)"
        )
        if not path:
            return

        data = {"blocks": []}
        for blk in self._blocks:
            pos = blk.scenePos()
            blk_data = {
                "text":          blk.source_text,
                "x":             pos.x(),
                "y":             pos.y(),
                "scale":         blk.scale_param,
                "line_spacing":  blk.line_spacing,
                "block_width":   blk.block_width,
                "word_spacing":  blk.word_spacing,
                # Embed stroke data so reload is instant (no re-synthesis needed)
                "stroke_data": [
                    [itm[0],
                     [[list(pt) for pt in stroke] for stroke in itm[1]],
                     list(itm[2]),
                     itm[3] if len(itm) > 3 else 0]
                    for itm in blk._stroke_data
                ],
            }
            data["blocks"].append(blk_data)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"Session saved → {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_open_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "",
            "Plotter Studio Session (*.pstudio);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Open Error", str(e))
            return

        # Clear existing blocks
        for blk in list(self._blocks):
            self.scene.removeItem(blk)
        self._blocks.clear()
        self.undo_stack.clear()

        for bd in data.get("blocks", []):
            blk = HandwritingBlock(
                bd["text"],
                bd.get("scale", 0.015),
                line_spacing=bd.get("line_spacing", 10.0),
                block_width=bd.get("block_width", 170.0),
                word_spacing=bd.get("word_spacing", 5.0),
            )
            blk.signals.regenerate_requested.connect(self._on_regenerate_block)
            blk.signals.delete_requested.connect(self._on_delete_block)
            blk.signals.edit_requested.connect(self._on_edit_block)

            self.scene.addItem(blk)
            self._blocks.append(blk)
            blk.setPos(bd.get("x", 20), bd.get("y", 20))

            raw_sd = bd.get("stroke_data", [])
            if raw_sd:
                stroke_data = []
                for item in raw_sd:
                    text  = item[0]
                    strks = [list(map(tuple, s)) for s in item[1]]
                    bbox  = tuple(item[2])
                    tries = item[3] if len(item) > 3 else 0
                    stroke_data.append((text, strks, bbox, tries))
                blk.apply_stroke_data(stroke_data)
            else:
                self._run_synthesis_worker(blk, stochastic=True)

        self.statusBar().showMessage(f"Session loaded: {len(self._blocks)} blocks.", 5000)


def main():

    from PyQt6.QtWidgets import QApplication
    import sys
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    window = PlotterStudio()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
