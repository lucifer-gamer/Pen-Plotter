"""
stroke_reviewer_v4.py  (v4 — text-aware geometric scoring engine)
==================================================================
Extends the v3 weighted sub-score engine with text-aware heuristics:

  * Per-character width weighting (narrow vs wide chars)
  * Ascender / descender topology validation
  * Ink-density pooling detector (stuck-pen / attention-loop)

Architecture:
  The six original V3 sub-scores are preserved with IDENTICAL weights.
  Topology and density are applied as MULTIPLICATIVE PENALTIES on top
  of the core V3 score.  They can only REDUCE a score, never inflate it.
  This preserves V3's proven detection sensitivity while catching NEW
  failure modes (missing descenders, stuck-pen loops).

Core weights (V3-identical, sum to 1.0):
  arc_score      0.28  — stroke arc-length relative to expectation
  x_mono_score   0.18  — left-to-right monotonicity
  extrema_score  0.24  — vertical direction changes per char (legibility)
  stroke_score   0.10  — stroke-count plausibility
  width_score    0.12  — bounding-box width sanity (now proportional)
  height_score   0.08  — bounding-box height sanity

Multiplicative penalties (applied after core score):
  topology  — ascender/descender shape validation  [NEW]
  density   — ink-pooling / stuck-pen detection     [NEW]

No external API calls, no GPU needed — runs in milliseconds per chunk.
"""

from __future__ import annotations
import math
import statistics
from datetime import datetime


# ---------------------------------------------------------------------------
# Character trait dictionaries
# ---------------------------------------------------------------------------

# Characters whose strokes should extend significantly above the x-height
ASCENDERS = set('bdfhklt')

# Characters whose strokes should drop below the baseline
DESCENDERS = set('gjpqy')

# Narrow characters — expect ~60% of average character width
NARROW_CHARS = set('iIlL1!|,.:;\'')

# Wide characters — expect ~130-140% of average character width
WIDE_CHARS = set('wWmMOQDG@#%')

# Characters that are about average width (everything else)
# No explicit set needed — falls through to 1.0 multiplier

# Width multipliers per character
CHAR_WIDTH_FACTOR: dict[str, float] = {}
for _c in NARROW_CHARS:
    CHAR_WIDTH_FACTOR[_c] = 0.60
for _c in WIDE_CHARS:
    CHAR_WIDTH_FACTOR[_c] = 1.35


# ---------------------------------------------------------------------------
# Core scoring weights — IDENTICAL to V3 (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "arc":     0.28,
    "x_mono":  0.18,
    "extrema": 0.24,
    "strokes": 0.10,
    "width":   0.12,
    "height":  0.08,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, f"Weights must sum to 1.0, got {sum(WEIGHTS.values())}"

# Multiplicative penalty keys (not additive — these only reduce the core score)
PENALTY_KEYS = ("topology", "density")


# ---------------------------------------------------------------------------
# Internal helpers  (carried forward from v3)
# ---------------------------------------------------------------------------

def _arc_length(strokes: list) -> float:
    """Total pen-down arc length across all strokes (in scene units)."""
    total = 0.0
    for stroke in strokes:
        for (x1, y1), (x2, y2) in zip(stroke, stroke[1:]):
            total += math.hypot(x2 - x1, y2 - y1)
    return total


def _bbox(strokes: list) -> tuple:
    """Bounding box (min_x, min_y, max_x, max_y) from stroke data."""
    if not strokes:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [x for s in strokes for x, y in s]
    ys = [y for s in strokes for x, y in s]
    return (min(xs), min(ys), max(xs), max(ys))


def _total_points(strokes: list) -> int:
    """Total number of points across all strokes."""
    return sum(len(s) for s in strokes)


def _y_extrema(strokes: list, hysteresis: float = 0.5) -> int:
    """
    Count the number of vertical direction changes in the stroke path.
    Legible handwriting typically requires > 1.2 Y-extrema per character.
    Uses hysteresis to avoid counting noise as legitimate direction changes.
    """
    y_extrema = 0
    for s in strokes:
        if len(s) < 2:
            continue

        direction = 0
        extreme_y = s[0][1]
        start_min_y = s[0][1]
        start_max_y = s[0][1]

        for x, y in s[1:]:
            if direction == 0:
                start_min_y = min(start_min_y, y)
                start_max_y = max(start_max_y, y)

                if y - start_min_y > hysteresis:
                    direction = 1
                    extreme_y = y
                elif start_max_y - y > hysteresis:
                    direction = -1
                    extreme_y = y

            elif direction == 1:
                if y > extreme_y:
                    extreme_y = y
                elif extreme_y - y > hysteresis:
                    direction = -1
                    y_extrema += 1
                    extreme_y = y

            elif direction == -1:
                if y < extreme_y:
                    extreme_y = y
                elif y - extreme_y > hysteresis:
                    direction = 1
                    y_extrema += 1
                    extreme_y = y
    return y_extrema


def _x_monotonicity(strokes: list) -> float:
    """
    Measure left-to-right progress ratio across all stroke points.
    Returns fraction of consecutive point-pairs where dx >= 0.
    """
    forward = 0
    total = 0
    for stroke in strokes:
        for (x1, _), (x2, _) in zip(stroke, stroke[1:]):
            total += 1
            if x2 >= x1:
                forward += 1
    if total == 0:
        return 0.5
    return forward / total


# ---------------------------------------------------------------------------
# NEW: Text-aware helpers
# ---------------------------------------------------------------------------

def _proportional_expected_width(text: str, median_char_width: float) -> float:
    """
    Compute expected chunk width using per-character width factors.

    Instead of naive `len(text) * avg_width`, narrow chars like 'i' contribute
    only 60% of the average width while wide chars like 'w' contribute 135%.
    This produces tighter and more accurate width expectations.
    """
    total = 0.0
    char_count = 0
    for ch in text.strip():
        if ch == ' ':
            total += median_char_width * 0.45  # space is about half a char
        else:
            factor = CHAR_WIDTH_FACTOR.get(ch, 1.0)
            total += median_char_width * factor
            char_count += 1
    return max(total, median_char_width)  # at least one char wide


def _topology_analysis(text: str, strokes: list, bbox: tuple,
                       median_height: float) -> float:
    """
    Text-aware topology sub-score (zone-based with absolute reference).

    Uses the block's median_height as the expected body height to define
    absolute zone boundaries, rather than the chunk's own bounding box.
    This way, a chunk missing its descender will have a shorter bbox than
    expected, and the descender zone won't be reached by any ink.

    Zones (relative to the chunk's vertical center):
      - Ascender zone:  y < center - median_height * 0.55
      - Body zone:      between ascender and descender
      - Descender zone: y > center + median_height * 0.55

    Returns 0.0 to 1.0.
    """
    if not strokes or not text.strip():
        return 0.0

    core = text.strip().lower()
    min_y = bbox[1]
    max_y = bbox[3]
    h = max_y - min_y
    if h < 0.01 or median_height < 0.01:
        return 0.0

    has_ascenders = any(c in ASCENDERS for c in core)
    has_descenders = any(c in DESCENDERS for c in core)

    # Collect all Y values
    all_ys = [y for s in strokes for _, y in s]
    n = len(all_ys)
    if n < 2:
        return 0.0

    # Use the median of all Y as the vertical center of the body
    all_ys_sorted = sorted(all_ys)
    center_y = all_ys_sorted[n // 2]

    # Zone boundaries based on the block's median height (absolute reference)
    # Body zone is centered on center_y, spanning ±0.45 * median_height
    body_half = median_height * 0.45
    asc_boundary = center_y - body_half      # above this = ascender zone
    desc_boundary = center_y + body_half     # below this = descender zone

    # Count points in each zone
    pts_asc = sum(1 for y in all_ys if y < asc_boundary)
    pts_desc = sum(1 for y in all_ys if y > desc_boundary)
    frac_asc = pts_asc / n
    frac_desc = pts_desc / n

    score = 1.0

    # --- Descender validation ---
    if has_descenders:
        # Text requires descenders: ink must reach the descender zone
        if frac_desc < 0.02:
            score *= 0.25   # Descender completely absent
        elif frac_desc < 0.05:
            score *= 0.60   # Stunted descender
    else:
        # No descenders expected: penalise ink leaking into descender zone
        if frac_desc > 0.15:
            score *= 0.50
        elif frac_desc > 0.10:
            score *= 0.75

    # --- Ascender validation ---
    if has_ascenders:
        if frac_asc < 0.02:
            score *= 0.30   # Ascender missing
        elif frac_asc < 0.05:
            score *= 0.65   # Stunted ascender
    else:
        if frac_asc > 0.15:
            score *= 0.55
        elif frac_asc > 0.10:
            score *= 0.80

    # --- Height plausibility (text-aware) ---
    if has_ascenders and has_descenders:
        # Text like "dog" should be taller than body-only text
        if h < median_height * 0.9:
            score *= 0.55   # suspiciously compact for asc+desc text
    elif has_ascenders or has_descenders:
        if h < median_height * 0.7:
            score *= 0.65   # somewhat short for text needing asc or desc
    else:
        # Pure x-height text should be compact
        if h > median_height * 2.2:
            score *= 0.55

    return max(0.0, min(1.0, score))




def _ink_density_score(strokes: list, bbox: tuple, grid_size: int = 8) -> float:
    """
    Ink-density pooling detector.

    Divides the chunk's bounding box into a grid_size × grid_size grid and
    counts how many stroke points fall into each cell. If any single cell
    contains a disproportionate share of all points, it means the pen got
    "stuck" in an attention loop — a common RNN failure mode.

    Returns 1.0 for well-distributed ink, 0.0 for extreme pooling.
    """
    if not strokes:
        return 0.0

    min_x, min_y, max_x, max_y = bbox
    w = max_x - min_x
    h = max_y - min_y
    if w < 0.01 or h < 0.01:
        return 0.0

    total_pts = 0
    grid = [[0] * grid_size for _ in range(grid_size)]

    for stroke in strokes:
        for x, y in stroke:
            # Map to grid cell
            col = min(grid_size - 1, int((x - min_x) / w * grid_size))
            row = min(grid_size - 1, int((y - min_y) / h * grid_size))
            grid[row][col] += 1
            total_pts += 1

    if total_pts == 0:
        return 0.0

    max_cell = max(cell for row in grid for cell in row)
    concentration = max_cell / total_pts

    # Also check how many cells are actually used (sparsity)
    occupied_cells = sum(1 for row in grid for cell in row if cell > 0)
    total_cells = grid_size * grid_size
    coverage = occupied_cells / total_cells

    # Scoring:
    # - concentration > 0.40 is suspicious (40% of all points in one cell)
    # - coverage < 0.10 is suspicious (ink only in < 10% of cells)
    if concentration > 0.50:
        return 0.05   # Extremely pooled — almost certainly a stuck-pen loop
    if concentration > 0.40:
        return 0.25
    if concentration > 0.30:
        return 0.55
    if concentration > 0.25:
        return 0.75

    # Coverage bonus: well-spread ink across the box
    if coverage < 0.08:
        return 0.30   # Very sparse — likely just a dot or short scrawl
    if coverage < 0.15:
        return 0.60

    return 1.0


# ---------------------------------------------------------------------------
# Block-level reference statistics (enhanced)
# ---------------------------------------------------------------------------

class BlockStats:
    """
    Captures median width-per-character and median height from an initial
    synthesis pass so the reviewer has a reference baseline.

    Enhanced over v3 to use proportional character widths.
    """

    MIN_ARC_PER_CHAR: float = 1.2
    MIN_CHAR_WIDTH:   float = 1.5
    MIN_HEIGHT:       float = 2.5

    def __init__(self, stroke_data: list):
        widths_per_char: list[float] = []
        heights: list[float] = []
        x_monos: list[float] = []

        for item in stroke_data:
            text, strokes, bbox = item[0], item[1], item[2]
            if text == '\n' or not strokes:
                continue
            char_count = len(text.strip())
            if char_count == 0:
                continue

            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            arc = _arc_length(strokes)

            if arc < char_count * self.MIN_ARC_PER_CHAR * 0.3:
                continue
            if len(strokes) < max(1, char_count // 4):
                continue

            if w > 0:
                widths_per_char.append(w / char_count)
            if h > 0:
                heights.append(h)

            mono = _x_monotonicity(strokes)
            if 0.3 < mono < 0.95:
                x_monos.append(mono)

        self.median_char_width: float = (
            statistics.median(widths_per_char)
            if len(widths_per_char) >= 2
            else self.MIN_CHAR_WIDTH
        )
        self.median_char_width = max(self.median_char_width, self.MIN_CHAR_WIDTH)

        self.median_height: float = (
            statistics.median(heights)
            if len(heights) >= 2
            else self.MIN_HEIGHT
        )
        self.median_height = max(self.median_height, self.MIN_HEIGHT)

        self.median_x_mono: float = (
            statistics.median(x_monos)
            if len(x_monos) >= 2
            else 0.68
        )
        self.median_x_mono = max(0.55, min(0.85, self.median_x_mono))

    def expected_width(self, text: str) -> float:
        """Text-aware proportional width estimation."""
        return _proportional_expected_width(text, self.median_char_width)

    def expected_arc(self, text: str) -> float:
        char_count = max(1, len(text.strip()))
        stats_arc = self.expected_width(text) * 2.5
        absolute_floor = char_count * self.MIN_ARC_PER_CHAR
        return max(stats_arc, absolute_floor)

    def expected_strokes(self, text: str) -> tuple[float, float]:
        char_count = max(1, len(text.strip()))
        return (max(1, char_count * 0.35), char_count * 3.0)


# ---------------------------------------------------------------------------
# Per-chunk sub-score functions (v3 originals — unchanged)
# ---------------------------------------------------------------------------

def _arc_sub_score(arc: float, expected_arc: float) -> float:
    if expected_arc <= 0:
        return 1.0
    ratio = arc / expected_arc
    if ratio >= 0.80:
        return 1.0
    if ratio >= 0.60:
        return 0.75
    if ratio >= 0.45:
        return 0.40
    if ratio >= 0.25:
        return 0.15
    return 0.0


def _x_mono_sub_score(mono: float, ref_mono: float) -> float:
    lo = max(0.45, ref_mono - 0.15)
    hi = min(0.92, ref_mono + 0.15)

    if lo <= mono <= hi:
        return 1.0
    if mono < lo:
        gap = lo - mono
        if gap < 0.08:
            return 0.70
        if gap < 0.15:
            return 0.35
        return 0.0
    gap = mono - hi
    if gap < 0.08:
        return 0.80
    if gap < 0.15:
        return 0.45
    return 0.10


def _extrema_sub_score(extrema_per_char: float) -> float:
    if extrema_per_char >= 1.1:
        return 1.0
    if extrema_per_char >= 0.9:
        return 0.82
    if extrema_per_char >= 0.7:
        return 0.50
    if extrema_per_char >= 0.4:
        return 0.20
    return 0.0


def _strokes_sub_score(n_strokes: int, min_exp: float, max_exp: float) -> float:
    if min_exp <= n_strokes <= max_exp:
        return 1.0
    if n_strokes < min_exp:
        ratio = n_strokes / min_exp
        if ratio >= 0.7:
            return 0.60
        if ratio >= 0.4:
            return 0.25
        return 0.0
    ratio = max_exp / n_strokes
    if ratio >= 0.6:
        return 0.70
    if ratio >= 0.35:
        return 0.30
    return 0.05


def _width_sub_score(w: float, expected_w: float) -> float:
    if expected_w <= 0:
        return 1.0
    ratio = w / expected_w
    if 0.80 <= ratio <= 1.60:
        return 1.0
    if ratio > 1.60:
        if ratio < 2.2:
            return 0.60
        if ratio < 3.0:
            return 0.20
        return 0.0
    if ratio >= 0.60:
        return 0.65
    if ratio >= 0.35:
        return 0.20
    return 0.0


def _height_sub_score(h: float, median_height: float) -> float:
    if median_height <= 0:
        return 1.0
    ratio = h / median_height
    if 0.55 <= ratio <= 2.0:
        return 1.0
    if ratio > 2.0:
        if ratio < 3.0:
            return 0.50
        return 0.0
    if ratio >= 0.35:
        return 0.45
    return 0.0


# ---------------------------------------------------------------------------
# Aggregate chunk score  (v4 — V3 core + multiplicative penalties)
# ---------------------------------------------------------------------------

def score_chunk(text: str, strokes: list, bbox: tuple, stats: BlockStats) -> float:
    """
    Return a quality score in [0.0, 1.0] for a single synthesized chunk.

    V4 architecture:
      1. Compute V3 core score from 6 weighted sub-scores (identical to v3).
      2. Compute topology and density as penalty multipliers [0..1].
      3. Final = core_score × topology × density.

    This means topology and density can only REDUCE a score, never inflate it.
    A perfect chunk scores exactly like V3.  A chunk with missing descenders
    or pooled ink gets aggressively penalised on top.
    """
    if not strokes:
        return 0.0

    char_count = max(1, len(text.strip()))

    # --- V3 core sub-scores (identical weights) ---
    arc = _arc_length(strokes)
    expected_arc = stats.expected_arc(text)
    arc_s = _arc_sub_score(arc, expected_arc)

    mono = _x_monotonicity(strokes)
    mono_s = _x_mono_sub_score(mono, stats.median_x_mono)

    hysteresis = max(0.4, stats.median_height * 0.12 + 0.02 * char_count)
    extrema_count = _y_extrema(strokes, hysteresis)
    ext_pc = extrema_count / char_count
    ext_s = _extrema_sub_score(ext_pc)

    min_exp, max_exp = stats.expected_strokes(text)
    strk_s = _strokes_sub_score(len(strokes), min_exp, max_exp)

    w = bbox[2] - bbox[0]
    expected_w = max(stats.expected_width(text), char_count * BlockStats.MIN_CHAR_WIDTH)
    w_s = _width_sub_score(w, expected_w)

    h = bbox[3] - bbox[1]
    h_s = _height_sub_score(h, stats.median_height)

    core_scores = [arc_s, mono_s, ext_s, strk_s, w_s, h_s]
    core_weights = list(WEIGHTS.values())
    core_score = sum(s * wt for s, wt in zip(core_scores, core_weights))

    # --- V4 multiplicative penalties ---
    topo_s = _topology_analysis(text, strokes, bbox, stats.median_height)
    dens_s = _ink_density_score(strokes, bbox)

    final = core_score * topo_s * dens_s
    return max(0.0, min(1.0, final))


def score_chunk_detailed(
    text: str, strokes: list, bbox: tuple, stats: BlockStats
) -> tuple[float, dict]:
    """
    Same as score_chunk but also returns a dict of sub-score breakdown.
    Returns (final_score, subs_dict) where subs_dict includes both
    core sub-scores and penalty multipliers.
    """
    ALL_KEYS = list(WEIGHTS.keys()) + list(PENALTY_KEYS)
    if not strokes:
        return 0.0, {k: 0.0 for k in ALL_KEYS}

    char_count = max(1, len(text.strip()))

    arc = _arc_length(strokes)
    expected_arc = stats.expected_arc(text)
    arc_s = _arc_sub_score(arc, expected_arc)

    mono = _x_monotonicity(strokes)
    mono_s = _x_mono_sub_score(mono, stats.median_x_mono)

    hysteresis = max(0.4, stats.median_height * 0.12 + 0.02 * char_count)
    extrema_count = _y_extrema(strokes, hysteresis)
    ext_pc = extrema_count / char_count
    ext_s = _extrema_sub_score(ext_pc)

    min_exp, max_exp = stats.expected_strokes(text)
    strk_s = _strokes_sub_score(len(strokes), min_exp, max_exp)

    w = bbox[2] - bbox[0]
    expected_w = max(stats.expected_width(text), char_count * BlockStats.MIN_CHAR_WIDTH)
    w_s = _width_sub_score(w, expected_w)

    h = bbox[3] - bbox[1]
    h_s = _height_sub_score(h, stats.median_height)

    topo_s = _topology_analysis(text, strokes, bbox, stats.median_height)
    dens_s = _ink_density_score(strokes, bbox)

    # Build sub-scores dict (core + penalties)
    subs = {}
    for key, val in zip(WEIGHTS.keys(), [arc_s, mono_s, ext_s, strk_s, w_s, h_s]):
        subs[key] = val
    subs["topology"] = topo_s
    subs["density"] = dens_s

    # Core score (V3-identical)
    core_score = sum(subs[k] * WEIGHTS[k] for k in WEIGHTS)
    # Multiplicative penalties
    final = core_score * topo_s * dens_s
    final = max(0.0, min(1.0, final))
    return final, subs


def _score_reasons(text: str, strokes: list, bbox: tuple, stats: BlockStats) -> str:
    """Return a short human-readable string explaining score deductions."""
    if not strokes:
        return "empty output"

    _, subs = score_chunk_detailed(text, strokes, bbox, stats)

    reasons = []
    char_count = max(1, len(text.strip()))

    arc = _arc_length(strokes)
    expected_arc = stats.expected_arc(text)
    arc_ratio = arc / expected_arc if expected_arc > 0 else 1.0
    if subs["arc"] < 0.5:
        reasons.append(f"arc={arc_ratio:.2f}x")

    mono = _x_monotonicity(strokes)
    if subs["x_mono"] < 0.5:
        reasons.append(f"x_mono={mono:.2f}")

    hysteresis = max(0.4, stats.median_height * 0.12 + 0.02 * char_count)
    ext_pc = _y_extrema(strokes, hysteresis) / char_count
    if subs["extrema"] < 0.5:
        reasons.append(f"ext={ext_pc:.2f}/c")

    min_exp, max_exp = stats.expected_strokes(text)
    if subs["strokes"] < 0.5:
        reasons.append(f"strk={len(strokes)}(exp {min_exp:.0f}-{max_exp:.0f})")

    w = bbox[2] - bbox[0]
    expected_w = max(stats.expected_width(text), char_count * BlockStats.MIN_CHAR_WIDTH)
    w_ratio = w / expected_w if expected_w > 0 else 1.0
    if subs["width"] < 0.5:
        reasons.append(f"w={w_ratio:.1f}x")

    h = bbox[3] - bbox[1]
    h_ratio = h / stats.median_height
    if subs["height"] < 0.5:
        reasons.append(f"h={h_ratio:.1f}x")

    if subs["topology"] < 0.5:
        core = text.strip().lower()
        has_asc = any(c in ASCENDERS for c in core)
        has_desc = any(c in DESCENDERS for c in core)
        tag = "asc" if has_asc and not has_desc else "desc" if has_desc and not has_asc else "asc+desc"
        reasons.append(f"topo({tag})={subs['topology']:.2f}")

    if subs["density"] < 0.5:
        reasons.append(f"inkpool={subs['density']:.2f}")

    return ", ".join(reasons) if reasons else "ok"


# ---------------------------------------------------------------------------
# Main reviewer
# ---------------------------------------------------------------------------

class StrokeReviewer:
    """
    Text-aware quality reviewer (v4).

    Evaluates synthesized stroke_data and surgically retries bad chunks.
    Identical API to v3 StrokeReviewer for drop-in replacement.
    """

    def __init__(self, threshold: float = 0.60, max_retries: int = 3):
        self.threshold = max(0.0, min(1.0, threshold))
        self.max_retries = max(1, max_retries)

    def review_and_improve(
        self,
        engine,
        stroke_data: list,
        scale: float,
        stochastic: bool,
        status_cb=None,
        log_cb=None,
    ) -> list:
        """
        Review every chunk in stroke_data. For any chunk below self.threshold,
        re-synthesize up to self.max_retries times and keep the best attempt.

        Returns a new list of 4-tuples:
            (text, best_strokes, best_bbox, attempts_used)
        """

        def _log(msg: str):
            if log_cb:
                log_cb(msg)

        def _status(msg: str):
            if status_cb:
                status_cb(msg)

        ts = datetime.now().strftime("%H:%M:%S")

        # Build block-level statistics from the initial pass
        stats = BlockStats(stroke_data)

        _log("")
        _log(f"[OK] [{ts}] ════ AI Review Pass (v4 text-aware scorer) ════")
        _log(f"[OK]   Threshold   : {self.threshold:.2f}  │  Max retries: {self.max_retries}")
        _log(f"[OK]   Ref char-w  : {stats.median_char_width:.2f}mm  │  "
             f"Ref height: {stats.median_height:.2f}mm  │  "
             f"Ref x-mono: {stats.median_x_mono:.2f}")
        _log(f"[OK]   Core weights (V3): arc={WEIGHTS['arc']:.2f} x_mono={WEIGHTS['x_mono']:.2f} "
             f"ext={WEIGHTS['extrema']:.2f} strk={WEIGHTS['strokes']:.2f} "
             f"w={WEIGHTS['width']:.2f} h={WEIGHTS['height']:.2f}")
        _log(f"[OK]   Penalties (×mul): topology + density")

        # Parse 4-tuples (backward compat for 3-tuples)
        improved = []
        for item in stroke_data:
            text, strokes, bbox = item[0], item[1], item[2]
            retries = item[3] if len(item) > 3 else 0
            improved.append([text, strokes, bbox, retries])

        total = len(improved)

        # Score every chunk
        initial_scores: list = []
        bad_indices: list[int] = []
        for i, (text, strokes, bbox, _) in enumerate(improved):
            if text == '\n':
                initial_scores.append(None)
                continue
            s = score_chunk(text, strokes, bbox, stats)
            initial_scores.append(s)
            if s < self.threshold:
                bad_indices.append(i)

        good_count  = sum(1 for s in initial_scores if s is not None and s >= self.threshold)
        bad_count   = len(bad_indices)
        skip_count  = sum(1 for s in initial_scores if s is None)

        _log(f"[OK]   Chunks: {total} total │ {good_count} passed │"
             f" {bad_count} need retry │ {skip_count} skipped (newlines)")

        if not bad_indices:
            _log(f"[OK]   → All chunks passed! No retries needed.")
            _status(f"AI Review: all {total} chunks passed.")
            return [tuple(item) for item in improved]

        if not stochastic:
            _log(f"[!]   → {bad_count} chunk(s) below threshold, retries skipped (stochastic=False).")
            _status(f"AI Review: {bad_count} bad chunks, skipping retries (deterministic mode).")
            return [tuple(item) for item in improved]

        # Header table for bad chunks (expanded for v4)
        _log(f"[!]   → {bad_count} chunk(s) below threshold:")
        _log(f"[OK]")
        _log(f"[OK]   {'#':<4} {'Text':<22} {'Score':>6}  "
             f"{'arc':>5} {'x↔':>5} {'ext':>5} {'strk':>5} {'w':>5} {'h':>5} "
             f"{'topo':>5} {'dens':>5}  Reason")
        _log(f"[OK]   {'─'*4} {'─'*22} {'─'*6}  "
             f"{'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} "
             f"{'─'*5} {'─'*5}  {'─'*22}")

        for i in bad_indices:
            text, strokes, bbox, _ = improved[i]
            sc, subs = score_chunk_detailed(text, strokes, bbox, stats)
            reason = _score_reasons(text, strokes, bbox, stats)
            _log(
                f"[!]   {i:<4} {repr(text[:20]):<22} {sc:>5.2f}  "
                f"{subs['arc']:>5.2f} {subs['x_mono']:>5.2f} {subs['extrema']:>5.2f} "
                f"{subs['strokes']:>5.2f} {subs['width']:>5.2f} {subs['height']:>5.2f} "
                f"{subs['topology']:>5.2f} {subs['density']:>5.2f}  "
                f"{reason}"
            )

        _log(f"[OK]")
        _log(f"[OK]   Retrying...")

        improved_count = 0
        for rank, i in enumerate(bad_indices, start=1):
            text, strokes, bbox, _ = improved[i]
            best_strokes, best_bbox = strokes, bbox
            best_score = initial_scores[i]
            attempts_used = 0

            _status(f"AI Review: retrying chunk {rank}/{bad_count} \"{text[:20]}\"...")
            _log(f"[OK]")
            _log(f"[>>]   [{rank}/{bad_count}] \"{text[:25]}\"  (init score={best_score:.2f})")

            for attempt in range(1, self.max_retries + 1):
                attempts_used = attempt
                try:
                    new_strokes, new_bbox = engine.synthesize_word(
                        text=text, scale=scale, stochastic=stochastic
                    )
                except Exception as e:
                    _log(f"[!]     attempt {attempt}: ERROR - {e}")
                    continue

                new_score, new_subs = score_chunk_detailed(text, new_strokes, new_bbox, stats)
                delta = new_score - best_score
                delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                sub_str = (
                    f"arc={new_subs['arc']:.2f} x↔={new_subs['x_mono']:.2f} "
                    f"ext={new_subs['extrema']:.2f} topo={new_subs['topology']:.2f} "
                    f"dens={new_subs['density']:.2f}"
                )

                _log(f"[>>]     att{attempt}: {new_score:.2f} ({delta_str})  [{sub_str}]")

                if new_score > best_score:
                    best_score   = new_score
                    best_strokes = new_strokes
                    best_bbox    = new_bbox

                if best_score >= self.threshold:
                    break

            improved[i] = [text, best_strokes, best_bbox, attempts_used]
            if best_score >= self.threshold:
                improved_count += 1

            tag = "[✓]" if best_score >= self.threshold else "[✗]"
            _log(
                f"{tag}     → {'✓ PASS' if best_score >= self.threshold else '✗ BEST'}  "
                f"final={best_score:.2f}  "
                f"Δ={best_score - initial_scores[i]:+.2f}  "
                f"retries={attempts_used}"
            )

        _log(f"[OK]")
        _log(
            f"[OK]   Summary: {bad_count} retried │ "
            f"{improved_count}/{bad_count} crossed threshold │ "
            f"{bad_count - improved_count} stayed below"
        )
        _log(f"[OK] [{ts}] ════ Done ════")

        _status(
            f"AI Review done: {bad_count} retried, "
            f"{improved_count} improved, "
            f"{bad_count - improved_count} below threshold."
        )

        return [tuple(item) for item in improved]
