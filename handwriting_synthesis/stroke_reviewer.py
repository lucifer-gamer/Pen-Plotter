"""
stroke_reviewer.py  (v3 — weighted sub-score scoring engine)
=============================================================
Offline, heuristic quality-scoring engine for synthesized handwriting chunks.

After the RNN produces stroke data for each word/chunk, this module:
  1. Scores each chunk's geometric quality on 6 independent sub-scores,
     combining them with calibrated weights into a [0, 1] final score.
  2. Surgically re-synthesizes any chunk scoring below a threshold.
  3. Keeps the best (highest-scoring) attempt out of N retries and logs
     a detailed breakdown table to the caller's log panel.

No external API calls, no GPU needed — runs in milliseconds per chunk.

Sub-score weights (must sum to 1.0):
  arc_score    0.28  — stroke arc-length relative to expectation
  x_mono_score 0.18  — left-to-right monotonicity (cursive direction check)
  extrema_score 0.24  — vertical direction changes per char (legibility)
  stroke_score 0.10  — stroke-count plausibility
  width_score  0.12  — bounding-box width sanity
  height_score 0.08  — bounding-box height sanity

Data format:
  stroke_data items are 4-tuples:  (text, strokes, bbox, retries_used)
  retries_used starts at 0 from generate(); reviewer sets it to the actual
  number of synthesis attempts performed for each bad chunk.

Log format emitted via log_cb:
  Rich structured strings; callers can display them in QPlainTextEdit.
  Lines are tagged with prefix symbols so the GUI can colour them:
    [OK]  — passed, or header info
    [>>]  — retry attempt
    [✓]  — chunk crossed threshold
    [✗]  — chunk stayed below threshold
    [!]  — warning / anomaly
"""

from __future__ import annotations
import math
import statistics
from datetime import datetime


# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "arc":     0.28,
    "x_mono":  0.18,
    "extrema": 0.24,
    "strokes": 0.10,
    "width":   0.12,
    "height":  0.08,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Internal helpers
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


def _avg_points_per_stroke(strokes: list) -> float:
    """Average number of points per stroke (dot/collapse detector)."""
    if not strokes:
        return 0.0
    return sum(len(s) for s in strokes) / len(strokes)


def _total_points(strokes: list) -> int:
    """Total number of points across all strokes."""
    return sum(len(s) for s in strokes)


def _y_extrema(strokes: list, hysteresis: float = 0.5) -> int:
    """
    Count the number of vertical direction changes in the stroke path.
    Legible handwriting typically requires > 1.2 Y-extrema per character.
    Below 1.0 indicates cursive letters are unformed or rushed into wavy lines.
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
    Legible cursive: mostly left-to-right with limited backtracking.
    Expected: 0.55 – 0.85 (too high = unnatural; too low = collapses/reversals).
    """
    forward = 0
    total = 0
    for stroke in strokes:
        for (x1, _), (x2, _) in zip(stroke, stroke[1:]):
            total += 1
            if x2 >= x1:
                forward += 1
    if total == 0:
        return 0.5  # neutral
    return forward / total


# ---------------------------------------------------------------------------
# Block-level reference statistics
# ---------------------------------------------------------------------------

class BlockStats:
    """
    Captures median width-per-character and median height from an initial
    synthesis pass so the reviewer has a reference baseline.

    Precision fix: only includes chunks with at least 3 strokes and non-trivial
    arc length (>= 1.0 scene units per character) so bad first-pass chunks
    don't poison the reference distribution.
    """

    # Hard-coded absolute fallbacks in scene units (≈ mm at default scale)
    MIN_ARC_PER_CHAR: float = 1.2    # minimum plausible arc per character
    MIN_CHAR_WIDTH:   float = 1.5    # minimum plausible width per character
    MIN_HEIGHT:       float = 2.5    # minimum plausible line height

    def __init__(self, stroke_data: list):
        """
        stroke_data: list of (text, strokes, bbox[, retries]) as produced by
        HandwritingBlock.generate().
        Skips newlines, empty strokes, and statistically suspicious chunks.
        """
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

            # Skip obviously bad chunks from polluting the reference baseline
            if arc < char_count * self.MIN_ARC_PER_CHAR * 0.3:
                continue
            if len(strokes) < max(1, char_count // 4):
                continue

            if w > 0:
                widths_per_char.append(w / char_count)
            if h > 0:
                heights.append(h)

            mono = _x_monotonicity(strokes)
            # Only include plausible (non-degenerate) chunks in reference
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

        # Reference X-monotonicity from clean chunks (clamped to plausible range)
        self.median_x_mono: float = (
            statistics.median(x_monos)
            if len(x_monos) >= 2
            else 0.68
        )
        self.median_x_mono = max(0.55, min(0.85, self.median_x_mono))

    def expected_width(self, text: str) -> float:
        char_count = max(1, len(text.strip()))
        return char_count * self.median_char_width

    def expected_arc(self, text: str) -> float:
        char_count = max(1, len(text.strip()))
        stats_arc = self.expected_width(text) * 2.5
        absolute_floor = char_count * self.MIN_ARC_PER_CHAR
        return max(stats_arc, absolute_floor)

    def expected_strokes(self, text: str) -> tuple[float, float]:
        """Return (min_expected, max_expected) stroke count for a chunk."""
        char_count = max(1, len(text.strip()))
        return (max(1, char_count * 0.35), char_count * 3.0)


# ---------------------------------------------------------------------------
# Per-chunk sub-score functions (each returns 0.0–1.0)
# ---------------------------------------------------------------------------

def _arc_sub_score(arc: float, expected_arc: float) -> float:
    """Arc-length sub-score: 1.0 = on target, 0.0 = collapsed."""
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
    """X-monotonicity sub-score: penalise both too-low (collapse) and too-high (unnatural)."""
    # Ideal window: ref ± 0.12
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
    # mono > hi (too directional — may be flat line)
    gap = mono - hi
    if gap < 0.08:
        return 0.80
    if gap < 0.15:
        return 0.45
    return 0.10


def _extrema_sub_score(extrema_per_char: float) -> float:
    """Y-extrema legibility sub-score."""
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
    """Stroke-count plausibility sub-score."""
    if min_exp <= n_strokes <= max_exp:
        return 1.0
    if n_strokes < min_exp:
        ratio = n_strokes / min_exp
        if ratio >= 0.7:
            return 0.60
        if ratio >= 0.4:
            return 0.25
        return 0.0
    # too many strokes
    ratio = max_exp / n_strokes
    if ratio >= 0.6:
        return 0.70
    if ratio >= 0.35:
        return 0.30
    return 0.05


def _width_sub_score(w: float, expected_w: float) -> float:
    """Width-ratio sub-score."""
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
    # too narrow
    if ratio >= 0.60:
        return 0.65
    if ratio >= 0.35:
        return 0.20
    return 0.0


def _height_sub_score(h: float, median_height: float) -> float:
    """Height-ratio sub-score."""
    if median_height <= 0:
        return 1.0
    ratio = h / median_height
    if 0.55 <= ratio <= 2.0:
        return 1.0
    if ratio > 2.0:
        if ratio < 3.0:
            return 0.50
        return 0.0
    # too short
    if ratio >= 0.35:
        return 0.45
    return 0.0


# ---------------------------------------------------------------------------
# Aggregate chunk score
# ---------------------------------------------------------------------------

def score_chunk(text: str, strokes: list, bbox: tuple, stats: BlockStats) -> float:
    """
    Return a quality score in [0.0, 1.0] for a single synthesized chunk.
    Uses 6 weighted sub-scores for robustness and tunability.
    """
    if not strokes:
        return 0.0

    char_count = max(1, len(text.strip()))

    # 1. Arc length
    arc = _arc_length(strokes)
    expected_arc = stats.expected_arc(text)
    arc_s = _arc_sub_score(arc, expected_arc)

    # 2. X-monotonicity
    mono = _x_monotonicity(strokes)
    mono_s = _x_mono_sub_score(mono, stats.median_x_mono)

    # 3. Y-extrema (adaptive hysteresis)
    hysteresis = max(0.4, stats.median_height * 0.12 + 0.02 * char_count)
    extrema_count = _y_extrema(strokes, hysteresis)
    ext_pc = extrema_count / char_count
    ext_s = _extrema_sub_score(ext_pc)

    # 4. Stroke count
    min_exp, max_exp = stats.expected_strokes(text)
    strk_s = _strokes_sub_score(len(strokes), min_exp, max_exp)

    # 5. Width
    w = bbox[2] - bbox[0]
    expected_w = max(stats.expected_width(text), char_count * BlockStats.MIN_CHAR_WIDTH)
    w_s = _width_sub_score(w, expected_w)

    # 6. Height
    h = bbox[3] - bbox[1]
    h_s = _height_sub_score(h, stats.median_height)

    sub_scores = [arc_s, mono_s, ext_s, strk_s, w_s, h_s]
    weights    = list(WEIGHTS.values())

    final = sum(s * w for s, w in zip(sub_scores, weights))
    return max(0.0, min(1.0, final))


def score_chunk_detailed(
    text: str, strokes: list, bbox: tuple, stats: BlockStats
) -> tuple[float, dict]:
    """
    Same as score_chunk but also returns a dict of sub-score breakdown
    for display in the debug panel.
    Returns (final_score, sub_scores_dict).
    """
    if not strokes:
        return 0.0, {k: 0.0 for k in WEIGHTS}

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

    subs = dict(zip(WEIGHTS.keys(), [arc_s, mono_s, ext_s, strk_s, w_s, h_s]))
    weights = list(WEIGHTS.values())
    final = sum(s * wt for s, wt in zip(subs.values(), weights))
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

    return ", ".join(reasons) if reasons else "ok"


# ---------------------------------------------------------------------------
# Main reviewer
# ---------------------------------------------------------------------------

class StrokeReviewer:
    """
    Evaluates synthesized stroke_data and surgically retries bad chunks.

    Input / output format:
      stroke_data is a list of 4-tuples:  (text, strokes, bbox, retries_used)
      The 4th element is 0 on entry (from generate()). On exit it is set to
      the actual number of synthesis attempts that were tried for each chunk.
      Chunks that passed first-time keep retries_used == 0.

    Usage::

        reviewer = StrokeReviewer(threshold=0.60, max_retries=3)
        stroke_data = reviewer.review_and_improve(
            engine, stroke_data, scale, stochastic,
            status_cb=..., log_cb=...,
        )
        # Each item is now (text, best_strokes, best_bbox, attempts_used)
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
        where attempts_used == 0 means the chunk passed on first try.

        status_cb(msg): one-liner status bar update
        log_cb(msg):    persistent log panel append — gets detailed per-chunk info
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
        _log(f"[OK] [{ts}] ════ AI Review Pass (v3 weighted scorer) ════")
        _log(f"[OK]   Threshold   : {self.threshold:.2f}  │  Max retries: {self.max_retries}")
        _log(f"[OK]   Ref char-w  : {stats.median_char_width:.2f}mm  │  "
             f"Ref height: {stats.median_height:.2f}mm  │  "
             f"Ref x-mono: {stats.median_x_mono:.2f}")
        _log(f"[OK]   Weights: arc={WEIGHTS['arc']:.2f} x_mono={WEIGHTS['x_mono']:.2f} "
             f"ext={WEIGHTS['extrema']:.2f} strk={WEIGHTS['strokes']:.2f} "
             f"w={WEIGHTS['width']:.2f} h={WEIGHTS['height']:.2f}")

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

        # Header table for bad chunks
        _log(f"[!]   → {bad_count} chunk(s) below threshold:")
        _log(f"[OK]")
        _log(f"[OK]   {'#':<4} {'Text':<22} {'Score':>6}  {'arc':>5} {'x↔':>5} {'ext':>5} {'strk':>5} {'w':>5} {'h':>5}  Reason")
        _log(f"[OK]   {'─'*4} {'─'*22} {'─'*6}  {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}  {'─'*22}")

        for i in bad_indices:
            text, strokes, bbox, _ = improved[i]
            sc, subs = score_chunk_detailed(text, strokes, bbox, stats)
            reason = _score_reasons(text, strokes, bbox, stats)
            _log(
                f"[!]   {i:<4} {repr(text[:20]):<22} {sc:>5.2f}  "
                f"{subs['arc']:>5.2f} {subs['x_mono']:>5.2f} {subs['extrema']:>5.2f} "
                f"{subs['strokes']:>5.2f} {subs['width']:>5.2f} {subs['height']:>5.2f}  "
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

            attempts_log = []
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
                    f"ext={new_subs['extrema']:.2f} strk={new_subs['strokes']:.2f}"
                )
                attempts_log.append(
                    f"att{attempt}={new_score:.2f}({delta_str})"
                )

                _log(f"[>>]     att{attempt}: {new_score:.2f} ({delta_str})  [{sub_str}]")

                if new_score > best_score:
                    best_score   = new_score
                    best_strokes = new_strokes
                    best_bbox    = new_bbox

                if best_score >= self.threshold:
                    break

            outcome = "✓ PASS" if best_score >= self.threshold else "✗ BEST"
            improved[i] = [text, best_strokes, best_bbox, attempts_used]
            if best_score >= self.threshold:
                improved_count += 1

            tag = "[✓]" if best_score >= self.threshold else "[✗]"
            _log(
                f"{tag}     → {outcome}  final={best_score:.2f}  "
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
