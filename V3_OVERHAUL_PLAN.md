# Plotter Studio v3.0 — Full Overhaul Plan

## Overview

A comprehensive improvement of the entire pytorch-handwriting-synthesis-toolkit stack:
synthesizer engine, score/review pipeline, and the PyQt6 GUI. The goal is to
make every layer best-in-class without breaking the existing checkpoint format.

---

## User Review Required

> [!IMPORTANT]
> This plan touches **every major file** in the project. It is a significant
> rewrite. Review the scope carefully before approving execution.

> [!WARNING]
> The model/checkpoint format is **not changed**, so existing `checkpoints/Epoch_46`
> weights will continue to work. Backend API for `HandwritingSynthesizer.load()` is
> preserved.

---

## Proposed Changes

### 1 — Synthesis Engine Improvements (`plotter_studio.py` → `SynthesisEngine`)

#### [MODIFY] plotter_studio.py

**a) Adaptive step-count estimation**
Currently `estimated_steps = max(200, len(text) * 50)`. This over-allocates
for short chunks and under-allocates for long ones. Replace with a calibrated
formula:
```
estimated_steps = max(300, len(text) * 65 + 100)
```
This gives the attention window proper runway.

**b) Early-exit guard with min-step floor**
The model's `_is_end_of_string` can fire after only 20 steps on degenerate
sequences. Add a **minimum physical-step gate**: don't honour EOS until at
least `max(80, len(text) * 20)` steps have produced non-trivial output (arc
length > 0.5 per char). This requires a small wrapper around `sample_means`.

**c) Stroke density post-filter**
After baseline normalisation, apply a lightweight Douglas-Peucker simplification
to remove redundant collinear points (reduces G-code bloat by ~30 %) while
preserving curve fidelity.

**d) Pen-lift elision**
Merge consecutive strokes whose gap is < 0.3 mm into a single stroke, reducing
unnecessary pen-up/down cycles.

---

### 2 — Heuristic Scorer Improvements (`handwriting_synthesis/stroke_reviewer.py`)

#### [MODIFY] stroke_reviewer.py

**a) New metric: X-monotonicity ratio**
Legible cursive moves left-to-right. Measure the fraction of consecutive points
where `dx > 0`. A score below 0.55 (lots of backwards loops) indicates sequence
collapse; penalise −0.30.

**b) New metric: Stroke-count plausibility**
Expected strokes ≈ `max(1, char_count * 0.4)` to `char_count * 2.5`. Both
extremes (too few = collapsed, too many = explosion) should be penalised.

**c) Weighted scoring**
Replace the current flat-subtract approach with a weighted sum of sub-scores,
making penalisation more graduated and easier to tune in one place:

```
WEIGHTS = {
    "arc":         0.30,
    "x_mono":      0.20,
    "y_extrema":   0.25,
    "width":       0.15,
    "height":      0.10,
}
final_score = sum(sub * w for sub, w in zip(sub_scores, WEIGHTS.values()))
```

**d) Adaptive hysteresis**
Current hysteresis is `max(0.5, median_height * 0.15)`. Change to also scale
with text length: `max(0.4, median_height * 0.12 + 0.02 * char_count)`.

**e) Best-of-N selection with score logging**
Currently keeps track only of the best score. Extend to log a per-attempt
mini-table in the debug panel (already wired up, just needs richer output).

---

### 3 — GUI Redesign (`plotter_studio.py` — UI classes)

This is the most impactful visual change. The existing UI is functional but
visually plain. We will completely rebuild the styling to a **VS Code Dark+
inspired premium look** with glassmorphism accents, gradient buttons, and
animated status indicators.

#### [MODIFY] plotter_studio.py — UI layer

**a) Sidebar redesign**
- Replace the plain `QGroupBox` approach with custom-painted collapsible
  section headers (▶/▼ triangles) that animate open/close.
- Add a **style preset picker** at the top: "Neat / Normal / Loose / Sloppy"
  which applies a preset combination of bias + randomness + scale.
- Add a **Paper Size** combo (A4, Letter, A5) with scene auto-resize.
- Add a **Pen Pressure Curve** slider (maps to `S` value in G-code pen-down).

**b) Canvas enhancements**
- **Ruler overlays**: thin pixel rulers along the top and left edges (mm marks)
  that scale with zoom, like Illustrator.
- **Snap-to-grid**: optional 5-mm grid with dim dotted lines; blocks snap to
  grid when near a gridline (toggle in View menu).
- **Block thumbnails in panel**: a small sidebar list of all blocks with their
  first 20 chars and a ≡ drag handle to reorder Z-order.
- **Multi-select + group move**: Ctrl-click to select multiple blocks; arrow
  keys nudge selection by 1 mm.

**c) Status bar→ Progress bar**
Replace the `QLabel` status with a proper `QProgressBar` + spinning indicator
for synthesis in progress. During regeneration the progress bar fills chunk-by-
chunk as the reviewer emits callbacks.

**d) Menu bar**
Add a proper `QMenuBar`:
- **File**: New Canvas, Open (load `.pstudio` JSON), Save, Export G-Code, Export SVG, Export PNG Preview
- **Edit**: Select All, Deselect, Delete Selected
- **View**: Zoom In/Out/Fit, Toggle Grid, Toggle Rulers, Toggle Debug Log
- **Block**: Add Block, Regenerate Selected, Regenerate All

**e) Export improvements**
- **SVG export** of the canvas (uses current `svgwrite` in `utils.py`)
- **PNG preview export** (flat raster of the page)
- **Session save/load** (`.pstudio` JSON): serialise all block positions, text,
  spacing params so a session can be reopened later.

**f) Undo/Redo**
Add a simple command stack (`collections.deque`, max depth 20) that records
`(action_type, args)` tuples. Actions: add_block, delete_block, move_block,
edit_block, regen_block. `Ctrl-Z` / `Ctrl-Y` navigate the stack.

---

### 4 — Block-level improvements

#### [MODIFY] plotter_studio.py — `HandwritingBlock`

**a) Per-block style overrides**
Allow each block to have its own bias, scale, and randomness independent of
the global sliders. Expose these in the right-click context menu → "Block
Properties" dialog.

**b) Block resize handles**
Currently block width is only set globally. Add draggable right-edge resize
handle on each block so the block width (and thus word-wrap) can be adjusted
visually.

**c) Score overlay**
Optional: when debug mode is on, render a small semi-transparent score badge
on each rendered word chunk (colour-coded using the existing `RETRY_COLORS`).

---

### 5 — New: Background Worker Thread

Currently synthesis runs on the Qt main thread, freezing the UI.

#### [NEW] Worker thread in `plotter_studio.py`

Add a `QThread`-based `SynthesisWorker` that:
- Accepts `(block, engine, params)` in a queue
- Emits `progress(int)`, `chunk_ready(str)`, `finished(block)`, `error(str)`
- The main thread connects to these signals to update the progress bar and log
  panel without freezing.

This is the single biggest UX improvement possible.

---

### 6 — New: SVG Import / Trace

#### [NEW] `svg_import.py`

A lightweight parser that reads an SVG file produced by the toolkit's own
`create_strokes_svg` output and converts paths back to stroke lists. Useful
for re-importing previously exported layouts and repositioning them.

(Stretch goal — implement only if time allows after the above.)

---

### 7 — Debug Panel Improvements

#### [MODIFY] `DebugLogPanel`

- **Colour-coded log lines**: PASS lines in green, FAIL lines in red, header
  lines in blue — using `QTextCharFormat` and HTML rich text.
- **Expandable chunk rows**: clicking a chunk row in the log expands it to show
  the full sub-score breakdown table.
- **Stats summary card** at the top: total chunks, % passed first try, avg
  retries, avg final score — updated live.
- **Copy log to clipboard** button.

---

## Verification Plan

### Automated Tests
- Run `python plotter_studio.py` and confirm no import errors.
- Synthesise a multi-paragraph text block and confirm the debug panel shows
  rich coloured output.
- Export to G-Code and verify the file opens correctly in a G-Code viewer.
- Export SVG and PNG preview.
- Test session save and reload.

### Manual Verification
- Visually inspect the new sidebar (collapsible groups, preset picker, paper
  size combo).
- Verify rulers and snap-to-grid work at various zoom levels.
- Test undo/redo across add, delete, move, and regenerate operations.
- Verify the synthesis worker thread keeps the UI responsive during generation.

---

## Implementation Order

| Priority | Area | Effort |
|----------|------|--------|
| 1 | Background worker thread | Medium |
| 2 | GUI redesign (sidebar, menu, status bar) | High |
| 3 | Scorer improvements (new metrics, weighted scoring) | Low |
| 4 | Engine improvements (step count, DP simplification) | Low |
| 5 | Block resize handles + per-block props | Medium |
| 6 | Session save/load + SVG/PNG export | Medium |
| 7 | Debug panel rich text | Low |
| 8 | Undo/redo | Medium |
| 9 | SVG import/trace | Low (stretch) |

---

## Open Questions

> [!IMPORTANT]
> **Q1** — Should the background worker thread be a single global worker (queue-
> based) or a per-block QThread? Queue-based is simpler but means blocks are
> synthesised sequentially; per-block allows parallelism but risks GPU memory
> contention (not an issue on CPU-only, which is the current setup).

> [!IMPORTANT]
> **Q2** — For session save/load (`.pstudio`), should synthesised stroke data be
> embedded in the JSON (large file, instant reload) or regenerated on open
> (small file, requires model at load time)?

> [!NOTE]
> **Q3** — The paper-size change (A4 → Letter) changes `A4_WIDTH_MM /
> A4_HEIGHT_MM` constants. Existing sessions saved with A4 dimensions would
> need migration. Is this acceptable?
