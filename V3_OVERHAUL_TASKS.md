# Plotter Studio v3.0 — Task Checklist

## Phase 1 — Core Engine & Scorer
- [x] **Scorer**: Weighted scoring, X-monotonicity, stroke-count plausibility, adaptive hysteresis
- [/] **Engine**: Adaptive step count, Douglas-Peucker simplification, pen-lift elision

## Phase 2 — Background Worker Thread
- [/] `SynthesisWorker` QThread with progress/chunk_ready/finished/error signals
- [/] Wire progress bar + status to worker signals
- [ ] Disable controls during synthesis, re-enable on finish

## Phase 3 — GUI Redesign
- [/] Menu bar (File/Edit/View/Block)
- [/] Collapsible sidebar sections
- [/] Style preset picker (Neat/Normal/Loose/Sloppy)
- [/] Paper size picker (A4/Letter/A5)
- [/] Progress bar replacing status label
- [ ] Ruler overlays (mm scale, zoom-aware)
- [ ] Snap-to-grid (5mm, dotted lines, toggle)

## Phase 4 — Block Improvements
- [ ] Per-block style override (right-click → Block Properties dialog)
- [ ] Block resize handle (right-edge drag)

## Phase 5 — Export & Session
- [/] SVG canvas export
- [/] PNG preview export
- [/] Session save/load (.pstudio JSON with embedded strokes)

## Phase 6 — Undo/Redo
- [ ] Command stack (add/delete/move/edit/regen)
- [ ] Ctrl-Z / Ctrl-Y hotkeys

## Phase 7 — Debug Panel
- [ ] Rich coloured log lines (green PASS, red FAIL, blue headers)
- [ ] Live stats summary card
- [ ] Copy-to-clipboard button
