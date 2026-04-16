# Plotter Studio v2.0 & Handwriting Synthesis Toolkit 
**Project Handover & Context File for AI Agents**

## 🎯 1. Project Overview & Objective
This repository contains **Plotter Studio**, a Python/PyQt6 graphical toolkit wrapping a PyTorch-based Recurrent Neural Network (RNN) designed for handwriting synthesis. 

The goal of the project is to convert digital text into physically plausible, human-like cursive handwriting toolpaths, specifically targeting hardware like **Pen Plotters (e.g., AxiDraw)** or custom CNC machines. Beyond simple generation, Plotter Studio features a high-fidelity digital canvas to arrange blocks of text (layout editor), adjust physical stroke properties (spacing, scale, bias), and enforce "organic" structures without typical robotic artifacts.

## 🚀 2. Current Progress & Completed Milestones
We have extensively rebuilt and stabilized the core application (`plotter_studio.py`):
- **GUI Reconstruction**: Perfectly restored the PyQt6 Plotter Studio v2.0 dark-mode UI. It features a complete left-hand control sidebar (Scale, Bias, Spacing sliders) and an interactive QGraphicsScene Canvas for visually manipulating draggable text blocks.
- **Asynchronous Pipeline**: The synthesis engine operates on a background `QThread` (`SynthesisWorker`), preventing UI freezing during heavy PyTorch sequence generation. 
- **AI Review & Heuristic Engine**: Integrated an automated post-generation review system. If the RNN generates "flat" or collapsed sequences (e.g., straight vertical lines or dots instead of cursive), the system aggressively penalizes the chunk based on geometric heuristics and silently re-generates it up to `N` times before rendering.
- **Tokenization**: We implemented spatial-aware tokenization to split words while retaining physical space structures and handling punctuation independently.

## ⚠️ 3. What We Are Lagging Behind On (TODO List)
The upcoming agent or developer needs to focus on the following missing functionality:
1. **Export Mechanisms (High Priority)**: 
   - The UI buttons for exporting are wired to placeholder functions (`_on_export`, `_on_export_svg`, `_on_export_gcode`). 
   - We need to implement the actual geometry conversion converting the flattened PyTorch output `(x, y)` arrays into valid SVG `<path>` elements and standard `G-Code` (`G0`, `G1`) instructions utilizing the hardware inputs (Draw Speed, Move Speed, Pen Down/Up).
2. **Session Persistence**: 
   - Projects cannot be accurately saved and loaded. We need to implement `_on_save_session` and `_on_load_session` to dump the canvas blocks (text, x/y position, scale, spacing values) into a JSON representation and recreate them on demand.
3. **Hardware Integration Debugging**:
   - The "Flip X/Y Axes" toggle in the GUI needs to be explicitly factored into the G-Code coordinate conversion math during export.

## 🧠 4. Architecture & Key Files
- `plotter_studio.py`: THE core file. Contains the entire PyQt6 application, including `PlotterStudio` (Main Window), `HandwritingBlock` (Canvas items), `SynthesisWorker` (Threads), and `ReviewPanel` (AI Debug Log).
- `synthesize.py` / `handwriting_synthesis/`: The underlying PyTorch model handlers.
- `.gitignore`: Configured to ignore our scratchpads, legacy patch scripts, and heavy `.pth` model checkpoints. If you see old files like `plotter_gui_v2.py` or `ui_extract.py`, they are deprecated redundant files.

## 💡 5. Agent Instructions
- **Do NOT** modify `plotter_studio.py` layout dimensions or visual styling without explicit permission. The user spent significant time perfecting the exact v2.0 GUI replica.
- Prioritize implementing the **Export to G-Code/SVG** functions seamlessly into the existing UI framework.
- Always check that your PyTorch/NumPy array logic strictly adheres to taking coordinates offset by `block.scenePos()`.
