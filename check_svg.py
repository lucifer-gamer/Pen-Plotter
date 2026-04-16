import numpy as np
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
import sys

def verify_svg(filepath):
    try:
        paths, attributes = svg2paths(filepath)
    except Exception as e:
        print(f"Error loading SVG: {e}")
        sys.exit(1)

    all_strokes = []
    
    print(f"Found {len(paths)} raw paths in {filepath}.")

    # ==========================================
    # 1. Extraction
    # ==========================================
    for path in paths:
        path_length = path.length()
        
        # Filter: If stroke is incredibly short (< 5 pixels), drop it
        if path_length < 5:
            continue
            
        num_samples = max(10, int(path_length / 2))
        
        stroke_points = []
        for i in range(num_samples):
            # t goes from 0.0 to 1.0
            t = i / float(num_samples - 1 if num_samples > 1 else 1)
            complex_point = path.point(t)
            # Extract real (X) and imaginary (Y) parts
            stroke_points.append((complex_point.real, complex_point.imag))
            
        all_strokes.append(stroke_points)

    if not all_strokes:
        print("No valid strokes found after extraction.")
        sys.exit(0)

    # ==========================================
    # 2. Filtering
    # ==========================================
    
    all_x = [p[0] for stroke in all_strokes for p in stroke]
    all_y = [p[1] for stroke in all_strokes for p in stroke]
    
    # Calculate median Y base line across all valid coordinates
    median_y = np.median(all_y)

    # Pass 1: Rogue Noise / Palm Rejection
    pass1_strokes = []
    for stroke in all_strokes:
        stroke_y = [p[1] for p in stroke]
        stroke_center_y = np.mean(stroke_y)
        
        # If unusually far from main cluster (> 300px diff), it's a palm drop
        if abs(stroke_center_y - median_y) > 300:
            print("Filtered: Rogue Palm/Noise touch drop")
            continue
            
        pass1_strokes.append(stroke)

    # Pass 2: Canvas Box Filtering
    # Recalculate global bounding box perfectly after rogue noise is gone
    pass1_x = [p[0] for stroke in pass1_strokes for p in stroke]
    global_min_x, global_max_x = min(pass1_x), max(pass1_x)
    global_width = global_max_x - global_min_x

    cleaned_strokes = []
    
    for stroke in pass1_strokes:
        stroke_x = [p[0] for p in stroke]
        stroke_width = max(stroke_x) - min(stroke_x)
        
        # If the stroke width is >= 90% of the total canvas width, it's the bounding frame
        if global_width > 0 and (stroke_width >= 0.90 * global_width):
            print("Filtered: Background Canvas Box drop")
            continue
            
        cleaned_strokes.append(stroke)

    print(f"Extracted {len(cleaned_strokes)} valid strokes after cleanup.")

    if not cleaned_strokes:
        print("All strokes were filtered out!")
        sys.exit(0)

    # ==========================================
    # 3. Plotting
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.title(f"Cleaned Path Data Verification: {filepath}")

    for idx, stroke in enumerate(cleaned_strokes):
        x_coords = [p[0] for p in stroke]
        # Invert the Y-axis values for matplotlib
        y_coords = [-p[1] for p in stroke]
        
        plt.plot(x_coords, y_coords, marker='.', markersize=4, linestyle='-', label=f'Stroke {idx+1}')

    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if len(cleaned_strokes) <= 10:
        plt.legend(loc='best', fontsize='small')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    svg_file = "001.svg"
    verify_svg(svg_file)
