import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_track1_layout_corrected():
    """
    Correctly reconstructs the Track 1 tabletop configuration based on Track1Details.pdf.
    Fixed matplotlib syntax and improved coordinate logic.
    """
    # 1. Setup Figure (Corrected Syntax)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- Parameters from PDF (cm) ---
    # Sources:
    
    # Table dimensions
    # Total width is 60.0 cm. Depth is 60.0 cm.
    table_w = 60.0
    table_h = 60.0
    
    # Key Anchor: Front Camera Optical Center
    # X = 31.6 cm from left edge
    # Y = 26.0 cm from bottom edge
    cam_x = 31.6
    cam_y = 26.0
    
    # Grid Dimensions
    # Box Heights: 16.4 cm
    # Box Widths: Left=16.6, Mid=15.6, Right=16.6
    # Line Width (Black tape): 1.80 cm
    box_h = 16.4
    box_w_left = 16.6
    box_w_mid = 15.6
    box_w_right = 16.6
    line_w = 1.8
    
    # Robot Base Dimensions
    # Left Margin=6.4, Base=11.0, Gap=20.4, Base=11.0
    base_w = 11.0
    base_h = 15.0 # Estimated from visual proportion in Fig 2, not explicitly labeled height for base box
    gap_left = 6.4
    gap_mid = 20.4
    
    # --- Drawing Logic ---
    
    # 1. Draw Table Surface (0,0 is bottom-left)
    rect_table = patches.Rectangle((0, 0), table_w, table_h, 
                                   linewidth=2, edgecolor='black', facecolor='#f9f9f9', label='Table')
    ax.add_patch(rect_table)
    
    # 2. Draw Task Grid (Centered on Camera)
    # The diagram shows the camera center (red dot) is inside the middle box.
    # We assume the camera (31.6, 26.0) is the GEOMETRIC CENTER of the Middle Box.
    # This is the most standard calibration assumption.
    
    # Calculate Middle Box Coordinates
    mid_box_x_center = cam_x
    mid_box_y_center = cam_y
    
    # Middle Box (Inner Area)
    mid_box_x = mid_box_x_center - (box_w_mid / 2)
    mid_box_y = mid_box_y_center - (box_h / 2)
    
    # Left Box (Inner Area)
    # Position: Left of Middle Box - Line Width
    left_box_x = mid_box_x - line_w - box_w_left
    left_box_y = mid_box_y # Same height
    
    # Right Box (Inner Area)
    # Position: Right of Middle Box + Line Width
    right_box_x = mid_box_x + box_w_mid + line_w
    right_box_y = mid_box_y # Same height

    # Function to draw a "Grid Box" with thick borders
    def add_box(x, y, w, h, label):
        # Draw the black tape border (simulating 1.8cm width by drawing a larger black rect behind)
        border = patches.Rectangle((x - line_w/2, y - line_w/2), w + line_w, h + line_w,
                                   linewidth=0, facecolor='black')
        ax.add_patch(border)
        # Draw the inner white area
        inner = patches.Rectangle((x, y), w, h,
                                  linewidth=1, edgecolor='gray', facecolor='white')
        ax.add_patch(inner)
        # Add label
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, color='blue')

    add_box(left_box_x, left_box_y, box_w_left, box_h, f"Left\n{box_w_left}x{box_h}")
    add_box(mid_box_x, mid_box_y, box_w_mid, box_h, f"Middle\n{box_w_mid}x{box_h}")
    add_box(right_box_x, right_box_y, box_w_right, box_h, f"Right\n{box_w_right}x{box_h}")
    
    # 3. Draw Robot Bases (Bottom Row)
    # Note: The manual measurements in PDF (6.4+11+20.4+11+6.4 = 55.2) do not sum to 60.0.
    # This implies the labels are relative to local features, not absolute table width.
    # We will draw them using the explicit "6.4 cm from left" as the starting point.
    
    # Left Base
    base_l_x = gap_left
    rect_base_l = patches.Rectangle((base_l_x, 0), base_w, base_h, 
                                    linewidth=1, edgecolor='black', facecolor='#e0e0e0', hatch='///')
    ax.add_patch(rect_base_l)
    ax.text(base_l_x + base_w/2, base_h/2, "Left Base", ha='center', va='center', fontsize=8)
    
    # Right Base
    # Position: Left Base End + Gap 20.4
    base_r_x = base_l_x + base_w + gap_mid
    rect_base_r = patches.Rectangle((base_r_x, 0), base_w, base_h, 
                                    linewidth=1, edgecolor='black', facecolor='#e0e0e0', hatch='///')
    ax.add_patch(rect_base_r)
    ax.text(base_r_x + base_w/2, base_h/2, "Right Base", ha='center', va='center', fontsize=8)

    # 4. Draw Camera
    ax.plot(cam_x, cam_y, 'ro', markersize=8, label='Front Camera Optical Center')
    ax.text(cam_x + 1, cam_y + 1, f"({cam_x}, {cam_y})", color='red', fontsize=10)

    # 5. Settings
    ax.set_xlim(-5, 65)
    ax.set_ylim(-5, 65)
    ax.set_aspect('equal')
    ax.set_xlabel('Width (cm)')
    ax.set_ylabel('Depth (cm)')
    ax.set_title('Track 1 Tabletop Simulation Layout\n(Reconstructed from Figure 2)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    # Save automatically
    plt.savefig('track1_layout_corrected.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    draw_track1_layout_corrected()