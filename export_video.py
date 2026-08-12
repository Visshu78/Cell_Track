"""
Export Cell Tracking Dataset to Animated Video (GIF/HTML Interactive Player).
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from data_loader import load_masks
import matplotlib.pyplot as plt


def generate_distinct_colors(num_colors: int = 256):
    """
    Generates RGB color palette where background (0) is dark navy/black and cells have distinct vibrant colors.
    """
    np.random.seed(42)
    colors = {}
    colors[0] = (15, 23, 42)  # Dark navy background (#0f172a)
    
    # Generate visually distinct vibrant HSL/RGB colors
    for i in range(1, num_colors + 1):
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio separation
        sat = 0.8 + np.random.uniform(-0.1, 0.1)
        val = 0.9 + np.random.uniform(-0.1, 0.1)
        
        # Convert HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
        
    return colors


def render_frame(
    mask_frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    colors_dict: dict,
    history_centroids: dict = None,
    scale_factor: float = 0.5
) -> Image.Image:
    """
    Renders a single video frame with color-coded cell masks, centroid markers, cell IDs, and motion trail histories.
    """
    h, w = mask_frame.shape
    out_w, out_h = int(w * scale_factor), int(h * scale_factor)

    # Downsample mask frame if needed for crisp rendering speed
    if scale_factor != 1.0:
        img_mask = Image.fromarray(mask_frame.astype(np.int32))
        img_mask = img_mask.resize((out_w, out_h), Image.NEAREST)
        scaled_mask = np.array(img_mask)
    else:
        scaled_mask = mask_frame

    # Create RGB canvas
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    unique_ids = np.unique(scaled_mask)
    
    for cell_id in unique_ids:
        if cell_id == 0:
            rgb[scaled_mask == 0] = colors_dict.get(0, (15, 23, 42))
        else:
            color = colors_dict.get(int(cell_id), (56, 189, 248))
            rgb[scaled_mask == cell_id] = color

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    # Overlay Cell Info, Centroids, and History Trails
    current_centroids = {}
    for cell_id in unique_ids:
        if cell_id == 0:
            continue
        coords = np.argwhere(scaled_mask == cell_id)
        if len(coords) == 0:
            continue
        cy, cx = coords.mean(axis=0)
        current_centroids[int(cell_id)] = (cx, cy)
        
        # Add centroid point
        r = 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255), outline=(0, 0, 0))
        
        # Add Label text
        draw.text((cx + 5, cy - 5), f"ID:{cell_id}", fill=(255, 255, 255))

    # Update history trails if provided
    if history_centroids is not None:
        history_centroids[frame_idx] = current_centroids
        # Draw trails for last 8 frames
        for f in range(max(0, frame_idx - 8), frame_idx):
            if f in history_centroids:
                for c_id, (cx, cy) in current_centroids.items():
                    if c_id in history_centroids[f]:
                        prev_cx, prev_cy = history_centroids[f][c_id]
                        draw.line([(prev_cx, prev_cy), (cx, cy)], fill=(255, 255, 255, 128), width=2)

    # Frame header banner
    banner_height = 36
    header_img = Image.new("RGB", (out_w, banner_height), (30, 41, 59))
    header_draw = ImageDraw.Draw(header_img)
    header_draw.text((12, 10), f"Cell Tracking Video | Frame {frame_idx + 1}/{total_frames} | Active Cells: {len(current_centroids)}", fill=(241, 245, 249))
    
    # Combine header and frame
    combined = Image.new("RGB", (out_w, out_h + banner_height))
    combined.paste(header_img, (0, 0))
    combined.paste(img, (0, banner_height))

    return combined


def create_cell_tracking_video(
    masks: np.ndarray,
    output_gif: str = "cell_tracking_video.gif",
    fps: int = 5
) -> Path:
    """
    Renders 3D mask array into an animated GIF video.
    """
    total_frames = masks.shape[0]
    unique_all = np.unique(masks)
    max_id = int(unique_all.max()) if len(unique_all) > 0 else 100
    colors_dict = generate_distinct_colors(max_id + 50)

    print(f"[ExportVideo] Rendering {total_frames} frames to animated GIF ({output_gif})...")
    frames = []
    history_centroids = {}
    
    for t in range(total_frames):
        frame_img = render_frame(
            masks[t],
            frame_idx=t,
            total_frames=total_frames,
            colors_dict=colors_dict,
            history_centroids=history_centroids,
            scale_factor=0.5
        )
        frames.append(frame_img)

    duration_ms = int(1000 / fps)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0
    )
    print(f"[ExportVideo] Successfully exported animated video to: {Path(output_gif).resolve()}")
    return Path(output_gif)


if __name__ == "__main__":
    masks = load_masks()
    create_cell_tracking_video(masks)
