"""
Web Visualizer Generator: Converts NPZ dataset into a self-contained interactive web video player.
"""

import base64
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import colorsys
from data_loader import load_masks


def get_color_map(max_id=100):
    colors = {0: (15, 23, 42)}  # Dark navy slate
    for i in range(1, max_id + 50):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def render_synthetic_microscopy(mask_frame: np.ndarray, scale: float = 0.5) -> Image.Image:
    """
    Generates a realistic synthetic fluorescence cell microscopy image 
    from the binary/label mask so the user can visually see 'cells' like in a microscope.
    """
    if scale != 1.0:
        img_m = Image.fromarray(mask_frame.astype(np.int32))
        img_m = img_m.resize((int(mask_frame.shape[1] * scale), int(mask_frame.shape[0] * scale)), Image.NEAREST)
        mask_frame = np.array(img_m)

    h, w = mask_frame.shape
    np.random.seed(123)
    bg = np.random.normal(15, 3, (h, w)).clip(0, 255).astype(np.float32)
    
    cell_intensity = np.zeros((h, w), dtype=np.float32)
    unique_ids = np.unique(mask_frame)
    
    for c_id in unique_ids:
        if c_id == 0:
            continue
        mask = (mask_frame == c_id)
        if not np.any(mask):
            continue
        
        # Calculate cell bounding box slice for fast localized computation
        ys_idx, xs_idx = np.where(mask)
        cy, cx = ys_idx.mean(), xs_idx.mean()
        
        min_y, max_y = max(0, int(ys_idx.min())), min(h, int(ys_idx.max()) + 1)
        min_x, max_x = max(0, int(xs_idx.min())), min(w, int(xs_idx.max()) + 1)
        
        ys_sub, xs_sub = np.ogrid[min_y:max_y, min_x:max_x]
        dist_sq = (xs_sub - cx)**2 + (ys_sub - cy)**2
        radius = np.sqrt(mask.sum() / np.pi) + 1.0
        
        intensity_sub = np.exp(- (dist_sq / (radius * 0.95)**2)) * 220
        cell_intensity[min_y:max_y, min_x:max_x][mask[min_y:max_y, min_x:max_x]] = intensity_sub[mask[min_y:max_y, min_x:max_x]]

    img_data = (bg + cell_intensity).clip(0, 255).astype(np.uint8)
    
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = (img_data * 0.15).astype(np.uint8)
    rgb[:, :, 1] = (img_data * 0.85).astype(np.uint8)
    rgb[:, :, 2] = (img_data * 0.95).astype(np.uint8)
    
    return Image.fromarray(rgb)


def render_mask_color(mask_frame: np.ndarray, colors_dict: dict, scale: float = 0.5) -> Image.Image:
    if scale != 1.0:
        img_m = Image.fromarray(mask_frame.astype(np.int32))
        img_m = img_m.resize((int(mask_frame.shape[1] * scale), int(mask_frame.shape[0] * scale)), Image.NEAREST)
        mask_frame = np.array(img_m)

    h, w = mask_frame.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(mask_frame)
    for c_id in unique_ids:
        rgb[mask_frame == c_id] = colors_dict.get(int(c_id), (56, 189, 248))
    return Image.fromarray(rgb)


def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def build_web_visualizer():
    print("[WebVisualizer] Loading dataset masks...")
    masks = load_masks()
    total_frames = len(masks)
    colors_dict = get_color_map(int(masks.max()))
    
    mask_b64_list = []
    synth_b64_list = []
    frame_stats = []

    print(f"[WebVisualizer] Encoding {total_frames} video frames to base64...")
    for t in range(total_frames):
        m_img = render_mask_color(masks[t], colors_dict)
        s_img = render_synthetic_microscopy(masks[t])
        
        mask_b64_list.append(image_to_base64(m_img))
        synth_b64_list.append(image_to_base64(s_img))
        
        # Calculate cell centroids & areas for frame stats
        c_stats = []
        u_ids = [int(i) for i in np.unique(masks[t]) if i != 0]
        for c_id in u_ids:
            coords = np.argwhere(masks[t] == c_id)
            if len(coords) > 0:
                cy, cx = coords.mean(axis=0)
                c_stats.append({
                    "id": c_id,
                    "x": round(float(cx * 0.5), 1),
                    "y": round(float(cy * 0.5), 1),
                    "area": int(len(coords))
                })
        frame_stats.append({
            "frame": t,
            "count": len(u_ids),
            "cells": c_stats
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cell Track - Time-Lapse Video Player & Analytics</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Outfit', sans-serif; background-color: #0b0f19; color: #f1f5f9; }}
        .glass-panel {{ background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }}
        .slider-thumb::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #38bdf8;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
        }}
    </style>
</head>
<body class="min-h-screen p-4 md:p-8 flex flex-col">
    <!-- Header -->
    <header class="max-w-7xl w-full mx-auto mb-6 flex flex-wrap items-center justify-between gap-4 glass-panel p-6 rounded-2xl">
        <div class="flex items-center gap-3">
            <div class="p-3 bg-cyan-500/20 text-cyan-400 rounded-xl border border-cyan-500/30">
                🔬
            </div>
            <div>
                <h1 class="text-2xl font-bold text-white tracking-tight">Cell Tracking Time-Lapse Video Visualizer</h1>
                <p class="text-xs text-slate-400">Interactive frame-by-frame analysis, synthetic microscopy render & cell motion trajectories</p>
            </div>
        </div>
        <div class="flex items-center gap-3 text-sm">
            <span class="px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                ● Live Video Stream
            </span>
            <span class="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 border border-slate-700">
                30 Frames (1024x1024)
            </span>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl w-full mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">
        <!-- Video Player Section (2 cols) -->
        <div class="lg:col-span-2 flex flex-col gap-4 glass-panel p-6 rounded-2xl">
            <!-- View Mode Switcher -->
            <div class="flex items-center justify-between border-b border-slate-700/50 pb-4">
                <div class="flex items-center gap-2">
                    <button id="btn-mode-synth" onclick="setMode('synth')" class="px-4 py-2 text-xs font-semibold rounded-lg bg-cyan-500 text-slate-950 transition hover:bg-cyan-400 shadow-lg shadow-cyan-500/20">
                        Synthetic Fluorescence Video
                    </button>
                    <button id="btn-mode-mask" onclick="setMode('mask')" class="px-4 py-2 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 transition">
                        Segmentation Mask Labels
                    </button>
                </div>
                <div class="flex items-center gap-2 text-xs text-slate-400">
                    <label class="cursor-pointer flex items-center gap-2">
                        <input type="checkbox" id="chk-trails" checked onchange="drawFrame()" class="rounded bg-slate-800 border-slate-700 text-cyan-500">
                        Motion Trails
                    </label>
                    <label class="cursor-pointer flex items-center gap-2 ml-3">
                        <input type="checkbox" id="chk-labels" checked onchange="drawFrame()" class="rounded bg-slate-800 border-slate-700 text-cyan-500">
                        Cell IDs
                    </label>
                </div>
            </div>

            <!-- Canvas Video Viewer Container -->
            <div class="relative w-full aspect-square max-h-[512px] mx-auto bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
                <canvas id="videoCanvas" width="512" height="512" class="w-full h-full object-contain"></canvas>
                
                <!-- Frame HUD Overlay -->
                <div class="absolute top-3 left-3 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700/50 text-xs font-mono text-cyan-300">
                    Frame: <span id="hud-frame">0</span>/29 | Cells: <span id="hud-count">0</span>
                </div>
            </div>

            <!-- Player Controls -->
            <div class="flex flex-col gap-3 pt-2">
                <!-- Timeline Slider -->
                <div class="flex items-center gap-3">
                    <span class="text-xs font-mono text-slate-400 min-w-[36px]" id="time-curr">00:00</span>
                    <input type="range" id="timeline" min="0" max="{total_frames - 1}" value="0" step="1" oninput="onSeek(this.value)" class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer slider-thumb">
                    <span class="text-xs font-mono text-slate-400 min-w-[36px]">00:30</span>
                </div>

                <!-- Play/Pause & Speed Buttons -->
                <div class="flex items-center justify-between flex-wrap gap-3">
                    <div class="flex items-center gap-2">
                        <button onclick="prevFrame()" class="p-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition">
                            ⏮️
                        </button>
                        <button id="btn-play" onclick="togglePlay()" class="px-5 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold transition flex items-center gap-2">
                            <span>▶</span> Play Video
                        </button>
                        <button onclick="nextFrame()" class="p-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition">
                            ⏭️
                        </button>
                    </div>

                    <div class="flex items-center gap-2 text-xs text-slate-400">
                        <span>Speed:</span>
                        <button onclick="setSpeed(0.5)" class="spd-btn px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700">0.5x</button>
                        <button onclick="setSpeed(1.0)" class="spd-btn px-2.5 py-1 rounded bg-cyan-500 text-slate-950 font-bold">1.0x</button>
                        <button onclick="setSpeed(2.0)" class="spd-btn px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700">2.0x</button>
                        <button onclick="setSpeed(4.0)" class="spd-btn px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700">4.0x</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Sidebar Analytics & Cell Inspector -->
        <div class="flex flex-col gap-6">
            <!-- Dataset Info Card -->
            <div class="glass-panel p-6 rounded-2xl">
                <h3 class="text-lg font-bold text-white mb-3 flex items-center gap-2">
                    📊 Dataset Overview
                </h3>
                <div class="space-y-3 text-xs text-slate-300">
                    <div class="flex justify-between py-1.5 border-b border-slate-700/50">
                        <span class="text-slate-400">Data File:</span>
                        <span class="font-mono text-cyan-400">masks_pred.npz</span>
                    </div>
                    <div class="flex justify-between py-1.5 border-b border-slate-700/50">
                        <span class="text-slate-400">Time Dimensions:</span>
                        <span class="font-mono">30 Time Frames</span>
                    </div>
                    <div class="flex justify-between py-1.5 border-b border-slate-700/50">
                        <span class="text-slate-400">Spatial Resolution:</span>
                        <span class="font-mono">1024 x 1024 Pixels</span>
                    </div>
                    <div class="flex justify-between py-1.5 border-b border-slate-700/50">
                        <span class="text-slate-400">Distinct Cell Tracks:</span>
                        <span class="font-mono text-emerald-400">33 Tracked Cells</span>
                    </div>
                    <div class="flex justify-between py-1.5">
                        <span class="text-slate-400">Modality:</span>
                        <span>2D Time-Lapse Microscopy</span>
                    </div>
                </div>
            </div>

            <!-- Active Cell Inspector -->
            <div class="glass-panel p-6 rounded-2xl flex-1 flex flex-col">
                <h3 class="text-lg font-bold text-white mb-3 flex items-center gap-2">
                    🔍 Frame Cell List & Positions
                </h3>
                <div id="cell-list" class="space-y-2 overflow-y-auto max-h-[300px] pr-1 flex-1 text-xs">
                    <!-- Cell items injected via JS -->
                </div>
            </div>
        </div>
    </main>

    <!-- Data Injection JS -->
    <script>
        const maskImages = {mask_b64_list};
        const synthImages = {synth_b64_list};
        const frameData = {frame_stats};

        let currentFrame = 0;
        let isPlaying = false;
        let playInterval = null;
        let speed = 1.0;
        let viewMode = 'synth';

        const canvas = document.getElementById('videoCanvas');
        const ctx = canvas.getContext('2d');
        const imgElementsMask = [];
        const imgElementsSynth = [];

        // Preload images
        for(let i=0; i<maskImages.length; i++) {{
            const imgM = new Image();
            imgM.src = "data:image/png;base64," + maskImages[i];
            imgElementsMask.push(imgM);

            const imgS = new Image();
            imgS.src = "data:image/png;base64," + synthImages[i];
            imgElementsSynth.push(imgS);
        }}

        function setMode(mode) {{
            viewMode = mode;
            document.getElementById('btn-mode-synth').className = mode === 'synth' 
                ? "px-4 py-2 text-xs font-semibold rounded-lg bg-cyan-500 text-slate-950 transition hover:bg-cyan-400 shadow-lg shadow-cyan-500/20"
                : "px-4 py-2 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 transition";
            document.getElementById('btn-mode-mask').className = mode === 'mask' 
                ? "px-4 py-2 text-xs font-semibold rounded-lg bg-cyan-500 text-slate-950 transition hover:bg-cyan-400 shadow-lg shadow-cyan-500/20"
                : "px-4 py-2 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 transition";
            drawFrame();
        }}

        function drawFrame() {{
            const img = viewMode === 'synth' ? imgElementsSynth[currentFrame] : imgElementsMask[currentFrame];
            if (!img.complete) {{
                img.onload = () => drawFrame();
                return;
            }}

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            const data = frameData[currentFrame];
            document.getElementById('hud-frame').innerText = currentFrame;
            document.getElementById('hud-count').innerText = data.count;
            document.getElementById('timeline').value = currentFrame;
            document.getElementById('time-curr').innerText = "00:" + (currentFrame < 10 ? "0" + currentFrame : currentFrame);

            const showTrails = document.getElementById('chk-trails').checked;
            const showLabels = document.getElementById('chk-labels').checked;

            // Draw motion trails (last 8 frames)
            if (showTrails && currentFrame > 0) {{
                const startF = Math.max(0, currentFrame - 8);
                data.cells.forEach(cell => {{
                    ctx.beginPath();
                    ctx.strokeStyle = "rgba(56, 189, 248, 0.6)";
                    ctx.lineWidth = 2;
                    let started = false;

                    for (let f = startF; f <= currentFrame; f++) {{
                        const prevCell = frameData[f].cells.find(c => c.id === cell.id);
                        if (prevCell) {{
                            if (!started) {{
                                ctx.moveTo(prevCell.x, prevCell.y);
                                started = true;
                            }} else {{
                                ctx.lineTo(prevCell.x, prevCell.y);
                            }}
                        }}
                    }}
                    ctx.stroke();
                }});
            }}

            // Draw Cell Centroids & Labels
            data.cells.forEach(cell => {{
                // Centroid point
                ctx.beginPath();
                ctx.arc(cell.x, cell.y, 3.5, 0, 2 * Math.PI);
                ctx.fillStyle = "#ffffff";
                ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
                ctx.shadowBlur = 4;
                ctx.fill();

                if (showLabels) {{
                    ctx.font = "bold 11px sans-serif";
                    ctx.fillStyle = "#38bdf8";
                    ctx.fillText("ID:" + cell.id, cell.x + 6, cell.y - 4);
                }}
            }});

            updateSidebarList(data.cells);
        }}

        function updateSidebarList(cells) {{
            const listEl = document.getElementById('cell-list');
            listEl.innerHTML = '';
            cells.forEach(c => {{
                const row = document.createElement('div');
                row.className = "flex items-center justify-between p-2 rounded-lg bg-slate-900/60 border border-slate-800 hover:border-cyan-500/40 transition";
                row.innerHTML = `
                    <div class="flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
                        <span class="font-bold text-slate-200">Cell #${{c.id}}</span>
                    </div>
                    <div class="font-mono text-slate-400">
                        (${{c.x}}, ${{c.y}}) | ${{c.area}}px²
                    </div>
                `;
                listEl.appendChild(row);
            }});
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('btn-play');
            if (isPlaying) {{
                btn.innerHTML = "<span>⏸</span> Pause";
                btn.className = "px-5 py-2.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold transition flex items-center gap-2";
                playInterval = setInterval(() => {{
                    currentFrame = (currentFrame + 1) % {total_frames};
                    drawFrame();
                }}, 200 / speed);
            }} else {{
                btn.innerHTML = "<span>▶</span> Play Video";
                btn.className = "px-5 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold transition flex items-center gap-2";
                clearInterval(playInterval);
            }}
        }}

        function onSeek(val) {{
            currentFrame = parseInt(val);
            drawFrame();
        }}

        function prevFrame() {{
            currentFrame = (currentFrame - 1 + {total_frames}) % {total_frames};
            drawFrame();
        }}

        function nextFrame() {{
            currentFrame = (currentFrame + 1) % {total_frames};
            drawFrame();
        }}

        function setSpeed(sp) {{
            speed = sp;
            document.querySelectorAll('.spd-btn').forEach(b => {{
                b.className = "spd-btn px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700";
            }});
            event.target.className = "spd-btn px-2.5 py-1 rounded bg-cyan-500 text-slate-950 font-bold";
            if (isPlaying) {{
                togglePlay();
                togglePlay();
            }}
        }}

        // Initial draw on window load
        window.onload = () => drawFrame();
    </script>
</body>
</html>
"""
    out_path = Path("cell_tracker_viewer.html")
    out_path.write_text(html_content, encoding="utf-8")
    print(f"[WebVisualizer] Generated HTML visualizer: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    build_web_visualizer()
