"""
app.py - Component 2: Standalone Real-Time FastAPI Web Application

Serves an interactive web interface for real-time cell tracking, mitosis trajectory playback,
live event stream timeline, and dynamic motility charts.

Run standalone via CLI:
    python app.py
Open in browser:
    http://localhost:8000
"""

import io
import os
import sys
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Initialize FastAPI App
app = FastAPI(
    title="BioTrack-X Live Cell Tracking Platform",
    description="Real-Time Cell Tracking, Mitosis Trajectory Playback, and Motility Analytics Interface",
    version="2.0.0"
)

# Setup Templates & Static Files
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


def encode_array_to_base64(img_arr: np.ndarray) -> str:
    """Converts a 2D numpy array to a base64 PNG data URL."""
    # Normalize image to 0-255 uint8
    img_min, img_max = img_arr.min(), img_arr.max()
    if img_max > img_min:
        norm = ((img_arr - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        norm = img_arr.astype(np.uint8)

    img = Image.fromarray(norm)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Renders the standalone interactive live player web dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/datasets")
async def get_datasets():
    """Returns available CTC datasets and sequences."""
    sys.path.insert(0, ".")
    from ctc_loader import CTC_DATASETS
    datasets_list = []
    for key, meta in CTC_DATASETS.items():
        datasets_list.append({
            "key": key,
            "name": meta["name"],
            "url": meta["url"],
        })
    return JSONResponse({"status": "success", "datasets": datasets_list})


@app.post("/api/track")
async def run_tracking(
    dataset: str = Form("Fluo-N2DL-HeLa"),
    seq: str = Form("01"),
    subset: int = Form(25),
):
    """
    Runs BioTrack-X inference on a selected dataset and returns real-time tracking JSON payload.
    """
    try:
        sys.path.insert(0, ".")
        from ctc_loader import load_ctc_gt_masks, resolve_ctc_dataset
        from data_cleaner import clean_mask_sequence
        from biotrack_x.inference import run_biotrackx_inference
        from morphology import extract_dataset_morphology, get_morphology_summary_stats
        from lineage import detect_cell_events
        from behavior import compute_cell_kinematics, compute_population_behavior_summary

        ds_canon = resolve_ctc_dataset(dataset)
        masks, lineage_records = load_ctc_gt_masks(seq_name=seq, dataset_name=ds_canon, max_frames=subset, downsample_factor=2)
        masks, clean_stats = clean_mask_sequence(masks, min_area=15, boundary_smoothing=True)

        tracked_masks, track_graph = run_biotrackx_inference(masks)
        df_morphology = extract_dataset_morphology(tracked_masks)
        morph_summary = get_morphology_summary_stats(df_morphology)

        df_events, event_summary = detect_cell_events(track_graph, total_frames=masks.shape[0])
        df_kinematics = compute_cell_kinematics(df_morphology)
        behavior_summary = compute_population_behavior_summary(df_kinematics)

        from ctc_loader import extract_division_labels_from_lineage
        div_frames, parent_map = extract_division_labels_from_lineage(lineage_records, masks)

        # Build per-frame payload
        T, H, W = tracked_masks.shape
        frames_payload = []
        for t in range(T):
            frame_mask = tracked_masks[t]
            labels = [c for c in np.unique(frame_mask) if c > 0]
            
            cells_info = []
            for cid in labels:
                ys, xs = np.where(frame_mask == cid)
                if len(ys) > 0:
                    cy, cx = float(np.mean(ys)), float(np.mean(xs))
                    min_y, max_y = int(np.min(ys)), int(np.max(ys))
                    min_x, max_x = int(np.min(xs)), int(np.max(xs))
                    area = int(len(ys))
                    is_dividing = (int(cid) in parent_map) and (div_frames.get(int(cid)) == t)
                    daughters = parent_map.get(int(cid), []) if is_dividing else []

                    cells_info.append({
                        "cell_id": int(cid),
                        "centroid": [cy, cx],
                        "bbox": [min_y, min_x, max_y, max_x],
                        "area": area,
                        "is_dividing": is_dividing,
                        "daughters": daughters,
                    })

            frame_b64 = encode_array_to_base64(frame_mask)
            frames_payload.append({
                "frame_index": t,
                "cell_count": len(labels),
                "cells": cells_info,
                "image_b64": frame_b64,
            })

        events_list = df_events.to_dict(orient="records") if not df_events.empty else []
        from database import extract_cell_id, extract_frame_idx, DatabaseManager
        for ev in events_list:
            c_id = extract_cell_id(ev)
            f_idx = extract_frame_idx(ev)
            ev["cell_id"] = c_id
            ev["label_id"] = c_id
            ev["frame"] = f_idx
            ev["frame_index"] = f_idx
            etype = str(ev.get("event_type", "continuity")).lower()
            if not ev.get("details") or "N/A" in str(ev.get("details")):
                ev["details"] = f"Cell #{c_id} {etype} at Frame {f_idx}."

        from disease_analyzer import DiseaseBiomarkerAnalyzer
        analyzer = DiseaseBiomarkerAnalyzer()
        biomarkers = analyzer.analyze_biomarkers(
            behavior_summary=behavior_summary,
            morphology_summary=morph_summary,
            event_summary=event_summary,
        )

        # Persist experiment run to SQLite database
        db = DatabaseManager()
        exp_id = db.save_experiment_run(
            dataset_name=ds_canon,
            seq_name=seq,
            total_frames=T,
            total_cells=len(events_list),
            events=events_list,
            frames_payload=frames_payload,
            biomarkers=biomarkers,
        )

        return JSONResponse({
            "status": "success",
            "experiment_id": exp_id,
            "dataset": ds_canon,
            "seq": seq,
            "total_frames": T,
            "width": W,
            "height": H,
            "frames": frames_payload,
            "events": events_list,
            "event_summary": event_summary,
            "behavior_summary": behavior_summary,
            "morphology_summary": morph_summary,
            "clean_stats": clean_stats,
            "biomarkers": biomarkers,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/history")
async def get_tracking_history():
    """Returns past tracking experiment records from SQLite database."""
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        recent = db.get_recent_experiments(limit=15)
        return JSONResponse({"status": "success", "history": recent})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/generate-llm-report")
async def generate_llm_report_api(
    dataset: str = Form("Fluo-N2DL-HeLa"),
    seq: str = Form("01"),
    subset: int = Form(25),
):
    """
    Generates a full AI LLM Biological Narrative Report with frame timestamps, cell analysis, and disease risk screening.
    """
    try:
        sys.path.insert(0, ".")
        from llm_reporter import LLMBiologicalReporter
        reporter = LLMBiologicalReporter()
        report_text = reporter.generate_report_from_dataset(
            dataset_name=dataset,
            seq_name=seq,
            subset=subset,
        )
        return JSONResponse({
            "status": "success",
            "report_markdown": report_text,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/upload")
async def upload_custom_file(file: UploadFile = File(...)):
    """Accepts custom frame image or zip uploads for processing."""
    contents = await file.read()
    return JSONResponse({
        "status": "success",
        "filename": file.filename,
        "size": len(contents),
        "message": f"Successfully received {file.filename} ({len(contents)} bytes)."
    })


if __name__ == "__main__":
    import uvicorn
    print("==================================================")
    print("  BioTrack-X Real-Time FastAPI Server Starting   ")
    print("  Open in browser: http://localhost:8000         ")
    print("==================================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)
