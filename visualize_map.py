#!/usr/bin/env python3
"""
visualize_web.py
----------------
Browser-based heatmap visualiser. No display/X11 needed.
Opens a web server you visit at http://localhost:8001

Usage (inside vlmaps container):
    python3 visualize_web.py

Then open in your browser:
    http://localhost:8001
"""

import numpy as np
import cv2
import json
import io
import base64
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

SAVE_DIR = Path("/vlmaps/data")
app = FastAPI()


# ── Map helpers ───────────────────────────────────────────────────────────────

def load_map():
    meta_path = SAVE_DIR / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    feat_sum   = np.load(SAVE_DIR / "feat_sum.npy")
    feat_count = np.load(SAVE_DIR / "feat_count.npy")
    rgb_sum    = np.load(SAVE_DIR / "rgb_sum.npy")
    rgb_count  = np.load(SAVE_DIR / "rgb_count.npy")
    return feat_sum, feat_count, rgb_sum, rgb_count, meta


def get_feature_map(feat_sum, feat_count):
    count = feat_count[:, :, None].clip(1)
    avg   = feat_sum / count
    norms = np.linalg.norm(avg, axis=-1, keepdims=True).clip(1e-6)
    return avg / norms


def get_rgb_map(rgb_sum, rgb_count):
    count = rgb_count[:, :, None].clip(1)
    return (rgb_sum / count).clip(0, 255).astype(np.uint8)


def encode_text(text: str):
    import torch, open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        feat = model.encode_text(tok([text]).to(device))
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze().cpu().numpy()


def crop_bounds(mask, pad=30):
    obs = np.where(mask)
    if len(obs[0]) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    r0 = max(0, obs[0].min() - pad)
    r1 = min(mask.shape[0], obs[0].max() + pad)
    c0 = max(0, obs[1].min() - pad)
    c1 = min(mask.shape[1], obs[1].max() + pad)
    return r0, r1, c0, c1


def map_to_png(rgb_map, sim_map, obs_mask, alpha=0.55, max_px=700):
    r0, r1, c0, c1 = crop_bounds(obs_mask)
    rgb_crop = rgb_map[r0:r1, c0:c1]
    obs_crop = obs_mask[r0:r1, c0:c1]

    h, w = rgb_crop.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = max(1, max_px // max(h, w))
    dh, dw = h * scale, w * scale

    base = cv2.resize(cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR),
                      (dw, dh), interpolation=cv2.INTER_NEAREST)

    if sim_map is not None:
        sim_crop = sim_map[r0:r1, c0:c1].copy()
        sim_crop[~obs_crop] = sim_crop[obs_crop].min() if obs_crop.any() else 0
        mn, mx = sim_crop[obs_crop].min(), sim_crop[obs_crop].max()
        if mx > mn:
            sim_norm = ((sim_crop - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            sim_norm = np.zeros_like(sim_crop, dtype=np.uint8)

        heat = cv2.applyColorMap(
            cv2.resize(sim_norm, (dw, dh), interpolation=cv2.INTER_NEAREST),
            cv2.COLORMAP_JET
        )
        obs_up = cv2.resize(obs_crop.astype(np.uint8) * 255,
                            (dw, dh), interpolation=cv2.INTER_NEAREST)
        mask3  = np.stack([obs_up]*3, axis=-1) > 0
        canvas = base.copy()
        canvas[mask3] = cv2.addWeighted(base, 1-alpha, heat, alpha, 0)[mask3]
    else:
        canvas = base

    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


# ── Cache CLIP model ──────────────────────────────────────────────────────────
_clip_cache = {}

def get_clip():
    if "model" not in _clip_cache:
        import torch, open_clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        model.eval()
        tok = open_clip.get_tokenizer("ViT-B-32")
        _clip_cache["model"]  = model
        _clip_cache["tok"]    = tok
        _clip_cache["device"] = device
    return _clip_cache["model"], _clip_cache["tok"], _clip_cache["device"]


def encode(text):
    import torch
    model, tok, device = get_clip()
    with torch.no_grad():
        feat = model.encode_text(tok([text]).to(device))
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze().cpu().numpy()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/api/rgb")
def rgb_image():
    data = load_map()
    if data is None:
        return JSONResponse({"error": "No map yet"}, status_code=404)
    feat_sum, feat_count, rgb_sum, rgb_count, _ = data
    rgb_map  = get_rgb_map(rgb_sum, rgb_count)
    obs_mask = feat_count > 0
    buf = map_to_png(rgb_map, None, obs_mask)
    if buf is None:
        return JSONResponse({"error": "No observations yet"}, status_code=404)
    return StreamingResponse(io.BytesIO(buf), media_type="image/png")


@app.get("/api/heatmap")
def heatmap(q: str = ""):
    data = load_map()
    if data is None:
        return JSONResponse({"error": "No map yet"}, status_code=404)
    feat_sum, feat_count, rgb_sum, rgb_count, _ = data
    rgb_map  = get_rgb_map(rgb_sum, rgb_count)
    obs_mask = feat_count > 0

    sim_map = None
    best_score = None
    if q:
        feat_map   = get_feature_map(feat_sum, feat_count)
        text_feat  = encode(q)
        sim_map    = (feat_map @ text_feat).astype(np.float32)
        if obs_mask.any():
            best_score = float(sim_map[obs_mask].max())

    buf = map_to_png(rgb_map, sim_map, obs_mask)
    if buf is None:
        return JSONResponse({"error": "No observations yet"}, status_code=404)

    # Return image as base64 + metadata
    b64 = base64.b64encode(buf).decode()
    return JSONResponse({
        "image": b64,
        "query": q,
        "best_score": best_score,
        "observed_cells": int(obs_mask.sum()),
    })


@app.get("/api/status")
def status():
    data = load_map()
    if data is None:
        return {"ready": False, "message": "No map file found — start semantic_map_node.py"}
    feat_sum, feat_count, rgb_sum, rgb_count, meta = data
    obs = int((feat_count > 0).sum())
    total = feat_count.size
    return {
        "ready": True,
        "observed_cells": obs,
        "total_cells": total,
        "coverage_pct": round(100 * obs / total, 2),
        "resolution_m": meta["resolution"],
        "map_size_m": meta["size_cells"] * meta["resolution"],
    }


# ── HTML page (self-contained) ────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VLMaps Heatmap</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #eee; font-family: system-ui, sans-serif;
       display: flex; flex-direction: column; align-items: center;
       min-height: 100vh; padding: 24px; gap: 20px; }
h1 { font-size: 1.4rem; font-weight: 500; color: #fff; }
.card { background: #1e1e1e; border-radius: 12px; padding: 20px;
        width: 100%; max-width: 760px; }
.row { display: flex; gap: 10px; align-items: center; }
input[type=text] { flex: 1; padding: 10px 14px; border-radius: 8px;
                   border: 1px solid #444; background: #2a2a2a;
                   color: #fff; font-size: 1rem; outline: none; }
input[type=text]:focus { border-color: #7c6ff7; }
button { padding: 10px 20px; border-radius: 8px; border: none;
         background: #7c6ff7; color: #fff; font-size: 1rem;
         cursor: pointer; white-space: nowrap; }
button:hover { background: #9d93f9; }
button.secondary { background: #333; }
button.secondary:hover { background: #444; }
#status { font-size: 0.85rem; color: #888; padding: 8px 0; }
#score  { font-size: 0.9rem; color: #a3e6a3; padding: 4px 0; min-height: 1.4em; }
.img-wrap { position: relative; width: 100%; background: #0a0a0a;
            border-radius: 8px; overflow: hidden; min-height: 200px;
            display: flex; align-items: center; justify-content: center; }
#map-img { max-width: 100%; display: block; image-rendering: pixelated; }
.legend { display: flex; align-items: center; gap: 8px; font-size: 0.8rem;
          color: #888; padding-top: 8px; }
.legend-bar { width: 140px; height: 12px; border-radius: 4px;
              background: linear-gradient(to right, #00007f, #0000ff,
              #00ffff, #ffff00, #ff0000); }
.spinner { border: 3px solid #333; border-top-color: #7c6ff7;
           border-radius: 50%; width: 32px; height: 32px;
           animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
#placeholder { color: #555; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>VLMaps — Semantic Heatmap</h1>

<div class="card">
  <div class="row">
    <input id="query" type="text" placeholder="Type an object: chair, door, table..."
           onkeydown="if(event.key==='Enter') query()">
    <button onclick="query()">Search</button>
    <button class="secondary" onclick="reset()">Reset</button>
  </div>
  <div id="score"></div>
  <div id="status">Loading map status...</div>
</div>

<div class="card">
  <div class="img-wrap" id="img-wrap">
    <div id="placeholder">Waiting for map data...</div>
    <img id="map-img" style="display:none">
  </div>
  <div class="legend">
    <span>low</span>
    <div class="legend-bar"></div>
    <span>high similarity</span>
    <span style="margin-left:auto;color:#555" id="cell-count"></span>
  </div>
</div>

<script>
let currentQuery = '';
let autoRefresh  = null;

async function updateStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    if (d.ready) {
      document.getElementById('status').textContent =
        `Coverage: ${d.coverage_pct}% (${d.observed_cells.toLocaleString()} cells) — `+
        `${d.map_size_m}m map @ ${d.resolution_m}m/cell`;
      document.getElementById('cell-count').textContent =
        `${d.observed_cells} cells observed`;
    } else {
      document.getElementById('status').textContent = d.message;
    }
  } catch(e) {}
}

async function loadImage(q) {
  const wrap = document.getElementById('img-wrap');
  const img  = document.getElementById('map-img');
  const ph   = document.getElementById('placeholder');

  // Show spinner
  ph.innerHTML = '<div class="spinner"></div>';
  ph.style.display = 'flex';
  img.style.display = 'none';

  try {
    const url = q ? `/api/heatmap?q=${encodeURIComponent(q)}` : '/api/rgb';
    if (q) {
      const r = await fetch(url);
      const d = await r.json();
      if (d.error) { ph.textContent = d.error; return; }
      img.src = 'data:image/png;base64,' + d.image;
      img.style.display = 'block';
      ph.style.display  = 'none';
      const sc = d.best_score !== null
        ? `Best match score: ${d.best_score.toFixed(3)}`
        : '';
      document.getElementById('score').textContent = sc;
    } else {
      img.src = '/api/rgb?' + Date.now();
      img.onload = () => { img.style.display='block'; ph.style.display='none'; };
      img.onerror = () => { ph.textContent = 'No RGB data yet — walk the robot around.'; };
    }
  } catch(e) {
    ph.textContent = 'Error loading map.';
  }
}

function query() {
  const q = document.getElementById('query').value.trim();
  if (!q) return;
  currentQuery = q;
  document.getElementById('score').textContent = 'Computing...';
  loadImage(q);
  clearInterval(autoRefresh);
  autoRefresh = setInterval(() => loadImage(currentQuery), 10000);
}

function reset() {
  currentQuery = '';
  document.getElementById('query').value = '';
  document.getElementById('score').textContent = '';
  loadImage('');
  clearInterval(autoRefresh);
  autoRefresh = setInterval(() => loadImage(''), 5000);
}

// Init
updateStatus();
setInterval(updateStatus, 5000);
loadImage('');
autoRefresh = setInterval(() => loadImage(currentQuery), 5000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting VLMaps web visualiser at http://localhost:8001")
    print("Open that URL in your browser.")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
