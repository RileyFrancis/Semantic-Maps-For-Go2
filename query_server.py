#!/usr/bin/env python3
"""
query_server.py
---------------
FastAPI server that exposes the semantic map to an LLM.
Run this in a second terminal alongside semantic_map_node.py.

Usage:
    python3 query_server.py

Endpoints:
    POST /query        {"text": "chair", "top_k": 5}
    GET  /map/status   map stats
    GET  /map/rgb      RGB map as PNG (for visualisation)
    POST /save         force save map to disk
"""

import threading
import numpy as np
import cv2
import json
import io
import rclpy
from rclpy.executors import SingleThreadedExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Import our node
from semantic_map_node import SemanticMapNode, SAVE_DIR

# ── Shared node instance ──────────────────────────────────────────────────────
_node: SemanticMapNode = None
_ros_thread: threading.Thread = None


def start_ros():
    global _node
    rclpy.init()
    _node = SemanticMapNode()
    executor = SingleThreadedExecutor()
    executor.add_node(_node)
    try:
        executor.spin()
    except Exception:
        pass
    finally:
        _node.shutdown()
        _node.destroy_node()
        rclpy.shutdown()


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="VLMaps Go2 Query API")


@app.on_event("startup")
async def startup():
    global _ros_thread
    _ros_thread = threading.Thread(target=start_ros, daemon=True)
    _ros_thread.start()
    # Give ROS a moment to initialise
    import time; time.sleep(3.0)


class QueryRequest(BaseModel):
    text: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    results: list     # [{x, y, score, cell_row, cell_col}]
    map_cells_observed: int
    best_location: dict   # {x, y, score}


@app.post("/query", response_model=QueryResponse)
def query_map(req: QueryRequest):
    if _node is None:
        raise HTTPException(503, "ROS node not ready yet")

    results, sim_map = _node.query(req.text, top_k=req.top_k)

    if not results:
        raise HTTPException(404, "No observed cells in map yet — walk the robot around first")

    formatted = []
    for x, y, score in results:
        # Also return cell indices for downstream use
        cell = _node.sem_map.world_to_cell(np.array([x, y]))
        formatted.append({
            "x": round(x, 3),
            "y": round(y, 3),
            "score": round(float(score), 4),
            "cell_row": int(cell[0]) if cell is not None else -1,
            "cell_col": int(cell[1]) if cell is not None else -1,
        })

    observed = int((_node.sem_map.feat_count > 0).sum())

    return QueryResponse(
        query=req.text,
        results=formatted,
        map_cells_observed=observed,
        best_location=formatted[0],
    )


@app.get("/map/status")
def map_status():
    if _node is None:
        raise HTTPException(503, "ROS node not ready yet")

    sem = _node.sem_map
    observed_feat = int((sem.feat_count > 0).sum())
    observed_rgb  = int((sem.rgb_count > 0).sum())
    total         = sem.N * sem.N
    pose = _node.latest_pose

    return {
        "map_resolution_m": sem.res,
        "map_size_cells": sem.N,
        "map_size_m": sem.N * sem.res,
        "cells_with_clip_features": observed_feat,
        "cells_with_rgb": observed_rgb,
        "total_cells": total,
        "coverage_pct": round(100 * observed_feat / total, 2),
        "robot_pose": {
            "x": round(float(pose[0]), 3) if pose is not None else None,
            "y": round(float(pose[1]), 3) if pose is not None else None,
            "yaw_rad": round(float(pose[2]), 3) if pose is not None else None,
        },
        "save_dir": str(SAVE_DIR),
    }


@app.get("/map/rgb")
def map_rgb_image():
    """Return the RGB map as a PNG for quick visualisation."""
    if _node is None:
        raise HTTPException(503, "ROS node not ready yet")

    rgb_map = _node.sem_map.get_rgb_map()   # (N, N, 3)

    # Crop to observed region + padding
    observed = np.where(_node.sem_map.rgb_count > 0)
    if len(observed[0]) == 0:
        raise HTTPException(404, "No RGB data yet")

    r0, r1 = observed[0].min(), observed[0].max()
    c0, c1 = observed[1].min(), observed[1].max()
    pad = 20
    r0, r1 = max(0, r0-pad), min(rgb_map.shape[0], r1+pad)
    c0, c1 = max(0, c0-pad), min(rgb_map.shape[1], c1+pad)

    crop = rgb_map[r0:r1, c0:c1]
    # Scale up for visibility
    scale = max(1, 400 // max(crop.shape[:2]))
    vis = cv2.resize(crop, (crop.shape[1]*scale, crop.shape[0]*scale),
                     interpolation=cv2.INTER_NEAREST)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    _, buf = cv2.imencode(".png", vis_bgr)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


@app.post("/save")
def force_save():
    if _node is None:
        raise HTTPException(503, "ROS node not ready yet")
    _node.sem_map.save(SAVE_DIR)
    return {"status": "saved", "path": str(SAVE_DIR)}


@app.get("/llm_context")
def llm_context(objects: str = "chair,door,table,desk,sofa,window"):
    """
    Convenience endpoint: query multiple objects at once and return
    a JSON blob ready to paste into an LLM prompt.

    Example: GET /llm_context?objects=chair,door,table
    """
    if _node is None:
        raise HTTPException(503, "ROS node not ready yet")

    queries = [o.strip() for o in objects.split(",") if o.strip()]
    context = {}

    for q in queries:
        try:
            results, _ = _node.query(q, top_k=3)
            context[q] = [
                {"x": round(x, 2), "y": round(y, 2), "confidence": round(s, 3)}
                for x, y, s in results
            ]
        except Exception:
            context[q] = []

    pose = _node.latest_pose
    return {
        "robot_pose": {
            "x": round(float(pose[0]), 2) if pose is not None else 0,
            "y": round(float(pose[1]), 2) if pose is not None else 0,
        },
        "map_resolution_m": _node.sem_map.res,
        "objects": context,
        "usage": (
            "Coordinates are in metres, map frame. "
            "confidence is cosine similarity (0-1). "
            "Higher = more likely to be that object."
        ),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
