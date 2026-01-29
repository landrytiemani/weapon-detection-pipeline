import os
import time
import json
import yaml
import cv2
import numpy as np
import streamlit as st

import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from datetime import datetime

# Project imports (relative to repo root)
from stages.stage_2_persondetection import PersonDetectionStage
from stages.stage_3_weapondetection import WeaponDetectionStage
from weapon_detection_pipeline.evaluation import evaluate_predictions

# ---------------------------
# Helpers
# ---------------------------

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

@st.cache_resource(show_spinner=False)
def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource(show_spinner=False)
def init_stages(config: dict):
    stage2 = PersonDetectionStage(config['stage_2'])
    stage3 = WeaponDetectionStage(config['stage_3'])
    return stage2, stage3

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Weapon Detection Demo", layout="wide")
st.title("🔎 Real-Time Weapon Detection – Interactive Demo")

with st.sidebar:
    st.header("Configuration")
    default_cfg = st.text_input("config.yaml path", value="config.yaml")
    run_device = st.selectbox("Device", ["auto", "cpu"], index=0, help="Use 'cpu' if CUDA not available or for stability in the demo")
    frame_stride = st.slider("Frame stride (skip every N-1 frames)", 1, 10, 1)
    draw_boxes = st.checkbox("Draw overlays", value=True)
    do_eval = st.checkbox("Enable evaluation (needs GT JSON)", value=False)
    gt_json_path = st.text_input("Ground-truth JSON path (optional)", value="")
    max_frames = st.number_input("Max frames to process (0 = all)", min_value=0, value=0)
    st.markdown("---")
    st.caption("Upload a short .mp4 or provide a path below")
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi"]) 
    video_path_text = st.text_input("Or video path on disk", value="")

# Load config and stages
try:
    cfg = load_config(default_cfg)
except Exception as e:
    st.error(f"Failed to load config: {e}")
    st.stop()

# Respect sidebar toggles
cfg['stage_2']['draw'] = draw_boxes
cfg['stage_3']['draw'] = draw_boxes
cfg['pipeline'] = cfg.get('pipeline', {})

if uploaded is not None:
    tmp_dir = os.path.join(".streamlit_cache", "uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"uploaded_{int(time.time())}.mp4")
    with open(tmp_path, 'wb') as f:
        f.write(uploaded.read())
    video_source = tmp_path
elif video_path_text:
    video_source = video_path_text
else:
    video_source = cfg['pipeline'].get('video_source', '')

if not video_source or not os.path.exists(video_source):
    st.warning("Provide a valid video (upload or path).")
    st.stop()

# Initialize stages
try:
    stage2, stage3 = init_stages(cfg)
except Exception as e:
    st.error(f"Failed to initialize stages: {e}")
    st.stop()

# Main columns
left, right = st.columns([2, 1])
video_area = left.empty()
progress_bar = left.progress(0)

# Live stats placeholders
with right:
    st.subheader("Live Stats")
    fps_ph = st.metric("FPS", "-")
    persons_ph = st.metric("# Persons", "0")
    weapons_ph = st.metric("# Weapons", "0")
    confp_ph = st.metric("Avg Person Conf", "0.00")
    confw_ph = st.metric("Avg Weapon Conf", "0.00")
    st.markdown("---")
    map05_ph = st.metric("mAP@0.5 (snippet)", "-")
    map_avg_ph = st.metric("mAP@0.5–0.9 (snippet)", "-")

# Evaluation accumulators
predictions = {}
frame_id = 0

# Optional GT preload
_gt_data = None
if do_eval and gt_json_path and os.path.exists(gt_json_path):
    try:
        with open(gt_json_path, 'r') as f:
            _gt_data = json.load(f)
    except Exception as e:
        st.warning(f"Could not load GT JSON: {e}")

# Video capture
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    st.error("Cannot open video source")
    st.stop()

# Run loop
start_time_overall = time.time()
processed_frames = 0

def _draw_gt_boxes(frame, frame_idx, gt):
    # GT is assumed in xywh with 480px height scaled to 640, same as your main.py
    anns = [ann['bbox'] for ann in gt.get('annotations', []) if ann['image_id'] == frame_idx]
    scale_y = 640 / 480.0
    for x, y, w, h in anns:
        x1 = int(x)
        y1 = int(y * scale_y)
        x2 = int(x + w)
        y2 = int((y + h) * scale_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_id += 1
    if frame_stride > 1 and (frame_id - 1) % frame_stride != 0:
        continue

    t0 = time.time()
    if frame.shape[:2] != (640, 640):
        frame = cv2.resize(frame, (640, 640))

    # Stage 2 (persons)
    frame_vis, person_data, person_stats = stage2.run(frame, frame_id)

    # Stage 3 (weapons)
    frame_vis, weapon_stats = stage3.run(frame_vis, person_data)

    # Accumulate predictions for mAP snippets
    if do_eval:
        raw_boxes = weapon_stats.get("bboxes", [])
        predictions[frame_id] = [xyxy_to_xywh(b) for b in raw_boxes]
        if _gt_data is not None:
            _draw_gt_boxes(frame_vis, frame_id, _gt_data)

    # FPS
    fps = 1.0 / max(1e-6, (time.time() - t0))
    processed_frames += 1

    # Display
    video_area.image(cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB), channels="RGB")

    # Update stats
    fps_ph.delta(f"{fps:.2f}")
    persons_ph.delta(str(person_stats['count']))
    weapons_ph.delta(str(weapon_stats['count']))
    confp_ph.delta(f"{person_stats['avg_confidence']:.2f}")
    confw_ph.delta(f"{weapon_stats['avg_confidence']:.2f}")

    # Progress (rough)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames and processed_frames >= max_frames:
        break
    if total_frames > 0:
        progress_bar.progress(min(1.0, frame_id / total_frames))

# After loop: small evaluation snippet (optional)
map05_txt = "-"
mapavg_txt = "-"
if do_eval and predictions:
    try:
        res = evaluate_predictions(predictions, gt_json_path)
        # Your evaluate_predictions returns dict with mAPs under 'mAPs'. We try both variants
        mAPs = res.get('mAPs', {})
        map05_txt = f"{mAPs.get('mAP@0.5', mAPs.get('mAP@0.5', 0.0)):.4f}" if mAPs else "-"
        mapavg_txt = f"{mAPs.get('mAP@0.5:0.9', mAPs.get('mAP@0.5:0.9', 0.0)):.4f}" if mAPs else "-"
    except Exception as e:
        map05_txt = mapavg_txt = f"eval error: {e}"

map05_ph.delta(map05_txt)
map_avg_ph.delta(mapavg_txt)

cap.release()

left.success("Finished!")
right.caption(f"Processed frames: {processed_frames} | Elapsed: {time.time() - start_time_overall:.2f}s")
