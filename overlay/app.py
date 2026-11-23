"""
Local web app backend for the cvd_vit recolouring pipeline.

Wraps adjoint_multires_interp.py as a subprocess (the exact same script you've
been running by hand) so behaviour matches your validated CLI runs exactly --
this file does not reimplement any of the training logic.

Place this file directly in cvd_vit/ (next to adjoint_multires_interp.py),
and static/index.html in cvd_vit/static/index.html.

Run with:
    pip install flask
    python app.py
Then open http://127.0.0.1:5000 in a browser.

Design notes / limitations (fine for a single-user local demo, NOT safe for
multiple concurrent users):
- CustomDataset() always reads every *.png in data/custom/, so this app
  temporarily moves your existing data/custom images out of the way, runs
  on just the uploaded image, then moves your images back -- regardless of
  success or failure (see the `finally` block).
- The dev server is run single-threaded (threaded=False) so only one
  request is processed at a time, matching the fact that data/custom/ is
  a single shared folder.
"""
import base64
import glob
import io
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from PIL import Image
from scipy.ndimage import uniform_filter

from cvd_matrices import CVD_MATRICES

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.join(REPO_ROOT, 'data', 'custom')
BACKUP_DIR = os.path.join(REPO_ROOT, 'data', '_webapp_backup')
UPLOAD_NAME = 'webapp_upload.png'

# Frames used for the cheap LUT-based full-video playback (separate from
# the much smaller, GPU-optimised keyframe set controlled by the UI sliders).
FULL_PLAYBACK_MAX_FRAMES = 100
FULL_PLAYBACK_TARGET_FPS = 10.0
LUT_GRID_SIZE = 16

app = Flask(__name__, static_folder='static', static_url_path='')


def simulate_cvd(img_chw, matrix):
    # img_chw: (3, H, W) in [0,1] -> CVD-simulated (3, H, W)
    return np.clip(np.einsum('jhw,mj->mhw', img_chw, matrix), 0, 1)


def to_png_base64(img_chw):
    arr = (np.clip(img_chw, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def extract_frames(video_path, target_fps, max_frames):
    """
    Sample up to `max_frames` frames, evenly spaced across the video,
    aiming for roughly `target_fps` frames per second of source video
    (whichever is more restrictive wins). Saves them as zero-padded PNGs
    into CUSTOM_DIR (so CustomDataset picks them up in the right order)
    and returns (list_of_saved_paths, achieved_fps).
    """
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0

    if total_frames <= 0:
        cap.release()
        return [], target_fps

    desired_count = max(1, int(round(duration * target_fps))) if duration > 0 else max_frames
    frame_count = max(1, min(desired_count, max_frames, total_frames))
    wanted_indices = np.linspace(0, total_frames - 1, frame_count).astype(int).tolist()

    saved_paths = []
    frame_no = 0
    wanted_ptr = 0
    while wanted_ptr < len(wanted_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no == wanted_indices[wanted_ptr]:
            path = os.path.join(CUSTOM_DIR, f'frame_{wanted_ptr:04d}.png')
            cv2.imwrite(path, frame)  # frame is already BGR, which is what imwrite expects
            saved_paths.append(path)
            wanted_ptr += 1
        frame_no += 1
    cap.release()

    achieved_fps = frame_count / duration if duration > 0 else target_fps
    return saved_paths, achieved_fps


def simulate_cvd_hwc(img_hwc_uint8, matrix):
    # img_hwc_uint8: (H, W, 3) uint8 -> CVD-simulated (H, W, 3) uint8
    img_float = img_hwc_uint8.astype(np.float64) / 255.0
    out = np.clip(np.einsum('hwj,mj->hwm', img_float, matrix), 0, 1)
    return (out * 255).astype(np.uint8)


def extract_frame_arrays(video_path, target_fps, max_frames, size=256):
    """
    Like extract_frames, but returns in-memory RGB uint8 arrays (resized to
    size x size) instead of saving to disk -- used for the cheap LUT pass,
    which never needs to touch data/custom or go through the GPU pipeline.
    """
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0

    if total_frames <= 0:
        cap.release()
        return [], target_fps

    desired_count = max(1, int(round(duration * target_fps))) if duration > 0 else max_frames
    frame_count = max(1, min(desired_count, max_frames, total_frames))
    wanted_indices = np.linspace(0, total_frames - 1, frame_count).astype(int).tolist()

    frames = []
    frame_no = 0
    wanted_ptr = 0
    while wanted_ptr < len(wanted_indices):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_no == wanted_indices[wanted_ptr]:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (size, size))
            frames.append(frame_rgb)
            wanted_ptr += 1
        frame_no += 1
    cap.release()

    achieved_fps = frame_count / duration if duration > 0 else target_fps
    return frames, achieved_fps


def fit_color_lut(pairs, grid_size=LUT_GRID_SIZE, smooth_iters=3, smooth_size=3):
    """
    Fits a (grid_size, grid_size, grid_size, 3) colour lookup table from a
    list of (input_rgb_uint8 HxWx3, output_rgb_uint8 HxWx3) pairs, i.e. the
    actual before/after pixels the GPU optimiser produced on the keyframes.

    Colours that never appeared in any keyframe are filled in by smoothing
    outward from the colours that did (a normalised-convolution blend),
    falling back to "no change" (identity) wherever there's essentially no
    nearby evidence either way, rather than guessing wildly.
    """
    bin_size = 256.0 / grid_size
    sum_grid = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float64)
    count_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)

    for in_img, out_img in pairs:
        idx = np.clip((in_img.astype(np.int32) // bin_size).astype(np.int32), 0, grid_size - 1)
        flat_idx = (idx[..., 0] * grid_size + idx[..., 1]) * grid_size + idx[..., 2]
        flat_idx = flat_idx.ravel()
        out_flat = out_img.reshape(-1, 3).astype(np.float64)

        np.add.at(sum_grid.reshape(-1, 3), flat_idx, out_flat)
        np.add.at(count_grid.reshape(-1), flat_idx, 1)

    centres = (np.arange(grid_size) + 0.5) * bin_size
    identity_grid = np.stack(np.meshgrid(centres, centres, centres, indexing='ij'), axis=-1)

    observed = count_grid > 0
    mean_grid = np.where(observed[..., None], sum_grid / np.maximum(count_grid[..., None], 1), 0.0)

    # Push-pull smoothing to ESTIMATE unobserved bins from nearby observed
    # ones -- this only fills gaps. Bins we directly observed keep their
    # exact averaged value untouched below, so a strong, well-supported
    # mapping (like a whole flower recoloured consistently) is never
    # diluted by neighbouring empty bins.
    weight = observed.astype(np.float64)
    value = mean_grid * weight[..., None]
    for _ in range(smooth_iters):
        value = np.stack([uniform_filter(value[..., c], size=smooth_size) for c in range(3)], axis=-1)
        weight = uniform_filter(weight, size=smooth_size)
    weight_safe = np.maximum(weight, 1e-6)
    filled = value / weight_safe[..., None]

    lut = np.where(
        observed[..., None], mean_grid,
        np.where(weight[..., None] > 1e-3, filled, identity_grid)
    )
    return lut.astype(np.float32)


def apply_color_lut(frame_rgb_uint8, lut, grid_size=LUT_GRID_SIZE):
    """Nearest-bin LUT lookup, fully vectorised, no GPU/optimisation needed."""
    bin_size = 256.0 / grid_size
    idx = np.clip((frame_rgb_uint8.astype(np.int32) // bin_size).astype(np.int32), 0, grid_size - 1)
    mapped = lut[idx[..., 0], idx[..., 1], idx[..., 2]]
    return np.clip(mapped, 0, 255).astype(np.uint8)


def build_full_video_frames(video_path, lut, matrix):
    """
    Re-samples the source video far more densely than the GPU keyframes
    (cheap, since this is just a LUT lookup + a fixed matrix multiply --
    no ViT, no backprop), builds the 2x2 comparison grid per frame, and
    returns (list_of_base64_pngs, achieved_fps).
    """
    frames, achieved_fps = extract_frame_arrays(
        video_path, FULL_PLAYBACK_TARGET_FPS, FULL_PLAYBACK_MAX_FRAMES
    )
    frames_b64 = []
    for orig in frames:
        orig_cvd = simulate_cvd_hwc(orig, matrix)
        rec = apply_color_lut(orig, lut)
        rec_cvd = simulate_cvd_hwc(rec, matrix)

        top = np.concatenate([orig, orig_cvd], axis=1)
        bottom = np.concatenate([rec, rec_cvd], axis=1)
        grid_rgb = np.concatenate([top, bottom], axis=0)

        buf = io.BytesIO()
        Image.fromarray(grid_rgb).save(buf, format='PNG')
        frames_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

    return frames_b64, achieved_fps


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/run', methods=['POST'])
def run():
    file = request.files.get('image')
    if file is None:
        return jsonify({'error': 'No image uploaded'}), 400

    cvd_type = request.form.get('cvd_type', 'protan')
    if cvd_type not in ('protan', 'deutan', 'tritan'):
        return jsonify({'error': 'invalid cvd_type'}), 400

    severity = float(request.form.get('severity', '1.0'))
    if severity not in (0.4, 0.6, 1.0):
        return jsonify({'error': 'invalid severity'}), 400

    max_epochs = int(request.form.get('max_epochs', '150'))
    max_epochs = max(20, min(max_epochs, 300))

    os.makedirs(CUSTOM_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Move any of your existing data/custom images out of the way so this
    # run only processes the single uploaded image (and so it's fast).
    for f in glob.glob(os.path.join(CUSTOM_DIR, '*.png')):
        shutil.move(f, os.path.join(BACKUP_DIR, os.path.basename(f)))

    upload_path = os.path.join(CUSTOM_DIR, UPLOAD_NAME)
    results_folder = os.path.join(REPO_ROOT, 'results', 'custom', cvd_type, f'severity_{severity}')
    npz_path = os.path.join(results_folder, UPLOAD_NAME + '.npz')
    parquet_path = os.path.join(
        results_folder, f'custom_metric_results_{cvd_type}_{severity}_ws_4_split_0.parquet'
    )

    try:
        Image.open(file.stream).convert('RGB').save(upload_path)

        cmd = [
            sys.executable, 'adjoint_multires_interp.py',
            '--dataset_type', 'custom',
            '--cvd_type', cvd_type,
            '--severity', str(severity),
            '--batch_size', '1',
            '--max_epochs', str(max_epochs),
            '--val_interval', '10',
        ]
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900
        )

        if result.returncode != 0:
            return jsonify({
                'error': 'Training run failed',
                'stderr': result.stderr[-4000:],
            }), 500

        if not os.path.exists(npz_path):
            return jsonify({
                'error': 'No result was produced (try more epochs).',
                'stderr': result.stderr[-2000:],
            }), 500

        data = np.load(npz_path)
        input_image = data['input_image']
        output_image = data['image']
        matrix = CVD_MATRICES[severity][cvd_type]

        response = {
            'original': to_png_base64(input_image),
            'original_cvd': to_png_base64(simulate_cvd(input_image, matrix)),
            'recoloured': to_png_base64(output_image),
            'recoloured_cvd': to_png_base64(simulate_cvd(output_image, matrix)),
        }

        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            if len(df):
                row = df.iloc[-1]
                response['metrics'] = {
                    'base_cvd_base_error': float(row['base_cvd_base_error']),
                    'daltonised_base_error': float(row['daltonised_base_error']),
                    'dalt_boost': float(row['dalt_boost']),
                    'total_energy': float(row['total_energy']),
                }

        return jsonify(response)

    finally:
        # Clean up this run's temp files...
        if os.path.exists(upload_path):
            os.remove(upload_path)
        if os.path.exists(npz_path):
            os.remove(npz_path)
        # ...and always restore your original data/custom images, whether
        # this run succeeded or errored out.
        for f in glob.glob(os.path.join(CUSTOM_DIR, '*.png')):
            os.remove(f)
        for f in glob.glob(os.path.join(BACKUP_DIR, '*.png')):
            shutil.move(f, os.path.join(CUSTOM_DIR, os.path.basename(f)))


@app.route('/api/run_video', methods=['POST'])
def run_video():
    file = request.files.get('video')
    if file is None:
        return jsonify({'error': 'No video uploaded'}), 400

    cvd_type = request.form.get('cvd_type', 'protan')
    if cvd_type not in ('protan', 'deutan', 'tritan'):
        return jsonify({'error': 'invalid cvd_type'}), 400

    severity = float(request.form.get('severity', '1.0'))
    if severity not in (0.4, 0.6, 1.0):
        return jsonify({'error': 'invalid severity'}), 400

    max_epochs = max(20, min(int(request.form.get('max_epochs', '100')), 300))
    target_fps = max(0.2, min(float(request.form.get('sample_fps', '1.0')), 5.0))
    max_frames = max(4, min(int(request.form.get('max_frames', '20')), 60))
    batch_size = max(1, min(int(request.form.get('batch_size', '2')), 8))

    os.makedirs(CUSTOM_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Same isolation strategy as /api/run: move existing data/custom images
    # aside so this run only processes the extracted video frames.
    for f in glob.glob(os.path.join(CUSTOM_DIR, '*.png')):
        shutil.move(f, os.path.join(BACKUP_DIR, os.path.basename(f)))

    video_ext = os.path.splitext(file.filename or '')[1] or '.mp4'
    video_temp_path = os.path.join(REPO_ROOT, 'data', '_webapp_upload_video' + video_ext)
    results_folder = os.path.join(REPO_ROOT, 'results', 'custom', cvd_type, f'severity_{severity}')
    frame_paths = []

    try:
        file.save(video_temp_path)
        frame_paths, achieved_fps = extract_frames(video_temp_path, target_fps, max_frames)
        if not frame_paths:
            return jsonify({'error': 'Could not read any frames from that video.'}), 400

        # One training run optimises offsets for every sampled frame at
        # once (they're independent parameters, batched together each
        # epoch) -- this is far cheaper than one subprocess per frame.
        cmd = [
            sys.executable, 'adjoint_multires_interp.py',
            '--dataset_type', 'custom',
            '--cvd_type', cvd_type,
            '--severity', str(severity),
            '--batch_size', str(batch_size),
            '--max_epochs', str(max_epochs),
            '--val_interval', '10',
        ]
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600
        )

        if result.returncode != 0:
            return jsonify({
                'error': 'Training run failed',
                'stderr': result.stderr[-4000:],
            }), 500

        matrix = CVD_MATRICES[severity][cvd_type]

        # Pool the actual before/after pixel pairs the optimiser produced
        # on the keyframes, and fit one shared colour mapping from them.
        pairs = []
        for p in frame_paths:
            npz_path = os.path.join(results_folder, os.path.basename(p) + '.npz')
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path)
            in_hwc = (np.clip(data['input_image'], 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
            out_hwc = (np.clip(data['image'], 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
            pairs.append((in_hwc, out_hwc))

        if not pairs:
            return jsonify({
                'error': 'No frame results were produced (try more epochs).',
                'stderr': result.stderr[-2000:],
            }), 500

        lut = fit_color_lut(pairs)
        # video_temp_path still exists at this point (removed in `finally`)
        frames_b64, playback_fps = build_full_video_frames(video_temp_path, lut, matrix)

        if not frames_b64:
            return jsonify({'error': 'Could not resample the video for playback.'}), 500

        response = {'frames': frames_b64, 'fps': playback_fps}

        parquet_path = os.path.join(
            results_folder, f'custom_metric_results_{cvd_type}_{severity}_ws_4_split_0.parquet'
        )
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            if len(df):
                response['metrics'] = {
                    'base_cvd_base_error': float(df['base_cvd_base_error'].mean()),
                    'daltonised_base_error': float(df['daltonised_base_error'].mean()),
                    'dalt_boost': float(df['dalt_boost'].mean()),
                    'total_energy': float(df['total_energy'].mean()),
                    'n_frames': int(len(df)),
                }

        return jsonify(response)

    finally:
        if os.path.exists(video_temp_path):
            os.remove(video_temp_path)
        for p in frame_paths:
            if os.path.exists(p):
                os.remove(p)
            npz_path = os.path.join(results_folder, os.path.basename(p) + '.npz')
            if os.path.exists(npz_path):
                os.remove(npz_path)
        for f in glob.glob(os.path.join(CUSTOM_DIR, '*.png')):
            os.remove(f)
        for f in glob.glob(os.path.join(BACKUP_DIR, '*.png')):
            shutil.move(f, os.path.join(CUSTOM_DIR, os.path.basename(f)))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=False)
