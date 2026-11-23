"""
Local web app backend for the cvd_vit recolouring pipeline.

Wraps adjoint_multires_interp.py as a subprocess (the exact same script you've
been running by hand) so behaviour matches your validated CLI runs exactly --
this file does not reimplement any of the training logic.

Place this file directly in cvd_vit/ (next to adjoint_multires_interp.py),
and static/index.html in cvd_vit/static/index.html.
(setup.sh does this copying for you.)

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

HOW THIS FILE IS ORGANISED (read top to bottom)
  1. Imports and constants
  2. Small image helpers          (CVD simulation, PNG encoding)
  3. Video frame helpers          (pull keyframes out of an uploaded video)
  4. Colour lookup table (LUT)    (our video extension: learn a colour mapping,
                                   apply it to every frame, encode a smooth MP4)
  5. Flask routes                 (/  ,  /api/run  ,  /api/run_video  ,  /media/<file>)
  6. Server start-up
"""

# =============================================================================
# 1. IMPORTS AND CONSTANTS
# =============================================================================
import base64          # turns PNG bytes into text so they can travel inside JSON
import glob            # finds files by pattern, e.g. every *.png in a folder
import io              # in-memory byte buffers (so we don't write temp PNGs to disk)
import os
import shutil          # moving files between folders
import subprocess      # launches adjoint_multires_interp.py as a separate process
import sys
import uuid          # random names for the output video files

import cv2             # OpenCV: reads video files and resizes frames
import numpy as np
import pandas as pd    # reads the metrics .parquet file the optimiser writes
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image  # reads uploaded images and writes PNGs
from scipy.ndimage import map_coordinates, uniform_filter  # LUT gap-filling and smooth interpolation

# ffmpeg (bundled by the `imageio-ffmpeg` pip package) turns our recoloured frames
# into an H.264 MP4, the one video format every browser can play.
try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

# 3x3 matrices that simulate how a colour-blind viewer sees an image.
# Built in cvd_matrices.py: CVD_MATRICES[severity][cvd_type] -> 3x3 array.
from cvd_matrices import CVD_MATRICES

# Where things live on disk. REPO_ROOT is the folder this file sits in.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# The optimiser reads its input images from here (it always takes ALL *.png here).
CUSTOM_DIR = os.path.join(REPO_ROOT, 'data', 'custom')
# Your own images are parked here while the app runs, then put back afterwards.
BACKUP_DIR = os.path.join(REPO_ROOT, 'data', '_webapp_backup')
# File name given to an uploaded image while it is being processed.
UPLOAD_NAME = 'webapp_upload.png'

# The colour table is fitted from the few GPU-optimised keyframes, then applied to
# EVERY frame of the clip (cheap) so playback is as smooth as the original video.
LUT_GRID_SIZE = 16        # the fitted table is 16 x 16 x 16 bins (R, G, B)
DENSE_LUT_SIZE = 64       # ...and is interpolated up to 64 x 64 x 64 for fast, smooth lookups
OUTPUT_MAX_FPS = 30.0     # sources above this are thinned (e.g. 60 fps -> 30 fps)
OUTPUT_MAX_SECONDS = 30   # longest stretch of the clip that will be rendered
OUTPUT_PANEL_SIDE = 360   # each of the four panels is at most this many pixels on its longest side
# Finished videos are written here and served at /media/<name>.
OUTPUT_DIR = os.path.join(REPO_ROOT, 'data', '_webapp_outputs')

# Flask serves everything in ./static (our index.html) from the site root.
app = Flask(__name__, static_folder='static', static_url_path='')


# =============================================================================
# 2. SMALL IMAGE HELPERS
# =============================================================================
def simulate_cvd(img_chw, matrix):
    # img_chw: (3, H, W) in [0,1] -> CVD-simulated (3, H, W)
    # For every pixel, multiply its (R,G,B) by the 3x3 CVD matrix. einsum does
    # this for the whole image at once. clip keeps values inside [0,1].
    return np.clip(np.einsum('jhw,mj->mhw', img_chw, matrix), 0, 1)


def to_png_base64(img_chw):
    # Convert a (3,H,W) float image in [0,1] into a base64 PNG string that the
    # browser can show directly as <img src="data:image/png;base64,...">.
    arr = (np.clip(img_chw, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)  # -> (H,W,3) 0..255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# =============================================================================
# 3. VIDEO FRAME HELPERS
# =============================================================================
def extract_frames(video_path, target_fps, max_frames):
    """
    Sample up to `max_frames` frames, evenly spaced across the video,
    aiming for roughly `target_fps` frames per second of source video
    (whichever is more restrictive wins). Saves them as zero-padded PNGs
    into CUSTOM_DIR (so CustomDataset picks them up in the right order)
    and returns (list_of_saved_paths, achieved_fps).

    These are the KEYFRAMES: the few frames that get the full (slow) GPU
    optimisation.
    """
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0               # fall back to 30 if unknown
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0

    if total_frames <= 0:
        cap.release()
        return [], target_fps

    # How many frames do we want? duration x fps, but never more than the
    # caller's cap or the number of frames the video actually has.
    desired_count = max(1, int(round(duration * target_fps))) if duration > 0 else max_frames
    frame_count = max(1, min(desired_count, max_frames, total_frames))
    # Evenly spaced frame numbers across the whole clip.
    wanted_indices = np.linspace(0, total_frames - 1, frame_count).astype(int).tolist()

    saved_paths = []
    frame_no = 0      # index of the frame we've just read
    wanted_ptr = 0    # which entry of wanted_indices we're waiting for next
    while wanted_ptr < len(wanted_indices):
        ret, frame = cap.read()
        if not ret:   # ran out of frames early
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
    # Same idea as simulate_cvd, but for (H, W, 3) uint8 images (0..255), which
    # is the format OpenCV/PIL use. Convert to float, apply matrix, convert back.
    # img_hwc_uint8: (H, W, 3) uint8 -> CVD-simulated (H, W, 3) uint8
    img_float = img_hwc_uint8.astype(np.float64) / 255.0
    out = np.clip(np.einsum('hwj,mj->hwm', img_float, matrix), 0, 1)
    return (out * 255).astype(np.uint8)


# =============================================================================
# 4. COLOUR LOOKUP TABLE (LUT) AND VIDEO ENCODING -- OUR VIDEO EXTENSION, NOT IN THE PAPER
#
# Idea: the optimiser changes each keyframe's colours. We watch what it did
# ("this shade of red moved this far towards magenta"), store it as a table of
# colour SHIFTS keyed by input colour, and reuse the table on every frame. The
# same input colour always gets the same shift, so the video can't flicker.
# Trade-off: the table only knows colours, not positions, so it can't treat
# the same red differently on the subject vs. the background.
#
# Why shifts, not output colours: storing "the output colour for this bin"
# flattens every colour inside a bin to one value (banding), and the identity
# fallback would nudge pure black/white. A shift (output minus input) added to
# the real pixel keeps all the original detail, and "no change" is simply 0.
# =============================================================================
def fit_color_lut(pairs, grid_size=LUT_GRID_SIZE, smooth_iters=3, smooth_size=3):
    """
    Fits a (grid_size, grid_size, grid_size, 3) table of average colour SHIFTS
    from a list of (input_rgb_uint8 HxWx3, output_rgb_uint8 HxWx3) pairs, i.e.
    the actual before/after pixels the GPU optimiser produced on the keyframes.

    Colours that never appeared in any keyframe get a shift estimated by
    smoothing outward from the colours that did (a normalised-convolution
    blend), falling back to "no shift" (0) wherever there's essentially no
    nearby evidence either way, rather than guessing wildly.
    """
    # The RGB cube is cut into grid_size^3 small boxes ("bins"). With 16 bins
    # per axis each bin spans 256/16 = 16 intensity levels.
    bin_size = 256.0 / grid_size
    # sum_grid: running total of the colour SHIFTS seen in each bin.
    # count_grid: how many pixels landed in each bin. sum / count = average shift.
    sum_grid = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float64)
    count_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)

    # --- Step A: collect statistics from every keyframe's before/after pair ---
    for in_img, out_img in pairs:
        # Which bin does each INPUT pixel fall into? (integer division by bin size)
        idx = np.clip((in_img.astype(np.int32) // bin_size).astype(np.int32), 0, grid_size - 1)
        # Turn the (r_bin, g_bin, b_bin) triple into one flat index so we can
        # accumulate with a single vectorised call.
        flat_idx = (idx[..., 0] * grid_size + idx[..., 1]) * grid_size + idx[..., 2]
        flat_idx = flat_idx.ravel()
        shift_flat = out_img.reshape(-1, 3).astype(np.float64) - in_img.reshape(-1, 3).astype(np.float64)

        # add.at handles repeated indices correctly (many pixels, same bin).
        np.add.at(sum_grid.reshape(-1, 3), flat_idx, shift_flat)
        np.add.at(count_grid.reshape(-1), flat_idx, 1)

    # --- Step B: per-bin average shift, only where we saw pixels ---
    observed = count_grid > 0
    mean_grid = np.where(observed[..., None], sum_grid / np.maximum(count_grid[..., None], 1), 0.0)

    # --- Step C: estimate bins we never saw ---
    # Push-pull smoothing to ESTIMATE unobserved bins from nearby observed
    # ones -- this only fills gaps. Bins we directly observed keep their
    # exact averaged value untouched below, so a strong, well-supported
    # mapping (like a whole flower recoloured consistently) is never
    # diluted by neighbouring empty bins.
    # How it works: blur the values AND the "was this observed" weights by the
    # same amount, then divide. Where evidence exists nearby you get a
    # weighted average of it; where there is none, the weight stays ~0.
    weight = observed.astype(np.float64)
    value = mean_grid * weight[..., None]
    for _ in range(smooth_iters):
        value = np.stack([uniform_filter(value[..., c], size=smooth_size) for c in range(3)], axis=-1)
        weight = uniform_filter(weight, size=smooth_size)
    weight_safe = np.maximum(weight, 1e-6)   # avoid dividing by zero
    filled = value / weight_safe[..., None]

    # --- Step D: combine the three sources, in priority order ---
    #   1. observed bin            -> exact average shift we measured
    #   2. unobserved, has nearby  -> smoothed estimate
    #   3. unobserved, nothing near-> zero shift (leave the colour alone)
    lut = np.where(
        observed[..., None], mean_grid,
        np.where(weight[..., None] > 1e-3, filled, 0.0)
    )
    return lut.astype(np.float32)


def make_dense_lut(lut, dense_size=DENSE_LUT_SIZE):
    """
    Blows the coarse 16^3 shift table up to dense_size^3 with trilinear
    interpolation. Between two bin centres the shift changes gradually instead
    of jumping, which is what removes banding and shimmer when a colour drifts
    slowly across frames (lighting changes, camera motion). Done once per video.
    """
    g = lut.shape[0]
    # The input colour at the centre of each dense cell...
    centres = (np.arange(dense_size) + 0.5) * (256.0 / dense_size)
    # ...expressed as a position in the coarse grid (bin centres sit at i + 0.5).
    coords = np.clip(centres / (256.0 / g) - 0.5, 0, g - 1)
    mesh = np.meshgrid(coords, coords, coords, indexing='ij')
    pts = np.stack([m.ravel() for m in mesh])
    dense = np.stack(
        [map_coordinates(lut[..., c], pts, order=1, mode='nearest').reshape((dense_size,) * 3)
         for c in range(3)],
        axis=-1,
    )
    return dense.astype(np.float32)


def apply_color_lut(frame_rgb_uint8, dense_lut):
    """
    Recolour one frame: look up the shift for each pixel's colour and ADD it to the
    pixel's real value (so fine detail is untouched). Fully vectorised, no GPU.
    """
    shift_bits = int(np.log2(256 // dense_lut.shape[0]))   # 64 cells -> drop the lowest 2 bits
    idx = frame_rgb_uint8 >> shift_bits
    shift = dense_lut[idx[..., 0], idx[..., 1], idx[..., 2]]
    return np.clip(np.rint(frame_rgb_uint8.astype(np.float32) + shift), 0, 255).astype(np.uint8)


def find_ffmpeg():
    # Prefer the copy bundled with the imageio-ffmpeg pip package; fall back to a system ffmpeg.
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return shutil.which('ffmpeg')


def render_comparison_video(video_path, dense_lut, matrix, out_path):
    """
    Recolours EVERY frame of the clip (at its own frame rate, up to 30 fps and 30 s)
    and writes one H.264 MP4 showing the 2x2 comparison:
        top row    : original   | original as seen with CVD
        bottom row : recoloured | recoloured as seen with CVD
    Frames are streamed straight into ffmpeg one at a time, so memory use stays small.
    Returns (frames_written, fps, truncated).
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found. Run:  pip install imageio-ffmpeg")

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0 or src_fps > 240:
        src_fps = 30.0
    # Keep every `step`-th frame so the output never exceeds OUTPUT_MAX_FPS.
    step = max(1, int(round(src_fps / OUTPUT_MAX_FPS)))
    out_fps = src_fps / step
    max_out = int(OUTPUT_MAX_SECONDS * out_fps)

    ok, frame_bgr = cap.read()
    if not ok:
        cap.release()
        return 0, out_fps, False

    # Keep the video's own proportions (the old version squashed everything to a square).
    # H.264 needs even pixel dimensions, hence the rounding to a multiple of 2.
    h, w = frame_bgr.shape[:2]
    scale = OUTPUT_PANEL_SIDE / max(h, w)
    qw = max(2, int(round(w * scale / 2)) * 2)
    qh = max(2, int(round(h * scale / 2)) * 2)

    cmd = [
        ffmpeg, '-y', '-loglevel', 'error',
        # input: raw RGB frames arriving on stdin
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{2 * qw}x{2 * qh}', '-r', f'{out_fps:.3f}', '-i', '-',
        # output: H.264 in yuv420p with the index at the front (plays in every browser, starts at once)
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    n_written = 0
    frame_no = 0
    truncated = False
    try:
        while ok:
            if frame_no % step == 0:
                if n_written >= max_out:
                    truncated = True      # clip is longer than OUTPUT_MAX_SECONDS
                    break
                orig = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)          # OpenCV is BGR, we want RGB
                orig = cv2.resize(orig, (qw, qh), interpolation=cv2.INTER_AREA)
                orig_cvd = simulate_cvd_hwc(orig, matrix)                   # original, as a CVD viewer sees it
                rec = apply_color_lut(orig, dense_lut)                      # recoloured via the learned table
                rec_cvd = simulate_cvd_hwc(rec, matrix)                     # recoloured, as a CVD viewer sees it
                grid = np.concatenate(
                    [np.concatenate([orig, orig_cvd], axis=1),
                     np.concatenate([rec, rec_cvd], axis=1)], axis=0)
                proc.stdin.write(np.ascontiguousarray(grid).tobytes())
                n_written += 1
            ok, frame_bgr = cap.read()
            frame_no += 1
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        err = proc.stderr.read().decode('utf-8', errors='replace')
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError('ffmpeg failed: ' + err[-500:])
    return n_written, out_fps, truncated


# =============================================================================
# 5. FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    # Serve the web page (static/index.html) when you open http://127.0.0.1:5000
    return app.send_static_file('index.html')


@app.route('/media/<path:name>')
def media(name):
    # Serve a finished video. conditional=True enables HTTP range requests, which is
    # what lets the browser scrub and start playing before the whole file has arrived.
    return send_from_directory(OUTPUT_DIR, name, mimetype='video/mp4', conditional=True)


@app.route('/api/run', methods=['POST'])
def run():
    """
    IMAGE MODE. The browser POSTs one image plus the chosen settings.
    Steps: validate input -> park your own images -> save the upload ->
    run the paper's optimiser -> read its .npz/.parquet output -> reply with
    four PNGs and the metrics -> (always) clean up.
    """
    # --- Validate what the browser sent ---
    file = request.files.get('image')
    if file is None:
        return jsonify({'error': 'No image uploaded'}), 400

    cvd_type = request.form.get('cvd_type', 'protan')
    if cvd_type not in ('protan', 'deutan', 'tritan'):
        return jsonify({'error': 'invalid cvd_type'}), 400

    severity = float(request.form.get('severity', '1.0'))
    if severity not in (0.4, 0.6, 1.0):
        return jsonify({'error': 'invalid severity'}), 400

    # Clamp epochs to a sane range so nobody can request a 10-hour job.
    max_epochs = int(request.form.get('max_epochs', '150'))
    max_epochs = max(20, min(max_epochs, 300))

    os.makedirs(CUSTOM_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Move any of your existing data/custom images out of the way so this
    # run only processes the single uploaded image (and so it's fast).
    for f in glob.glob(os.path.join(CUSTOM_DIR, '*.png')):
        shutil.move(f, os.path.join(BACKUP_DIR, os.path.basename(f)))

    # Paths where the optimiser will leave its results for this image.
    upload_path = os.path.join(CUSTOM_DIR, UPLOAD_NAME)
    results_folder = os.path.join(REPO_ROOT, 'results', 'custom', cvd_type, f'severity_{severity}')
    npz_path = os.path.join(results_folder, UPLOAD_NAME + '.npz')   # the images
    parquet_path = os.path.join(                                    # the metrics table
        results_folder, f'custom_metric_results_{cvd_type}_{severity}_ws_4_split_0.parquet'
    )

    try:
        # Re-save the upload as an RGB PNG into the folder the optimiser reads.
        Image.open(file.stream).convert('RGB').save(upload_path)

        # The command line we would otherwise type by hand.
        cmd = [
            sys.executable, 'adjoint_multires_interp.py',
            '--dataset_type', 'custom',
            '--cvd_type', cvd_type,
            '--severity', str(severity),
            '--batch_size', '1',
            '--max_epochs', str(max_epochs),
            '--val_interval', '10',   # evaluate/save results every 10 epochs
        ]
        # Run it and wait. Times out after 15 minutes (900 s).
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900
        )

        # Non-zero exit code = the script crashed; show the tail of its error text.
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

        # The .npz holds two arrays, both (3, H, W) floats in [0,1]:
        #   input_image = the original   |   image = the recoloured result
        data = np.load(npz_path)
        input_image = data['input_image']
        output_image = data['image']
        matrix = CVD_MATRICES[severity][cvd_type]

        # The four pictures of the 2x2 comparison, as base64 PNG strings.
        response = {
            'original': to_png_base64(input_image),
            'original_cvd': to_png_base64(simulate_cvd(input_image, matrix)),
            'recoloured': to_png_base64(output_image),
            'recoloured_cvd': to_png_base64(simulate_cvd(output_image, matrix)),
        }

        # Add the metrics (if the optimiser got far enough to write them).
        # Use the last row = the most recent evaluation.
        #   base_cvd_base_error   contrast lost by the ORIGINAL under CVD
        #   daltonised_base_error contrast lost by the RECOLOURED image under CVD
        #   dalt_boost            improvement (positive = recoloured is better)
        #   total_energy          how much the image was changed (naturalness proxy)
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
        # `finally` runs whether we returned normally, returned an error, or crashed.
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
    """
    VIDEO MODE (our extension). Steps:
      1. pick a few KEYFRAMES from the video and save them as PNGs
      2. run the paper's GPU optimiser on those keyframes (slow part)
      3. learn one colour lookup table from the before/after keyframe pixels
      4. apply that table to EVERY frame of the clip (fast, no GPU) and encode an MP4
      5. reply with the MP4's address; the browser plays it like any normal video
    """
    # --- Validate input (same checks as image mode) ---
    file = request.files.get('video')
    if file is None:
        return jsonify({'error': 'No video uploaded'}), 400

    cvd_type = request.form.get('cvd_type', 'protan')
    if cvd_type not in ('protan', 'deutan', 'tritan'):
        return jsonify({'error': 'invalid cvd_type'}), 400

    severity = float(request.form.get('severity', '1.0'))
    if severity not in (0.4, 0.6, 1.0):
        return jsonify({'error': 'invalid severity'}), 400

    # Slider values, clamped to safe ranges.
    max_epochs = max(20, min(int(request.form.get('max_epochs', '100')), 300))
    target_fps = max(0.2, min(float(request.form.get('sample_fps', '1.0')), 5.0))  # keyframes per second
    max_frames = max(4, min(int(request.form.get('max_frames', '20')), 60))        # max keyframes
    batch_size = max(1, min(int(request.form.get('batch_size', '2')), 8))          # frames per GPU batch

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
        # Step 1: save the upload, then pull out the keyframes.
        file.save(video_temp_path)
        frame_paths, achieved_fps = extract_frames(video_temp_path, target_fps, max_frames)
        if not frame_paths:
            return jsonify({'error': 'Could not read any frames from that video.'}), 400

        # Step 2: one training run optimises offsets for every sampled frame at
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
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600  # 1 hour limit
        )

        if result.returncode != 0:
            return jsonify({
                'error': 'Training run failed',
                'stderr': result.stderr[-4000:],
            }), 500

        matrix = CVD_MATRICES[severity][cvd_type]

        # Step 3: Pool the actual before/after pixel pairs the optimiser produced
        # on the keyframes, and fit one shared colour mapping from them.
        pairs = []
        for p in frame_paths:
            # The optimiser wrote one .npz per keyframe, named after the frame.
            npz_path = os.path.join(results_folder, os.path.basename(p) + '.npz')
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path)
            # (3,H,W) float [0,1] -> (H,W,3) uint8 [0,255], the LUT's format
            in_hwc = (np.clip(data['input_image'], 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
            out_hwc = (np.clip(data['image'], 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
            pairs.append((in_hwc, out_hwc))

        if not pairs:
            return jsonify({
                'error': 'No frame results were produced (try more epochs).',
                'stderr': result.stderr[-2000:],
            }), 500

        # Step 4: learn the table, then apply it to every frame and encode the MP4.
        dense_lut = make_dense_lut(fit_color_lut(pairs))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for old in glob.glob(os.path.join(OUTPUT_DIR, '*.mp4')):   # keep only the latest result
            os.remove(old)
        out_name = f'recoloured_{uuid.uuid4().hex[:8]}.mp4'
        # video_temp_path still exists at this point (removed in `finally`)
        try:
            n_out, out_fps, truncated = render_comparison_video(
                video_temp_path, dense_lut, matrix, os.path.join(OUTPUT_DIR, out_name)
            )
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 500

        if n_out == 0:
            return jsonify({'error': 'Could not read the video for playback.'}), 500

        # Step 5: reply with where the finished video can be fetched.
        response = {
            'video_url': '/media/' + out_name,
            'fps': out_fps,
            'n_output_frames': n_out,
            'truncated': truncated,   # True if the clip was longer than OUTPUT_MAX_SECONDS
        }

        # Metrics are averaged over all keyframes (image mode just used one row).
        # These describe the keyframes the optimiser actually saw, not every
        # frame of the final playback.
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
                    # One row per keyframe (the table in the UI). im_idx is the
                    # frame's position in the sorted frame_0000.png, frame_0001.png... list.
                    'per_frame': [
                        {
                            'keyframe': int(r['im_idx']) + 1,   # shown to the user as 1, 2, 3...
                            'base_cvd_base_error': float(r['base_cvd_base_error']),
                            'daltonised_base_error': float(r['daltonised_base_error']),
                            'dalt_boost': float(r['dalt_boost']),
                            'total_energy': float(r['total_energy']),
                        }
                        for _, r in df.sort_values('im_idx').iterrows()
                    ],
                }

        return jsonify(response)

    finally:
        # Remove the uploaded video, the keyframe PNGs and their result files...
        if os.path.exists(video_temp_path):
            os.remove(video_temp_path)
        for p in frame_paths:
            if os.path.exists(p):
                os.remove(p)
            npz_path = os.path.join(results_folder, os.path.basename(p) + '.npz')
            if os.path.exists(npz_path):
                os.remove(npz_path)
        # ...then put your own data/custom images back.
        for f in glob.glob(os.path.join(CUSTOM_DIR, '*.png')):
            os.remove(f)
        for f in glob.glob(os.path.join(BACKUP_DIR, '*.png')):
            shutil.move(f, os.path.join(CUSTOM_DIR, os.path.basename(f)))


# =============================================================================
# 6. SERVER START-UP
# =============================================================================
if __name__ == '__main__':
    # 127.0.0.1 = only this computer can connect (keeps the app private).
    # threaded=False = one request at a time, because data/custom/ is shared.
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=False)
