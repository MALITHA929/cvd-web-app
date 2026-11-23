"""
Visualize the results of an adjoint_multires_interp.py run: for each image,
shows Original / Original (CVD-simulated) / Recoloured / Recoloured (CVD-simulated),
matching the layout of Fig. 1 in the paper. Also prints the per-image metrics
table from the run's .parquet file.

Usage (run from inside cvd_vit/, with the venv/conda env active):
    python visualize_results.py --dataset_type custom --cvd_type protan --severity 1.0

Output: results/<dataset>/<cvd_type>/severity_<s>/comparison_grid.png
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvd_matrices import CVD_MATRICES

# ---------------------------------------------------------------------------
# 1. Command-line options. They must match the options you gave the optimiser
#    so we look in the same results folder.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='custom')
parser.add_argument('--cvd_type', type=str, default='protan', choices=['protan', 'deutan', 'tritan'])
parser.add_argument('--severity', type=float, default=1.0, choices=[0.4, 0.6, 1.0])
parser.add_argument('--max_images', type=int, default=10, help='Cap how many images to plot')
args = parser.parse_args()

# The optimiser saves its output in this folder (one .npz per image + a .parquet metrics table).
results_folder = os.path.join('results', args.dataset_type, args.cvd_type, f'severity_{args.severity}')
# 3x3 colour-blindness simulation matrix for the chosen type and severity.
matrix = CVD_MATRICES[args.severity][args.cvd_type]  # (3,3)

# ---------------------------------------------------------------------------
# 2. Small helpers
# ---------------------------------------------------------------------------
def simulate_cvd(img_chw):
    # img_chw: (3, H, W) in [0,1] -> CVD-simulated (3, H, W)
    # Multiply every pixel's (R,G,B) by the CVD matrix, keep values in [0,1].
    return np.clip(np.einsum('jhw,mj->mhw', img_chw, matrix), 0, 1)

def to_hwc(img_chw):
    # matplotlib wants (H, W, 3); the optimiser stores (3, H, W).
    return np.clip(img_chw, 0, 1).transpose(1, 2, 0)

# ---------------------------------------------------------------------------
# 3. Print the metrics table
#    base_cvd_base_error    contrast the ORIGINAL loses when seen with CVD
#    daltonised_base_error  contrast the RECOLOURED image loses with CVD
#    dalt_boost             improvement (positive = recolouring helped)
#    total_energy           how much the image was changed
# ---------------------------------------------------------------------------
parquet_files = glob.glob(os.path.join(results_folder, '*.parquet'))
if parquet_files:
    df = pd.read_parquet(parquet_files[0])
    print(f'Loaded metrics for {len(df)} image(s) from {parquet_files[0]}\n')
    print(df[['base_cvd_base_error', 'daltonised_base_error', 'dalt_boost', 'total_energy']].describe())  # summary stats
    print()
    print(df[['base_cvd_base_error', 'daltonised_base_error', 'dalt_boost', 'total_energy']])             # one row per image
else:
    print(f'No .parquet file found in {results_folder} -- did the run actually reach a validation epoch?')

# ---------------------------------------------------------------------------
# 4. Build the comparison grid: one row per image, four columns
# ---------------------------------------------------------------------------
npz_files = sorted(glob.glob(os.path.join(results_folder, '*.npz')))[:args.max_images]
if not npz_files:
    raise SystemExit(f'No .npz result files found in {results_folder}')

fig, axes = plt.subplots(len(npz_files), 4, figsize=(12, 3 * len(npz_files)))
if len(npz_files) == 1:
    axes = axes[None, :]   # matplotlib returns a 1-D array for a single row; make it 2-D

col_titles = ['Original', 'Original (CVD sim)', 'Recoloured', 'Recoloured (CVD sim)']
for row, npz_path in enumerate(npz_files):
    data = np.load(npz_path)
    input_image = data['input_image']  # (3, H, W)  the original
    output_image = data['image']       # (3, H, W)  the recoloured result

    # The four pictures, in column order.
    panels = [
        to_hwc(input_image),
        to_hwc(simulate_cvd(input_image)),
        to_hwc(output_image),
        to_hwc(simulate_cvd(output_image)),
    ]
    for col, (panel, title) in enumerate(zip(panels, col_titles)):
        ax = axes[row, col]
        ax.imshow(panel)
        ax.axis('off')
        if row == 0:   # column titles only on the first row
            ax.set_title(title, fontsize=10)
    axes[row, 0].set_ylabel(os.path.basename(npz_path), fontsize=8)  # file name label

# ---------------------------------------------------------------------------
# 5. Save the figure next to the results
# ---------------------------------------------------------------------------
plt.suptitle(f'{args.dataset_type} / {args.cvd_type} / severity {args.severity}')
plt.tight_layout()
out_path = os.path.join(results_folder, 'comparison_grid.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved comparison grid to {out_path}')
