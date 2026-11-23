# Colour-vision-deficiency recolouring: local web app

A small local web app around the method from
**"Image Adaptation for Colour Vision Deficient Viewers Using Vision Transformers"**
(Gillooly, Thomas, Hardeberg, Guarnera, WACV 2025).

Upload a photo or a short video clip, pick a type and severity of colour vision
deficiency (protanopia, deuteranopia, tritanopia), and the app recolours it so
the content stays distinguishable when viewed with that deficiency. A 2x2
comparison shows the original and recoloured versions, each as seen normally
and as seen with the simulated deficiency.

This repository contains **only our additions**. The paper's code is not
copied here; `setup.sh` downloads it from its original repo
(<https://github.com/tomgillooly/cvd_vit>, which has no licence file, so we do
not redistribute it).

## What is ours and what is the paper's

| | |
|---|---|
| Paper's method (attention-guided optimisation, metric) | original authors, downloaded by `setup.sh` |
| `overlay/app.py`, `overlay/static/index.html` | web app (image mode and video mode) |
| `overlay/cvd_matrices.py` | rebuilds the `cvd_matrices` module the paper's code imports but doesn't ship |
| `dino_fixed.patch` | the paper's `dino.patch` has no file headers and fails to apply; this is the same change with valid headers |
| `overlay/smoke_test.py`, `overlay/visualize_results.py` | environment check and result grids |

`setup.sh` also adds `'custom'` to the `--dataset_type` choices in the paper's
`adjoint_multires_interp.py`, which the script supports internally but doesn't
list.

## Requirements

- An **NVIDIA GPU with CUDA** (developed on an RTX 2050 with 4 GB VRAM). The
  optimisation step will not run without one.
- **Python 3.9-3.11** (the pinned `torch==2.0.1` has no wheels for newer Python).
  Conda is the easiest way to get 3.10: `conda create -n cvdvit python=3.10 -y`
- `git`, and roughly 6 GB of disk (PyTorch and CUDA libraries are large).
- Windows users: use **WSL2** with Ubuntu, and update the NVIDIA driver on
  Windows (not inside WSL).

## Setup

```bash
git clone <this repo>
cd <this repo>
conda activate cvdvit        # or let setup.sh create ./venv
bash setup.sh
```

The `torch` download is about 620 MB. If it fails partway with an SSL error
(common on WSL2), just re-run `bash setup.sh`; pip resumes from its cache.

Then check everything works (downloads the ~80 MB DINO weights the first time):

```bash
bash run.sh --check
```

## Run

```bash
bash run.sh
```

Open <http://127.0.0.1:5000>.

- **Image tab:** upload a photo, choose type, severity and number of optimisation
  steps (about a minute on a small GPU).
- **Video tab:** see below.

Stop the server with Ctrl+C.

## How video mode works (our extension, not in the paper)

The paper only handles single images and lists video as future work. Running
the optimisation on every frame would take hours, and optimising frames
independently would flicker. So:

1. A small number of **keyframes** are sampled and recoloured with the paper's
   GPU optimisation.
2. A **3D colour table** is fitted from the before/after pixels of those
   keyframes. It stores the average colour *shift* for each region of colour
   space. Colours never seen are filled in from nearby ones, defaulting to
   "no shift", and the table is interpolated smoothly so slowly changing
   lighting does not cause banding or shimmer.
3. The shift is added to **every frame** of the clip (up to 30 fps and 30
   seconds), which is cheap and uses no GPU, and the frames are encoded into a
   normal H.264 MP4 (via the bundled `imageio-ffmpeg`) that plays smoothly in
   any browser.

The same colour always maps to the same output, so the result is temporally
consistent. The MP4 keeps the clip's own proportions and shows the 2x2
comparison; there is a download link under the player.

**Trade-off:** the paper's method is spatially aware (it can treat the same
red differently on the salient subject and on the background). A colour table
cannot, so video mode recolours every pixel of a given colour identically.
Clips with very uniform colour give the table little to learn from.

## Limitations

- Single user, one job at a time. The app temporarily uses `data/custom/` and
  restores it afterwards.
- No authentication. It binds to `127.0.0.1` only; don't expose it to a network.
- Results vary per image: in the paper's own examples (and ours) some images
  need no change, and a few get slightly *worse* contrast than the original.
  The `dalt_boost` value shown after each run says which.
- Tritanopia uses the shift approximation of the Machado et al. (2009) model.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named pkg_resources` | `pip install "setuptools<82"` |
| `Could not load library libcudnn... libcuda.so` (WSL2) | `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` (`run.sh` and `setup.sh` already do this) |
| pip `SSL: record layer failure` on `torch` | re-run `bash setup.sh`; or set `networkingMode=mirrored` in `%UserProfile%\.wslconfig` and `wsl --shutdown` |
| CUDA out of memory | lower the number of keyframes; images are processed one at a time |
| `ffmpeg was not found` (video mode) | `pip install imageio-ffmpeg` |
| Video won't play in the page | use the *Download the MP4* link; if it still won't play, send the `ffprobe` output of the file |
| `nvidia-smi` not found in WSL | update the NVIDIA driver on Windows, then `wsl --shutdown` |

## Credit

All credit for the method goes to the paper's authors. If you use this, please
cite them:

```
@InProceedings{Gillooly_2025_WACV,
    author    = {Gillooly, Thomas and Thomas, Jean-Baptiste and Hardeberg, Jon Yngve and Guarnera, Giuseppe Claudio},
    title     = {Image Adaptation for Colour Vision Deficient Viewers Using Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
}
```

DINO is by Facebook Research (<https://github.com/facebookresearch/dino>, Apache-2.0).
