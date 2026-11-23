"""
Smoke test for the cvd_vit environment. Run this from inside the cvd_vit/
repo root (with its venv active) BEFORE launching a real adjoint_multires_interp.py
job -- it exercises every moving part (DINO checkpoint download + patched
attention hook, the contrast metric, the CVD matrices) in a few seconds,
so setup problems show up immediately instead of after a long training run.

Usage:
    cd cvd_vit
    python smoke_test.py
"""
import os
import sys

import torch

def check(name, fn):
    print(f'--- {name} ---')
    try:
        fn()
        print(f'[OK] {name}')
    except Exception as e:
        print(f'[FAIL] {name}: {e}')
        raise
    print()


def test_cvd_matrices():
    from cvd_matrices import CVD_MATRICES
    for severity in [0.4, 0.6, 1.0]:
        for cvd_type in ['protan', 'deutan', 'tritan']:
            m = CVD_MATRICES[severity][cvd_type]
            assert m.shape == (3, 3), f'bad shape for {severity}/{cvd_type}: {m.shape}'
    print('CVD_MATRICES built for all (severity, cvd_type) combinations, shapes OK')


def test_contrast_metric():
    from adjoint_multires_metrics import get_contrast_metric
    a = torch.rand(2, 3, 64, 64)
    b = torch.rand(2, 3, 64, 64)
    out = get_contrast_metric(a, b)
    print(f'get_contrast_metric output shape: {tuple(out.shape)}')
    assert out.dim() == 3, 'expected (batch, h, w) contrast map'


def test_dino_vit_and_patch():
    sys.path.insert(0, os.path.abspath('dino'))
    import dino.vision_transformer as vits

    model = vits.__dict__['vit_small'](patch_size=8, num_classes=0)
    model.eval()

    url = 'dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth'
    checkpoint_filename = url.split('/')[-1]
    if not os.path.exists(checkpoint_filename):
        print('Downloading DINO ViT-S/8 checkpoint (~80MB, first run only)...')
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/dino/' + url
        )
        torch.save(state_dict, checkpoint_filename)
    else:
        state_dict = torch.load(checkpoint_filename, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    # This method only exists after the DINO patch is applied -- if this
    # AttributeErrors, the patch did not go in correctly.
    assert hasattr(model, 'get_first_n_attn'), (
        'model has no get_first_n_attn -- the DINO patch was not applied'
    )

    x = torch.rand(1, 3, 64, 64)  # must be divisible by patch_size (8)
    attn_maps = model.get_first_n_attn(x, n=4)
    assert len(attn_maps) == 4, f'expected 4 attention maps, got {len(attn_maps)}'
    print(f'get_first_n_attn returned {len(attn_maps)} maps, '
          f'first map shape: {tuple(attn_maps[0].shape)}')


def test_custom_dataset():
    from adjoint_multires_datasets import CustomDataset
    ds = CustomDataset()
    print(f'CustomDataset found {len(ds)} image(s) in data/custom/ and data/swin_t_sample/')
    if len(ds) == 0:
        print('  (put a few .png files in data/custom/ before running the real job)')


def test_video_encoder():
    # Video mode writes an H.264 MP4 through ffmpeg. This checks ffmpeg exists and can encode H.264.
    import shutil
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which('ffmpeg')
    assert ffmpeg, 'ffmpeg not found -- run: pip install imageio-ffmpeg'
    out = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    assert 'libx264' in out, 'this ffmpeg has no libx264 encoder (needed for browser-playable video)'
    print(f'ffmpeg with libx264 found: {ffmpeg}')


if __name__ == '__main__':
    check('CVD matrices', test_cvd_matrices)
    check('Contrast metric', test_contrast_metric)
    check('Custom dataset loader', test_custom_dataset)
    check('Video encoder (ffmpeg + H.264)', test_video_encoder)
    check('DINO ViT + patched attention hook (downloads checkpoint on first run)', test_dino_vit_and_patch)
    print('All smoke tests passed -- environment looks ready.')
