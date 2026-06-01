# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CatV2TON is a DiT-based method for Vision-Based Virtual Try-On (V2TON) using Temporal Concatenation of video frames with garment conditions. It supports both image try-on (VITONHD, DressCode) and video try-on (ViViD-S, VVT). Paper: arXiv 2501.11325. Pretrained weights live on HuggingFace at `zhengchong/CatV2TON`.

## Installation

Setting this up on a new machine has a few hard constraints that aren't obvious — follow these steps in order. Dependencies are listed in [requirements.txt](requirements.txt).

**1. Create a Python 3.9 environment.** Python 3.9 is *required* (not just recommended): the vendored [detectron2/](detectron2/) ships a prebuilt `_C.cpython-39-x86_64-linux-gnu.so` that only loads on CPython 3.9. Other versions fail with `ModuleNotFoundError: No module named 'detectron2._C'`.
```bash
conda create -n catvton python=3.9 -y
conda activate catvton
```

**2. Make sure pip is compatible with 3.9.** pip >= 26 dropped Python 3.9 and crashes on import (`TypeError: dataclass() got an unexpected keyword argument 'slots'`). If you hit that, downgrade:
```bash
python -m pip install "pip<26"   # if that pip is already broken, bootstrap: conda install -n catvton "pip<26"
```

**3. Install everything via requirements.txt.** torch/torchvision are pinned to `2.1.2+cu121` inside the file (with the PyTorch `--extra-index-url`), so a single command does it all:
```bash
pip install -r requirements.txt
```
- **CUDA:** the file targets `cu121`, which runs on any driver supporting CUDA >= 12.1 (check `nvidia-smi`). For a different CUDA, edit the `--extra-index-url` line AND the `+cu121` tags in [requirements.txt](requirements.txt) (e.g. `cu118`).
- **Do NOT bump torch to >= 2.4.** The detectron2 `.so` was built against `c10::optional`, which PyTorch 2.4 replaced with `std::optional`; newer torch fails with `undefined symbol: ...zeros_like...c10..optional...`. Valid range is `>=2.0,<2.4`.
- The HuggingFace stack (`diffusers==0.31.0`, `transformers==4.46.2`, `accelerate==1.0.1`) is pinned to the EasyAnimate-era APIs the vendored code uses. Newer `transformers` (>= 4.56) drops torch < 2.2 support and crashes on `torch.utils._pytree.register_pytree_node`.

**4. Verify the install** (no GPU or checkpoints needed — just import success):
```bash
python -c "from detectron2 import _C; print('detectron2 _C ok')"
python inference.py --help
```

**5. Run.** Inference needs a CUDA GPU and the HuggingFace checkpoints (`zhengchong/CatV2TON`, `alibaba-pai/EasyAnimateV4-XL-2-InP`), which auto-download on first run via `snapshot_download`. See **Common commands** below.

## Common commands

All entry points are top-level scripts; there is no build system, package manifest, or test suite.

Image try-on inference (VITONHD / DressCode):
```bash
CUDA_VISIBLE_DEVICES=0 python eval_image_try_on.py \
  --dataset vitonhd|dresscode --data_root_path <DATA> --output_dir <OUT> \
  --dataloader_num_workers 8 --batch_size 8 --seed 42 \
  --mixed_precision bf16 --allow_tf32 --repaint --eval_pair
```

Video try-on inference (ViViD / VVT):
```bash
CUDA_VISIBLE_DEVICES=0 python eval_video_try_on.py \
  --dataset vivid|vvt --data_root_path <DATA> --output_dir <OUT> \
  --dataloader_num_workers 8 --batch_size 8 --seed 42 \
  --mixed_precision bf16 --allow_tf32 --repaint --eval_pair
```

Metric evaluation against ground-truth folders:
```bash
python eval_image_metrics.py --gt_folder <GT> --pred_folder <PRED> --batch_size 16 --num_workers 16 --paired
python eval_video_metrics.py --gt_folder <GT> --pred_folder <PRED> --num_workers 16 --paired
```

`--eval_pair` / `--paired` switches between paired (self-reconstruction) and unpaired (cross-garment) evaluation. Image datasets expect `test_pairs_paired.txt` / `test_pairs_unpaired.txt` plus precomputed `densepose_gray/` and `agnostic-mask-new/` folders alongside `image/` and `cloth/`.

## Architecture

The pipeline is a 3D Hunyuan DiT (from EasyAnimate) adapted for try-on by **concatenating frames temporally** with the garment image, then jointly denoising in latent space with a MagVit VAE.

Core flow (see [modules/pipeline.py](modules/pipeline.py)):

1. `init_transformer3d_model` loads `HunyuanTransformer3DModel` from a base checkpoint, **strips unused attention modules** (`attn_temporal`, `attn_clip`, `attn2`, `norm_clip*`, `gate_clip`, `norm2`), then loads only the `attn1` weights of the fine-tuned try-on checkpoint. Most of the original Hunyuan branches (cross-attn to text, CLIP conditioning, temporal attention) are intentionally dead code paths for this model.
2. `V2TONPipeline` runs DDPM denoising over a latent built by **concatenating person, garment, mask, and densepose** along the temporal dimension. `prepare_image` always adds a frame dim (`unsqueeze(2)`) so image and video paths share one tensor layout.
3. `--repaint` blends generated and original pixels using the inpainting mask after VAE decode.
4. Video inference uses overlapping temporal windows; see chunking logic in [eval_video_try_on.py](eval_video_try_on.py).

Conditioning inputs:
- **Mask**: agnostic mask of the clothing region on the person. Image inference reads precomputed `agnostic-mask-new/`; on-the-fly mask generation uses [modules/cloth_masker.py](modules/cloth_masker.py) with SCHP human parsing ([modules/SCHP/](modules/SCHP/)) and DensePose.
- **Pose**: DensePose IUV converted to RGB via `densepose_to_rgb` ([data/utils.py](data/utils.py)) with `cv2.COLORMAP_VIRIDIS`. The `DensePose` wrapper in [modules/densepose.py](modules/densepose.py) drives the vendored detectron2 + densepose model.

### Vendored third-party code

These directories are forks copied into the repo, not pip installs — edits here are intentional and `pip install detectron2` will not substitute:
- [detectron2/](detectron2/) — includes a prebuilt `_C.cpython-39-x86_64-linux-gnu.so`, so Python 3.9 on Linux x86_64 is effectively required unless you rebuild the extension.
- [densepose/](densepose/) — DensePose model used for pose conditioning.
- [easyanimate/](easyanimate/) — provides `AutoencoderKLMagvit` and `HunyuanTransformer3DModel` imported by the pipeline.
- [modules/SCHP/](modules/SCHP/) — Self-Correction Human Parsing for mask synthesis.
- [modules/fid_metrics/](modules/fid_metrics/) — FID/FVD/SSIM/LPIPS implementations used by the eval scripts.

### Checkpoints

`eval_image_try_on.py` / `eval_video_try_on.py` call `huggingface_hub.snapshot_download` to pull `zhengchong/CatV2TON`. Two model sizes (256 and 512) are published; pick via the script's resolution flags.
