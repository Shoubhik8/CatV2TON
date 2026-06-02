"""Pre-compute and cache the VAE-encoded person-side tensors for fast inference.

For a single person image this computes, for every garment category, the latents
that depend only on the person (not on the garment being tried on):

  - ``masked_latents`` : VAE latents of the person with the agnostic region masked out
  - ``mask_latents``   : VAE latents of the (max-pooled) agnostic mask

The densepose / pose latents do **not** depend on the category, so they are
computed once and stored at the top level of the cache.

The computations here intentionally mirror ``V2TONPipeline.image_try_on`` so the
cached tensors are drop-in replacements for what that method produces on the fly.
At inference time, load the cache and pick the per-category entry whose category
matches the garment, then feed ``masked_latents`` / ``mask_latents`` / ``pose_latents``
straight into ``V2TONPipeline.denoising`` (the garment-side ``conditioned_latents``
and ``conditioning_mask_latents`` still have to be encoded from the cloth image).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from torch.nn import functional as F
from tqdm import tqdm

from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from modules.cloth_masker import AutoMasker
from modules.pipeline import prepare_densepose, prepare_image


# Every category AutoMasker can produce. We precompute one cache entry per category.
VALID_CATEGORIES = ["upper", "lower", "overall", "inner", "outer"]
PRECISION_DTYPES = {
    "no": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute and cache the VAE-encoded person/mask tensors, one entry per category."
    )
    parser.add_argument(
        "--person",
        type=str,
        required=True,
        help="Path to the person image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory the '<person>_precomputed.pt' cache is written to.",
    )
    parser.add_argument("--base_model_path", type=str, default="alibaba-pai/EasyAnimateV4-XL-2-InP")
    parser.add_argument(
        "--catvton_ckpt_path",
        type=str,
        default="zhengchong/CatVTON",
        help="Provides DensePose/ and SCHP/ subdirs used by AutoMasker.",
    )

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=list(PRECISION_DTYPES), default="bf16")
    return parser.parse_args()


@torch.no_grad()
def slice_vae(vae, pixel_values, device, weight_dtype):
    """VAE-encode pixel values to latents (mirrors ``V2TONPipeline._slice_vae``)."""
    if pixel_values.size(1) == 4:
        return pixel_values.to(device, weight_dtype)
    bs = pixel_values.shape[0]  # FIXME: Not use mini_batch
    new_pixel_values = []
    for i in range(0, bs, bs):
        pixel_values_bs = pixel_values[i : i + bs]
        pixel_values_bs = vae.encode(pixel_values_bs.to(device, weight_dtype))[0]
        pixel_values_bs = pixel_values_bs.sample()
        new_pixel_values.append(pixel_values_bs)
    return torch.cat(new_pixel_values, dim=0) * vae.config.scaling_factor


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda"
    weight_dtype = PRECISION_DTYPES[args.mixed_precision]
    size = (args.width, args.height)  # PIL resize expects (width, height)

    if not os.path.isfile(args.person):
        raise FileNotFoundError(f"--person not found: {args.person}")

    # Deterministic VAE sampling so repeated runs produce identical caches.
    torch.manual_seed(args.seed)

    # VAE only — the transformer / posenet are not needed for pre-computation.
    # AutoencoderKLMagvit.from_pretrained does not auto-download from the Hub, so
    # resolve the repo id to a local snapshot first (mirrors the eval scripts).
    base_model_path = (
        snapshot_download(args.base_model_path)
        if not os.path.exists(args.base_model_path)
        else args.base_model_path
    )
    vae = AutoencoderKLMagvit.from_pretrained(base_model_path, subfolder="vae").to(
        device, dtype=weight_dtype
    )
    vae.requires_grad_(False)
    vae.eval()

    # AutoMasker needs the DensePose/ and SCHP/ subdirs shipped in the CatVTON checkpoint.
    catvton_root = (
        snapshot_download(args.catvton_ckpt_path)
        if not os.path.exists(args.catvton_ckpt_path)
        else args.catvton_ckpt_path
    )
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(catvton_root, "DensePose"),
        schp_ckpt=os.path.join(catvton_root, "SCHP"),
        device=device,
    )

    # exif_transpose honours the camera orientation flag; without it a sideways
    # phone photo is fed to the model squashed/rotated and produces garbage.
    person_pil = ImageOps.exif_transpose(Image.open(args.person)).convert("RGB").resize(size, Image.BICUBIC)
    # Person latents and densepose are category-independent — encode them once.
    source_image = prepare_image(person_pil, device, dtype=weight_dtype)

    cache = {
        "meta": {
            "person": os.path.abspath(args.person),
            "height": args.height,
            "width": args.width,
            "mixed_precision": args.mixed_precision,
            "seed": args.seed,
        },
        # Resized RGB person, kept so --repaint works at inference without AutoMasker.
        "person_image": np.array(person_pil),  # uint8 (H, W, 3)
        "categories": {},
    }

    # Pose / densepose latents are the same for every category, so compute once.
    # AutoMasker returns the *gray* densepose; densepose_to_rgb (inside prepare_densepose)
    # converts it the same way inference does.
    densepose_pil = None

    for category in tqdm(VALID_CATEGORIES, desc="Precomputing per-category tensors"):
        cond = automasker(person_pil, mask_type=category)
        mask_pil = cond["mask"].resize(size, Image.NEAREST)
        if densepose_pil is None:
            densepose_pil = cond["densepose"].resize(size, Image.NEAREST)

        # Mask -> [0, 1], clamp, then max-pool exactly as image_try_on does.
        source_mask = prepare_image(mask_pil.convert("RGB"), device, dtype=weight_dtype) * 0.5 + 0.5
        source_mask = source_mask.clamp(0, 1)
        source_mask = F.max_pool2d(source_mask.squeeze(2), kernel_size=11, stride=1, padding=5)
        source_mask = source_mask.unsqueeze(2)

        # Mask out the agnostic region of the person, then VAE-encode.
        masked_image = source_image * (source_mask < 0.5) + -1 * torch.ones_like(source_image) * (
            source_mask >= 0.5
        )
        masked_latents = slice_vae(vae, masked_image, device, weight_dtype)
        mask_latents = slice_vae(vae, source_mask, device, weight_dtype)

        cache["categories"][category] = {
            "masked_latents": masked_latents.cpu(),
            "mask_latents": mask_latents.cpu(),
            # Raw agnostic mask (pre max-pool), kept for --repaint at inference.
            "mask_image": np.array(mask_pil.convert("L")),  # uint8 (H, W)
        }

    # Category-independent pose latents, stored once at the top level.
    pose_image = prepare_densepose(densepose_pil, device, dtype=weight_dtype)
    cache["pose_latents"] = slice_vae(vae, pose_image, device, weight_dtype).cpu()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{Path(args.person).stem}_precomputed.pt")
    torch.save(cache, out_path)
    print(f"Saved precomputed tensors for {len(VALID_CATEGORIES)} categories to {out_path}")


if __name__ == "__main__":
    main()
