"""CatV2TON image try-on from PRECOMPUTED person latents.

Unlike `inference.py`, this script does **not** run AutoMasker (DensePose + SCHP)
and does **not** VAE-encode the person/mask/pose. Those person-side latents are read
straight from a cache produced by `precompute_person.py`. Only the garment is encoded
on the fly, after which we drive `V2TONPipeline.denoising` + `decode_latents` directly
(`image_try_on` cannot accept pre-encoded latents).

The garment-side encoding here is reproduced verbatim from `image_try_on`
(modules/pipeline.py:460-473), so for a given cache + seed the result matches the
on-the-fly path.

Example:
    CUDA_VISIBLE_DEVICES=0 python inference_v2.py \
        --cache ./precomputed/person_precomputed.pt \
        --garments g1.jpg g2.jpg --categories upper lower \
        --output_dir ./out_v2 --repaint
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

from modules.pipeline import V2TONPipeline, prepare_image


VALID_CATEGORIES = {"upper", "lower", "overall", "inner", "outer"}
PRECISION_DTYPES = {
    "no": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CatV2TON image try-on using precomputed person-side latents."
    )
    parser.add_argument(
        "--cache",
        type=str,
        required=True,
        help="Path to the '<person>_precomputed.pt' file from precompute_person.py.",
    )
    parser.add_argument(
        "--garments",
        type=str,
        nargs="+",
        required=True,
        help="One or more garment image paths.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        required=True,
        help=f"Parallel list of categories, one per garment. Each in {sorted(VALID_CATEGORIES)}.",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--base_model_path", type=str, default="alibaba-pai/EasyAnimateV4-XL-2-InP")
    parser.add_argument("--catv2ton_ckpt_path", type=str, default="zhengchong/CatV2TON")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=list(PRECISION_DTYPES), default="bf16")
    parser.add_argument("--repaint", action="store_true")

    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.0)

    args = parser.parse_args()

    if len(args.garments) != len(args.categories):
        parser.error(
            f"--garments has {len(args.garments)} entries but --categories has "
            f"{len(args.categories)}. They must be parallel lists."
        )
    bad = [c for c in args.categories if c not in VALID_CATEGORIES]
    if bad:
        parser.error(f"Invalid categories: {bad}. Allowed: {sorted(VALID_CATEGORIES)}")

    if not os.path.isfile(args.cache):
        parser.error(f"--cache not found: {args.cache}")
    for g in args.garments:
        if not os.path.isfile(g):
            parser.error(f"garment not found: {g}")

    return args


def resolve_ckpt(path_or_repo: str) -> str:
    return path_or_repo if os.path.exists(path_or_repo) else snapshot_download(path_or_repo)


def build_pipeline(args) -> V2TONPipeline:
    base = resolve_ckpt(args.base_model_path)
    catv2ton_root = resolve_ckpt(args.catv2ton_ckpt_path)
    finetuned = os.path.join(catv2ton_root, "512-64K")
    return V2TONPipeline(
        base_model_path=base,
        finetuned_model_path=finetuned,
        load_pose=True,
        torch_dtype=PRECISION_DTYPES[args.mixed_precision],
        device="cuda",
    )


def output_path(out_dir: str, person_path: str, garment_path: str, category: str, ext: str) -> str:
    stem_p = Path(person_path).stem
    stem_g = Path(garment_path).stem
    return os.path.join(out_dir, f"{stem_p}__{stem_g}__{category}.{ext}")


def image_repaint(person_pil: Image.Image, mask_pil: Image.Image, result_pil: Image.Image) -> Image.Image:
    """Blend the result onto the original person outside the masked region (port of inference.py)."""
    m = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
    m = m[..., None]
    p = np.array(person_pil).astype(np.float32)
    r = np.array(result_pil).astype(np.float32)
    blended = p * (1.0 - m) + r * m
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def latents_to_pil(pipeline: V2TONPipeline, output_latents: torch.Tensor):
    """Decode denoised latents to a list of PIL images (replicates image_try_on:492-503)."""
    output_images = pipeline.decode_latents(output_latents)
    output_images = output_images.squeeze(2).permute(0, 2, 3, 1).cpu().numpy()
    output_images = (output_images * 0.5 + 0.5).clip(0, 1)
    return [Image.fromarray((img * 255).astype(np.uint8)) for img in output_images]


@torch.no_grad()
def run(args, pipeline: V2TONPipeline, cache: dict):
    os.makedirs(args.output_dir, exist_ok=True)
    device, dtype = pipeline.device, pipeline.weight_dtype

    meta = cache["meta"]
    size = (meta["width"], meta["height"])  # PIL.resize expects (W, H)
    person_path = meta["person"]

    # pose_latents are category-independent and shared across all garments.
    pose_latents = cache["pose_latents"].to(device, dtype)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    for garment_path, category in zip(args.garments, args.categories):
        entry = cache["categories"][category]
        masked_latents = entry["masked_latents"].to(device, dtype)
        mask_latents = entry["mask_latents"].to(device, dtype)

        # --- Garment-side encoding (the only VAE encode we still do at inference) ---
        garment_pil = ImageOps.exif_transpose(Image.open(garment_path)).convert("RGB").resize(size, Image.BICUBIC)
        conditioned_image = prepare_image(garment_pil, device, dtype=dtype)
        conditioned_latents = pipeline._slice_vae(conditioned_image)
        conditioning_mask_latents = pipeline._slice_vae(torch.zeros_like(conditioned_image))

        # --- Denoise using cached person-side latents + fresh garment latents ---
        output_latents = pipeline.denoising(
            masked_latents=masked_latents,
            mask_latents=mask_latents,
            conditioned_latents=conditioned_latents,
            conditioning_mask_latents=conditioning_mask_latents,
            pose_latents=pose_latents,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        result_pil = latents_to_pil(pipeline, output_latents)[0]

        if args.repaint:
            person_pil = Image.fromarray(cache["person_image"])
            mask_pil = Image.fromarray(entry["mask_image"])
            result_pil = image_repaint(person_pil, mask_pil, result_pil)

        out = output_path(args.output_dir, person_path, garment_path, category, "png")
        result_pil.save(out)
        print(f"[image-v2] wrote {out}")


def main():
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu")

    missing = [c for c in args.categories if c not in cache.get("categories", {})]
    if missing:
        available = sorted(cache.get("categories", {}).keys())
        raise SystemExit(
            f"Requested categories {missing} are not in the cache. Available: {available}"
        )
    if args.repaint and "person_image" not in cache:
        raise SystemExit(
            "--repaint needs pixel data that this cache predates. "
            "Regenerate it with the current precompute_person.py (adds person_image / mask_image)."
        )

    pipeline = build_pipeline(args)
    run(args, pipeline, cache)


if __name__ == "__main__":
    main()
