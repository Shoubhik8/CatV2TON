import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from torchvision.io import read_video, write_video
from tqdm import tqdm

from data.utils import paste_back
from modules.cloth_masker import AutoMasker
from modules.pipeline import V2TONPipeline


VALID_CATEGORIES = {"upper", "lower", "overall", "inner", "outer"}
PRECISION_DTYPES = {
    "no": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CatV2TON inference on user-supplied person image/video + garments."
    )
    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument(
        "--person",
        type=str,
        required=True,
        help="Path to the person image (mode=image) or person video (mode=video).",
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
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument(
        "--no_auto_crop",
        action="store_true",
        help="Disable automatic person detection + crop (image mode). By default the person is "
        "detected and tightly cropped (padded to 3:4) so in-the-wild full-frame photos match the "
        "model's expected framing, then the result is pasted back into the original photo.",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Default: 30 for image, 20 for video.",
    )
    parser.add_argument("--guidance_scale", type=float, default=3.0)

    parser.add_argument("--slice_frames", type=int, default=24, help="Video only.")
    parser.add_argument("--pre_frames", type=int, default=8, help="Video only.")
    parser.add_argument(
        "--no_adacn",
        action="store_true",
        help="Video only. Disable Adaptive Clip Normalization between chunks.",
    )

    args = parser.parse_args()

    if len(args.garments) != len(args.categories):
        parser.error(
            f"--garments has {len(args.garments)} entries but --categories has "
            f"{len(args.categories)}. They must be parallel lists."
        )
    bad = [c for c in args.categories if c not in VALID_CATEGORIES]
    if bad:
        parser.error(f"Invalid categories: {bad}. Allowed: {sorted(VALID_CATEGORIES)}")

    if not os.path.isfile(args.person):
        parser.error(f"--person not found: {args.person}")
    for g in args.garments:
        if not os.path.isfile(g):
            parser.error(f"garment not found: {g}")

    if args.num_inference_steps is None:
        args.num_inference_steps = 30 if args.mode == "image" else 20

    return args


def resolve_ckpt(path_or_repo: str) -> str:
    return path_or_repo if os.path.exists(path_or_repo) else snapshot_download(path_or_repo)


def build_pipeline_and_masker(args):
    base = resolve_ckpt(args.base_model_path)
    catv2ton_root = resolve_ckpt(args.catv2ton_ckpt_path)
    catvton_root = resolve_ckpt(args.catvton_ckpt_path)
    finetuned = os.path.join(catv2ton_root, "512-64K")

    dtype = PRECISION_DTYPES[args.mixed_precision]
    pipeline = V2TONPipeline(
        base_model_path=base,
        finetuned_model_path=finetuned,
        load_pose=True,
        torch_dtype=dtype,
        device="cuda",
    )
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(catvton_root, "DensePose"),
        schp_ckpt=os.path.join(catvton_root, "SCHP"),
        device="cuda",
    )
    return pipeline, automasker


def output_path(out_dir: str, person_path: str, garment_path: str, category: str, ext: str) -> str:
    stem_p = Path(person_path).stem
    stem_g = Path(garment_path).stem
    return os.path.join(out_dir, f"{stem_p}__{stem_g}__{category}.{ext}")


# -------------------- Image mode --------------------

def image_repaint(person_pil: Image.Image, mask_pil: Image.Image, result_pil: Image.Image) -> Image.Image:
    m = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
    m = m[..., None]
    p = np.array(person_pil).astype(np.float32)
    r = np.array(result_pil).astype(np.float32)
    blended = p * (1.0 - m) + r * m
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def run_image(args, pipeline: V2TONPipeline, automasker: AutoMasker):
    os.makedirs(args.output_dir, exist_ok=True)
    size = (args.width, args.height)  # PIL.resize expects (W, H)

    # exif_transpose honours the camera orientation flag; without it a sideways
    # phone photo is fed to the model squashed/rotated and produces garbage.
    person_full = ImageOps.exif_transpose(Image.open(args.person)).convert("RGB")

    # Detect + tightly crop the person (padded to 3:4) so the body fills the frame;
    # the try-on result is pasted back into person_full before saving.
    crop_box = None
    if not args.no_auto_crop:
        crop_box = automasker.detect_person_box(person_full, aspect_ratio=args.width / args.height)
        if crop_box is None:
            print("WARNING: no person detected; using the full frame (results may be poor).")
    person_pil = (person_full.crop(crop_box) if crop_box is not None else person_full).resize(size, Image.BICUBIC)

    cond_cache = {}
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    for garment_path, category in zip(args.garments, args.categories):
        garment_pil = Image.open(garment_path).convert("RGB").resize(size, Image.BICUBIC)

        if category not in cond_cache:
            cond = automasker(person_pil, mask_type=category)
            mask_pil = cond["mask"].resize(size, Image.NEAREST)
            densepose_pil = cond["densepose"].resize(size, Image.NEAREST)
            cond_cache[category] = (mask_pil, densepose_pil)
        mask_pil, densepose_pil = cond_cache[category]

        results = pipeline.image_try_on(
            source_image=person_pil,
            source_mask=mask_pil,
            conditioned_image=garment_pil,
            pose_image=densepose_pil,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        result_pil = results[0]
        

        if args.repaint:
            result_pil = image_repaint(person_pil, mask_pil, result_pil)

        # Paste the (cropped) try-on result back into the original full-frame photo.
        if crop_box is not None:
            result_pil = paste_back(person_full, result_pil, crop_box)

        out = output_path(args.output_dir, args.person, garment_path, category, "png")
        result_pil.save(out)
        # The fix below is to incorporate multiple images try on at the same time
        # person_pil = result_pil
        print(f"[image] wrote {out}")


# -------------------- Video mode --------------------

def video_repaint(person: torch.Tensor, mask: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """Port of repaint() from eval_video_try_on.py:426. All tensors (B, C, T, H, W)."""
    h = person.size(-1)
    k = max(1, h // 50)
    if k % 2 == 0:
        k += 1
    m = rearrange(mask, "b c f h w -> (b f) c h w")
    m = torch.nn.functional.avg_pool2d(m, k, stride=1, padding=k // 2)
    m = rearrange(m, "(b f) c h w -> b c f h w", b=person.size(0))
    return person * (1 - m) + result * m


def load_person_video(path: str, height: int, width: int):
    """Returns (raw_uint8_TCHW for masker, normalized_BCTHW float in [-1,1] for pipeline)."""
    video = read_video(path, pts_unit="sec", output_format="TCHW")[0]  # uint8 (T, C, H_src, W_src)
    if video.size(0) == 0:
        raise RuntimeError(f"Could not decode any frames from {path}")

    resize = T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    video = resize(video)  # still uint8 (T, C, H, W)

    raw_tchw = video.contiguous()  # for AutoMasker
    norm = video.float() / 127.5 - 1.0  # [-1, 1]
    norm = norm.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # (1, C, T, H, W)
    return raw_tchw, norm


def pad_to_multiple_of_4(*tensors):
    """Each tensor: (1, C, T, H, W). Pad T by replicating the last frame."""
    t = tensors[0].size(2)
    rem = t % 4
    if rem == 0:
        return tensors, t
    pad = 4 - rem
    padded = tuple(
        torch.cat([x, x[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2) for x in tensors
    )
    return padded, t


def run_video(args, pipeline: V2TONPipeline, automasker: AutoMasker):
    os.makedirs(args.output_dir, exist_ok=True)

    raw_person_tchw, person_v = load_person_video(args.person, args.height, args.width)
    person_v = person_v.to(pipeline.device, dtype=pipeline.weight_dtype)

    garment_transform = T.Compose([
        T.Resize((args.height, args.width), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    cond_cache = {}
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    for garment_path, category in zip(args.garments, args.categories):
        if category not in cond_cache:
            preprocess = automasker.process_video(
                mask_type=category,
                video_tensor=raw_person_tchw,
                densepose_colormap=cv2.COLORMAP_VIRIDIS,
            )
            # Each is (C, T, H, W) uint8.
            mask_u8 = preprocess["mask"]
            dp_u8 = preprocess["densepose"]
            mask_v = (mask_u8.float() / 255.0).unsqueeze(0)  # [0, 1]
            dp_v = (dp_u8.float() / 127.5 - 1.0).unsqueeze(0)  # [-1, 1]
            cond_cache[category] = (
                mask_v.to(pipeline.device, dtype=pipeline.weight_dtype),
                dp_v.to(pipeline.device, dtype=pipeline.weight_dtype),
            )
        mask_v, dp_v = cond_cache[category]

        garment_pil = Image.open(garment_path).convert("RGB")
        garment_v = garment_transform(garment_pil).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
        garment_v = garment_v.to(pipeline.device, dtype=pipeline.weight_dtype)

        (person_pad, mask_pad, dp_pad), original_frames = pad_to_multiple_of_4(
            person_v, mask_v, dp_v
        )

        result = pipeline.video_try_on(
            source_video=person_pad,
            mask_video=mask_pad,
            condition_image=garment_v,
            pose_video=dp_pad,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            slice_frames=args.slice_frames,
            pre_frames=args.pre_frames,
            use_adacn=not args.no_adacn,
            generator=generator,
        )  # (B, T, H, W, C) in [-1, 1]
        result = result.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)

        if args.repaint:
            result = video_repaint(person_pad.float(), mask_pad.float(), result.float())

        result = result[:, :, :original_frames]  # crop padding
        frames = (result[0] * 0.5 + 0.5).clamp(0, 1)  # (C, T, H, W) in [0, 1]
        frames = (frames.permute(1, 2, 3, 0).cpu().float() * 255.0).clamp(0, 255).to(torch.uint8)

        out = output_path(args.output_dir, args.person, garment_path, category, "mp4")
        write_video(out, frames, fps=24)
        print(f"[video] wrote {out}")


def main():
    args = parse_args()
    pipeline, automasker = build_pipeline_and_masker(args)
    if args.mode == "image":
        run_image(args, pipeline, automasker)
    else:
        run_video(args, pipeline, automasker)


if __name__ == "__main__":
    main()
