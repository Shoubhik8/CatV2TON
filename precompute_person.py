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
from PIL import Image
from torchvision.io import read_video, write_video
from tqdm import tqdm

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
        description="precomputing all person related tensors for faster inference"
    )
    parser.add_argument(
        "--person",
        type=str,
        required=True,
        help="Path to the person image (mode=image) or person video (mode=video).",
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