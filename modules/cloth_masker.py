import os
from PIL import Image
from typing import Union
import numpy as np
import cv2
from diffusers.image_processor import VaeImageProcessor
import torch
from tqdm import tqdm

from modules.SCHP import SCHP  # type: ignore
from modules.densepose import DensePose, densepose_to_rgb  # type: ignore
from data.utils import pad_box_to_aspect, read_video_frames


DENSE_INDEX_MAP = {
    "background": [0],
    "torso": [1, 2],
    "right hand": [3],
    "left hand": [4],
    "right foot": [5],
    "left foot": [6],
    "right thigh": [7, 9],
    "left thigh": [8, 10],
    "right leg": [11, 13],
    "left leg": [12, 14],
    "left big arm": [15, 17],
    "right big arm": [16, 18],
    "left forearm": [19, 21],
    "right forearm": [20, 22],
    "face": [23, 24],
    "thighs": [7, 8, 9, 10],
    "legs": [11, 12, 13, 14],
    "hands": [3, 4],
    "feet": [5, 6],
    "big arms": [15, 16, 17, 18],
    "forearms": [19, 20, 21, 22],
}
ATR_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3, 
    'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7,
    'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11, 
    'Left-leg': 12, 'Right-leg': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Bag': 16, 'Scarf': 17
}
LIP_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3, 
    'Sunglasses': 4, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
    'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11, 
    'Skirt': 12, 'Face': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18, 'Right-shoe': 19
}

PROTECT_BODY_PARTS = {
    'upper': ['Left-leg', 'Right-leg'],
    'lower': ['Right-arm', 'Left-arm', 'Face'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg'],
    'outer': ['Left-leg', 'Right-leg'],
}
PROTECT_CLOTH_PARTS = {
    'upper': {
        'ATR': ['Skirt', 'Pants'],
        'LIP': ['Skirt', 'Pants']
    },
    'lower': {
        'ATR': ['Upper-clothes'],
        'LIP': ['Upper-clothes', 'Coat']
    },
    'overall': {
        'ATR': [],
        'LIP': []
    },
    'inner': {
        'ATR': ['Dress', 'Coat', 'Skirt', 'Pants'],
        'LIP': ['Dress', 'Coat', 'Skirt', 'Pants', 'Jumpsuits']
    },
    'outer': {
        'ATR': ['Dress', 'Pants', 'Skirt'],
        'LIP': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Jumpsuits']
    }
}
MASK_CLOTH_PARTS = {
    'upper': ['Upper-clothes', 'Coat', 'Dress', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',]
}
MASK_DENSE_PARTS = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}
    
schp_public_protect_parts = ['Hat', 'Hair', 'Sunglasses', 'Left-shoe', 'Right-shoe', 'Bag', 'Glove', 'Scarf']
schp_protect_parts = {
    'upper': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits'],  
    'lower': ['Left-arm', 'Right-arm', 'Upper-clothes', 'Coat'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits', 'Coat'],
    'outer': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits', 'Upper-clothes']
}
schp_mask_parts = {
    'upper': ['Upper-clothes', 'Dress', 'Coat', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits', 'socks'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits', 'socks'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',]
}
dense_mask_parts = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}

def vis_mask(image, mask):
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    mask = mask / 255
    return Image.fromarray((image * (1 - mask)).astype(np.uint8))

def part_mask_of(part: Union[str, list],
                 parse: np.ndarray, mapping: dict):
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for _ in part:
        if _ not in mapping:
            continue
        if isinstance(mapping[_], list):
            for i in mapping[_]:
                mask += (parse == i)
        else:
            mask += (parse == mapping[_])
    return mask

def hull_mask(mask_area: np.ndarray):
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        hull_mask = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | hull_mask
    return hull_mask

def smooth_video_mask(
    video: torch.Tensor,
    space_kernal_size: int = 15,
    time_kernal_size: int = 7,
    device: str = 'cuda',
):
    """Smooth video mask
    Args:
        video (torch.Tensor): (C, T, H, W)
        space_kernal_size (int): kernel size for spatial smoothing
        time_kernal_size (int): kernel size for temporal smoothing
        device (str): device to move video to
    Returns:
        torch.Tensor: (C, T, H, W)
    """
    space_padding = space_kernal_size // 2
    time_padding = time_kernal_size // 2
    video = video.to(device)
    # video = video.permute(1, 0, 2, 3)  # CTHW
    video = video[0].float() / 255  # THW
    # 2D 卷积扩大 mask
    video = torch.nn.functional.avg_pool2d(video, kernel_size=(space_kernal_size, space_kernal_size), stride=(1, 1), padding=(space_padding, space_padding))   
    # 1D 卷积时序平滑 mask
    video = video.permute(1, 2, 0)  # HWT
    video = torch.nn.functional.avg_pool1d(video, kernel_size=time_kernal_size, stride=1, padding=time_padding).unsqueeze(0).repeat(3, 1, 1, 1)  # CHWT
    # 二值化 mask
    video[video > 0.5] = 1
    video[video <= 0.5] = 0
    video = (video * 255).byte()
    video = video.permute(0, 3, 1, 2).to('cpu') # to 
    return video
     

class AutoMasker:
    def __init__(
        self, 
        densepose_ckpt='./Models/DensePose', 
        schp_ckpt='./Models/SCHP', 
        device='cuda'):
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        self.densepose_processor = DensePose(densepose_ckpt, device)
        self.schp_processor_atr = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908301523-atr.pth'), device=device)
        self.schp_processor_lip = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908261155-lip.pth'), device=device)
        
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    def process_densepose(self, image_or_path):
        return self.densepose_processor(image_or_path, resize=1024)

    def process_schp_lip(self, image_or_path):
        return self.schp_processor_lip(image_or_path)

    def process_schp_atr(self, image_or_path):
        return self.schp_processor_atr(image_or_path)

    @torch.no_grad()
    def detect_person_box(self, image_pil: Image.Image, margin: float = 0.15, aspect_ratio: float = 3 / 4):
        """Detect the main person and return a crop box padded to `aspect_ratio` (w/h).

        Reuses the DensePose DefaultPredictor already loaded for masking (no extra model).
        The highest-scoring detection is expanded by `margin` on each side, padded to the
        target aspect ratio, and clamped to the image. Returns (left, top, right, bottom)
        ints, or None if nothing is detected (caller should fall back to the full frame).
        """
        img = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)  # predictor expects BGR
        instances = self.densepose_processor.predictor(img)["instances"]
        if len(instances) == 0 or not instances.has("pred_boxes"):
            return None

        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy() if instances.has("scores") else np.ones(len(boxes))
        left, top, right, bottom = boxes[int(scores.argmax())]

        w, h = image_pil.size
        bw, bh = right - left, bottom - top
        left -= margin * bw
        right += margin * bw
        top -= margin * bh
        bottom += margin * bh
        left, top = max(left, 0), max(top, 0)
        right, bottom = min(right, w), min(bottom, h)
        return pad_box_to_aspect((left, top, right, bottom), w, h, aspect_ratio)


    def preprocess_image(self, image_or_path):
        return {
            'densepose': self.densepose_processor(image_or_path, resize=1024),
            'schp_atr': self.schp_processor_atr(image_or_path),
            'schp_lip': self.schp_processor_lip(image_or_path)
        }
    
    @staticmethod
    def cloth_agnostic_mask(
        densepose_mask: Image.Image,
        schp_lip_mask: Image.Image,
        schp_atr_mask: Image.Image,
        part: str='overall',
        low_resolution_protect: bool=False,
        **kwargs
    ):
        assert part in ['upper', 'lower', 'overall', 'inner', 'outer'], f"part should be one of ['upper', 'lower', 'overall', 'inner', 'outer'], but got {part}"
        w, h = densepose_mask.size
        
        dilate_kernel = max(w, h) // 250
        dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
        dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        
        kernal_size = max(w, h) // 25
        kernal_size = kernal_size if kernal_size % 2 == 1 else kernal_size + 1
        
        densepose_mask = np.array(densepose_mask)
        schp_lip_mask = np.array(schp_lip_mask)
        schp_atr_mask = np.array(schp_atr_mask)
        
        # Strong Protect Area (Hands, Face, Accessory, Feet)
        hands_protect_area = part_mask_of(['hands', 'feet'], densepose_mask, DENSE_INDEX_MAP)
        hands_protect_area = cv2.dilate(hands_protect_area, dilate_kernel, iterations=1)
        hands_protect_area = hands_protect_area & \
            (part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_atr_mask, ATR_MAPPING) | \
             part_mask_of(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'], schp_lip_mask, LIP_MAPPING))
        face_protect_area = part_mask_of('Face', schp_lip_mask, LIP_MAPPING)

        strong_protect_area = hands_protect_area | face_protect_area 

        # Weak Protect Area (Hair, Irrelevant Clothes, Body Parts)
        body_protect_area = part_mask_of(PROTECT_BODY_PARTS[part], schp_lip_mask, LIP_MAPPING) | part_mask_of(PROTECT_BODY_PARTS[part], schp_atr_mask, ATR_MAPPING)
        hair_protect_area = part_mask_of(['Hair'], schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(['Hair'], schp_atr_mask, ATR_MAPPING)
        cloth_protect_area = part_mask_of(PROTECT_CLOTH_PARTS[part]['LIP'], schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(PROTECT_CLOTH_PARTS[part]['ATR'], schp_atr_mask, ATR_MAPPING)
        accessory_protect_area = part_mask_of((accessory_parts := ['Hat', 'Glove', 'Sunglasses', 'Bag', 'Left-shoe', 'Right-shoe', 'Scarf', 'Socks']), schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(accessory_parts, schp_atr_mask, ATR_MAPPING) 
        weak_protect_area = body_protect_area | cloth_protect_area | hair_protect_area | strong_protect_area | accessory_protect_area
        
        # Mask Area
        strong_mask_area = part_mask_of(MASK_CLOTH_PARTS[part], schp_lip_mask, LIP_MAPPING) | \
            part_mask_of(MASK_CLOTH_PARTS[part], schp_atr_mask, ATR_MAPPING)
        background_area = part_mask_of(['Background'], schp_lip_mask, LIP_MAPPING) & part_mask_of(['Background'], schp_atr_mask, ATR_MAPPING)
        mask_dense_area = part_mask_of(MASK_DENSE_PARTS[part], densepose_mask, DENSE_INDEX_MAP)
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        mask_dense_area = cv2.dilate(mask_dense_area, dilate_kernel, iterations=2)
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


        mask_area = (np.ones_like(densepose_mask) & (~weak_protect_area) & (~background_area)) | mask_dense_area

        mask_area = hull_mask(mask_area * 255) // 255  # Convex Hull to expand the mask area
        mask_area = mask_area & (~weak_protect_area)
        mask_area = cv2.GaussianBlur(mask_area * 255, (kernal_size, kernal_size), 0)
        mask_area[mask_area < 25] = 0
        mask_area[mask_area >= 25] = 1
        mask_area = (mask_area | strong_mask_area) & (~strong_protect_area) 
        mask_area = cv2.dilate(mask_area, dilate_kernel, iterations=1)
        
        if low_resolution_protect:  # to avoid dilation too much especially in low resolution
            mask_area = (mask_area | strong_mask_area) & (~strong_protect_area) 
        
        return Image.fromarray(mask_area * 255)
    
    def process_video(
        self, 
        mask_type: str='upper',
        video_path: str=None,
        video_tensor: torch.Tensor=None, # (T, C, H, W)
        densepose_colormap: int=None,
        **kwargs
    ):
        """Process video to get mask and densepose

        Args:
            mask_type (str, optional): mask type. Defaults to 'upper'.
            video_path (str, optional): video path. Defaults to None.
            video_tensor (torch.Tensor, optional): video tensor. Defaults to None.

        Raises:
            ValueError: video_path and video_tensor cannot be both None
            ValueError: video_tensor should have 4 dimensions (T, C, H, W)
        Returns:
            dict: mask and densepose, (C, T, H, W) in type uint8
        """
        if video_tensor is None and not video_path.endswith('.mp4'):
            return self.__call__(video_path, mask_type=mask_type)
        # if both video_path and video_tensor are provided, video_path will be ignored
        assert video_path is not None or video_tensor is not None, "video_path and video_tensor cannot be both None"
        if video_tensor is None:
            video_tensor = read_video_frames(video_path, normalize=False).squeeze(0).permute(1, 0, 2, 3)
        else:
            assert video_tensor.dim() == 4, "video_tensor should have 4 dimensions (T, C, H, W)"
            if -1 <= video_tensor.min() < 0 and video_tensor.max() < 1:
                video_tensor = (video_tensor + 1) / 2
            
        masks = []
        denseposes = []
        for frame_idx in tqdm(range(video_tensor.size(0)), desc="Processing video masks"):
            frame_tensor = video_tensor[frame_idx]
            if frame_tensor.dim() != 3:
                raise ValueError(f"Expected frame_tensor to have 3 dimensions (C,H,W), but got shape {frame_tensor.shape}")
            if frame_tensor.shape[0] != 3:
                raise ValueError(f"Expected frame_tensor to have 3 channels, but got {frame_tensor.shape[0]} channels")
            
            frame_pil = Image.fromarray(
                (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            results = self.__call__(frame_pil, mask_type=mask_type)
            masks.append(results['mask'])
            denseposes.append(
                densepose_to_rgb(results['densepose'], colormap=densepose_colormap) 
                if densepose_colormap is not None else results['densepose'].convert('RGB')
            )
        # pil list to tensor(T, H, W)
        masks = torch.stack([torch.from_numpy(np.array(mask)) for mask in masks])
        masks = masks.unsqueeze(1).repeat(1, 3, 1, 1)  # to (T, C, H, W) RGB
        masks = masks.permute(1, 0, 2, 3)  # to (C, T, H, W)
        masks = smooth_video_mask(masks)  # (C, T, H, W)
        denseposes = torch.stack([torch.from_numpy(np.array(densepose)) for densepose in denseposes])  # (T, H, W, C)
        denseposes = denseposes.permute(3, 0, 1, 2)  # (C, T, H, W)
        return {
            'mask': masks,
            'densepose': denseposes
        }
    
    def __call__(
        self,
        image: Union[str, Image.Image],
        mask_type: str = "upper",
    ):
        assert mask_type in ['upper', 'lower', 'overall', 'inner', 'outer'], f"mask_type should be one of ['upper', 'lower', 'overall', 'inner', 'outer'], but got {mask_type}"
        preprocess_results = self.preprocess_image(image)
        mask = self.cloth_agnostic_mask(
            preprocess_results['densepose'], 
            preprocess_results['schp_lip'], 
            preprocess_results['schp_atr'], 
            part=mask_type,
        )
        return {
            'mask': mask,
            'densepose': preprocess_results['densepose'],
            'schp_lip': preprocess_results['schp_lip'],
            'schp_atr': preprocess_results['schp_atr']
        }


if __name__ == '__main__':
    pass