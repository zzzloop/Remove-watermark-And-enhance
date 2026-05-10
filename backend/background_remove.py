import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_ROOT / "models" / "rembg"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("U2NET_HOME", str(MODEL_DIR))

_rembg_session = None


def _remove_with_rembg(image: Image.Image) -> Image.Image:
    global _rembg_session

    from rembg import new_session, remove

    if _rembg_session is None:
        print(f"[BG] 正在加载抠图模型 rembg/u2net，模型目录: {MODEL_DIR}")
        print("[BG] 如果本地没有模型，会优先自动下载；下载失败才会兜底。")
        _rembg_session = new_session("u2net")
        print("[BG] 抠图模型加载完成")

    result = remove(image.convert("RGBA"), session=_rembg_session)
    return result.convert("RGBA")


def _remove_with_grabcut(image: Image.Image) -> Image.Image:
    """Fallback subject cutout using OpenCV GrabCut."""
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if h < 4 or w < 4:
        return image.convert("RGBA")

    margin_x = max(1, int(w * 0.06))
    margin_y = max(1, int(h * 0.06))
    rect = (margin_x, margin_y, max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(rgb, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    alpha_img = Image.fromarray(alpha, mode="L")
    alpha_img = alpha_img.filter(ImageFilter.MedianFilter(size=5))
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=1.0))

    rgba = image.convert("RGBA")
    rgba.putalpha(alpha_img)
    return rgba


def remove_background(image: Image.Image) -> Image.Image:
    """
    Cut out the main subject and return an RGBA image with transparent background.

    rembg/U2-Net gives the best result. OpenCV GrabCut is a local fallback when
    rembg is not installed or its model cannot be downloaded.
    """
    try:
        return _remove_with_rembg(image)
    except Exception as exc:
        print(f"[BG] rembg/u2net 抠图不可用，最后使用 OpenCV 兜底: {exc}")
        return _remove_with_grabcut(image)
