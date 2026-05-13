import os
import sys
import threading
import urllib.request
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_ROOT / "models" / "rembg"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
MODEL_PATH = MODEL_DIR / "u2net.onnx"
BEN2_REPO_ID = "PramaLLC/BEN2"
BEN2_DIR = APP_ROOT / "models" / "ben2"
BEN2_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("U2NET_HOME", str(MODEL_DIR))

_rembg_session = None
_ben2_session = None
_download_progress = {
    "active": False,
    "name": "rembg/u2net 抠图模型",
    "percent": 0,
    "downloaded_mb": 0.0,
    "total_mb": 0.0,
    "status": "idle",
    "message": "",
}
_progress_lock = threading.Lock()


def get_download_progress():
    with _progress_lock:
        return dict(_download_progress)


def _set_progress(**kwargs):
    with _progress_lock:
        _download_progress.update(kwargs)


def finish_progress(name=None, message="抠图完成"):
    payload = {
        "active": False,
        "status": "done",
        "percent": 100,
        "downloaded_mb": 0.0,
        "total_mb": 0.0,
        "message": message,
    }
    if name:
        payload["name"] = name
    _set_progress(**payload)


def _print_progress_bar(prefix, downloaded_mb, total_mb, percent):
    bar_width = 28
    filled = int(bar_width * max(0, min(100, percent)) / 100)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r{prefix} [{bar}] {percent:5.1f}%  {downloaded_mb:.1f}MB / {total_mb:.1f}MB", end="")
    sys.stdout.flush()


class HfDownloadProgress:
    _lock = threading.RLock()

    def __init__(self, *args, **kwargs):
        self.iterable = args[0] if args else None
        self.total = kwargs.get("total") or 0
        self.n = kwargs.get("initial") or 0
        self.desc = kwargs.get("desc") or "HuggingFace"
        self.disable = kwargs.get("disable", False)

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def update(self, n=1):
        if self.disable:
            return
        self.n += n
        if self.total:
            percent = min(100, self.n * 100 / self.total)
            mb_down = self.n / (1024 * 1024)
            mb_total = self.total / (1024 * 1024)
            _print_progress_bar("[BG] BEN2 下载进度", mb_down, mb_total, percent)
            _set_progress(
                active=True,
                name="BEN2 高质量抠图模型",
                status="downloading",
                percent=round(percent, 1),
                downloaded_mb=round(mb_down, 1),
                total_mb=round(mb_total, 1),
                message=f"{self.desc}: {mb_down:.1f}MB / {mb_total:.1f}MB",
            )

    def __iter__(self):
        if self.iterable is None:
            return iter(())

        def generator():
            for item in self.iterable:
                yield item
                self.update(1)

        return generator()

    def __len__(self):
        if self.total:
            return int(self.total)
        try:
            return len(self.iterable)
        except Exception:
            return 0

    def close(self):
        print()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _ensure_u2net_model():
    if MODEL_PATH.exists():
        _set_progress(active=False, status="done", percent=100, message="模型已就绪")
        return

    tmp_path = MODEL_PATH.with_suffix(".onnx.part")
    _set_progress(active=True, status="connecting", percent=0, downloaded_mb=0.0, total_mb=0.0, message="正在连接下载源...")
    print(f"[BG] 正在下载 rembg/u2net 模型到 {MODEL_PATH} ...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            _print_progress_bar("[BG] 下载进度", mb_down, mb_total, percent)
            _set_progress(
                status="downloading",
                percent=round(percent, 1),
                downloaded_mb=round(mb_down, 1),
                total_mb=round(mb_total, 1),
                message=f"{mb_down:.1f}MB / {mb_total:.1f}MB",
            )

    try:
        for tick in range(1, 4):
            if tmp_path.exists() or MODEL_PATH.exists():
                break
            _set_progress(active=True, status="connecting", percent=0,
                          message=f"正在连接下载源... {tick}s")
            time.sleep(1)
        urllib.request.urlretrieve(MODEL_URL, tmp_path, progress_hook)
        tmp_path.replace(MODEL_PATH)
        print("\n[BG] rembg/u2net 模型下载完成!")
        _set_progress(active=False, status="done", percent=100, message="下载完成")
    except BaseException as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        if not isinstance(exc, Exception):
            _set_progress(active=False, status="error", message="下载已取消")
            raise
        _set_progress(active=False, status="error", message=str(exc))
        raise


def _remove_with_rembg(image: Image.Image) -> Image.Image:
    global _rembg_session

    if _rembg_session is None:
        print(f"[BG] 正在加载抠图模型 rembg/u2net，模型目录: {MODEL_DIR}")
        print("[BG] 如果本地没有模型，会优先自动下载；下载失败才会兜底。")
        if not MODEL_PATH.exists():
            _set_progress(active=True, status="connecting", percent=0,
                          downloaded_mb=0.0, total_mb=0.0,
                          message="正在准备抠图模型下载...")
        from rembg import new_session, remove
        _ensure_u2net_model()
        _rembg_session = new_session("u2net")
        print("[BG] 抠图模型加载完成")
    else:
        from rembg import remove

    result = remove(image.convert("RGBA"), session=_rembg_session)
    return result.convert("RGBA")


def _ensure_ben2_model():
    if (BEN2_DIR / "model.safetensors").exists() and (BEN2_DIR / "config.json").exists():
        _set_progress(active=False, name="BEN2 高质量抠图模型", status="done", percent=100, message="模型已就绪")
        return

    _set_progress(
        active=True,
        name="BEN2 高质量抠图模型",
        status="connecting",
        percent=0,
        downloaded_mb=0.0,
        total_mb=0.0,
        message="正在连接 Hugging Face...",
    )
    from huggingface_hub import snapshot_download

    last_exc = None
    for attempt in range(1, 4):
        try:
            if attempt > 1:
                _set_progress(
                    active=True,
                    name="BEN2 高质量抠图模型",
                    status="downloading",
                    message=f"网络中断，正在续传重试 {attempt}/3...",
                )
            snapshot_download(
                repo_id=BEN2_REPO_ID,
                local_dir=str(BEN2_DIR),
                allow_patterns=["config.json", "model.safetensors"],
                tqdm_class=HfDownloadProgress,
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt == 3:
                _set_progress(active=False, name="BEN2 高质量抠图模型", status="error", message=str(exc))
                raise
            time.sleep(1)
    if last_exc is not None:
        print("\n[BG] BEN2 下载已通过断点续传重试完成")
    _set_progress(active=False, name="BEN2 高质量抠图模型", status="done", percent=100, message="下载完成")


def _remove_with_ben2(image: Image.Image) -> Image.Image:
    global _ben2_session

    if _ben2_session is None:
        print(f"[BG] 正在加载 BEN2 高质量抠图模型，模型目录: {BEN2_DIR}")
        _ensure_ben2_model()
        import torch
        try:
            from ben2 import AutoModel as BEN2Model
        except ImportError:
            from ben2 import BEN_Base as BEN2Model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _set_progress(active=True, name="BEN2 高质量抠图模型", status="processing", percent=10, message="正在加载模型")
        _ben2_session = BEN2Model.from_pretrained(str(BEN2_DIR))
        _ben2_session.to(device).eval()
        print(f"[BG] BEN2 模型加载完成，设备: {device}")

    _set_progress(active=True, name="BEN2 高质量抠图模型", status="processing", percent=45, message="正在高质量抠图")
    try:
        result = _ben2_session.inference(image.convert("RGB"), refine_foreground=True)
    except TypeError:
        result = _ben2_session.inference(image.convert("RGB"))
    finish_progress(name="BEN2 高质量抠图模型", message="抠图完成")
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


def remove_background(image: Image.Image, model: str = "u2net") -> Image.Image:
    """
    Cut out the main subject and return an RGBA image with transparent background.

    u2net is the fast default. BEN2 is the high-quality option for complex edges.
    OpenCV GrabCut is a local fallback when model downloads or imports fail.
    """
    if model in ("ben2", "BEN2"):
        try:
            return _remove_with_ben2(image)
        except Exception as exc:
            print(f"[BG] BEN2 抠图不可用，回退到 rembg/u2net: {exc}")

    try:
        return _remove_with_rembg(image)
    except Exception as exc:
        print(f"[BG] rembg/u2net 抠图不可用，最后使用 OpenCV 兜底: {exc}")
        return _remove_with_grabcut(image)
