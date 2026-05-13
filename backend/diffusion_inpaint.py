import os
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageFilter

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_ROOT / "models" / "diffusers"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
MODEL_ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "feature_extractor/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
MODEL_IGNORE_PATTERNS = [
    "*.bin",
    "*.ckpt",
    "*.msgpack",
    "*.onnx",
    "*.pth",
    "*.pt",
    "*/diffusion_pytorch_model.bin",
    "*/diffusion_pytorch_model.fp16.bin",
    "*/pytorch_model.bin",
    "*/pytorch_model.fp16.bin",
    "*/model.bin",
    "*/model.fp16.bin",
]

_pipe = None
_download_progress = {
    "active": False,
    "name": "SD 1.5 Inpaint 生成式修复模型",
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


def _print_progress_bar(prefix, downloaded_mb, total_mb, percent):
    bar_width = 28
    filled = int(bar_width * max(0, min(100, percent)) / 100)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r{prefix} [{bar}] {percent:5.1f}%  {downloaded_mb:.1f}MB / {total_mb:.1f}MB", end="")
    sys.stdout.flush()


def model_status() -> dict:
    snapshot_root = MODEL_DIR / "models--stable-diffusion-v1-5--stable-diffusion-inpainting" / "snapshots"
    downloaded = snapshot_root.exists() and any(snapshot_root.iterdir())
    return {
        "id": "sd15",
        "name": "生成式修复 - SD 1.5 Inpaint",
        "model_id": MODEL_ID,
        "downloaded": downloaded,
        "path": str(MODEL_DIR),
        "note": "首次使用会下载到 models/diffusers；适合大块重绘，不一定比 LaMa 更适合去小水印。适合去大块水印，类似重绘效果。",
    }


def _resize_for_sd(image: Image.Image, mask: Image.Image, max_side: int = 1024):
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    new_w = max(64, int(w * scale) // 8 * 8)
    new_h = max(64, int(h * scale) // 8 * 8)
    if (new_w, new_h) == (w, h):
        return image, mask, (w, h)
    return (
        image.resize((new_w, new_h), Image.Resampling.LANCZOS),
        mask.resize((new_w, new_h), Image.Resampling.NEAREST),
        (w, h),
    )


def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("生成式修复模型需要 CUDA 显卡；当前环境未检测到 CUDA。")

    os.environ.setdefault("HF_HOME", str(APP_ROOT / "models" / ".cache" / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(APP_ROOT / "models" / ".cache" / "huggingface" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(APP_ROOT / "models" / ".cache" / "torch"))

    print(f"[SD-Inpaint] 正在加载模型: {MODEL_ID}")
    print(f"[SD-Inpaint] 模型目录: {MODEL_DIR}")

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
                _print_progress_bar("[SD-Inpaint] 下载进度", mb_down, mb_total, percent)
                _set_progress(
                    active=True,
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

    _set_progress(active=True, status="downloading", percent=0, downloaded_mb=0.0, total_mb=0.0, message="检查模型文件...")
    last_exc = None
    for attempt in range(1, 4):
        try:
            if attempt > 1:
                message = f"网络中断，正在续传重试 {attempt}/3..."
                print(f"[SD-Inpaint] {message}")
                _set_progress(active=True, status="downloading", message=message)
                time.sleep(2)
            snapshot_download(
                repo_id=MODEL_ID,
                cache_dir=str(MODEL_DIR),
                allow_patterns=MODEL_ALLOW_PATTERNS,
                ignore_patterns=MODEL_IGNORE_PATTERNS,
                resume_download=True,
                tqdm_class=HfDownloadProgress,
            )
            _set_progress(active=False, status="done", percent=100, message="下载完成")
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            text = str(exc)
            retryable = any(key in text for key in ("IncompleteRead", "ChunkedEncodingError", "Connection broken", "Read timed out", "ConnectionError"))
            if not retryable or attempt >= 3:
                _set_progress(active=False, status="error", message=text)
                raise
            _set_progress(active=True, status="downloading", message=f"网络中断，准备续传重试 {attempt + 1}/3...")
    if last_exc is not None:
        raise last_exc

    _pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=str(MODEL_DIR),
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    ).to("cuda")
    try:
        _pipe.enable_attention_slicing()
        _pipe.enable_vae_slicing()
    except Exception:
        pass
    print("[SD-Inpaint] 模型加载完成")
    return _pipe


def inpaint_diffusion(
    image: Image.Image,
    mask: Image.Image,
    prompt: str = "",
    negative_prompt: str = "",
    steps: int = 28,
    guidance_scale: float = 7.5,
) -> Image.Image:
    if mask.convert("L").getextrema()[1] < 1:
        return image

    pipe = load_pipeline()
    src = image.convert("RGB")
    mask_l = mask.convert("L")
    small_img, small_mask, original_size = _resize_for_sd(src, mask_l)

    if not prompt.strip():
        prompt = "clean original image, natural background, realistic texture, no watermark, no text, high quality"
    if not negative_prompt.strip():
        negative_prompt = "watermark, logo, text, letters, signature, blurry, distorted, artifact"

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=small_img,
        mask_image=small_mask,
        num_inference_steps=max(10, min(int(steps), 60)),
        guidance_scale=float(guidance_scale),
    ).images[0].convert("RGB")

    if result.size != original_size:
        result = result.resize(original_size, Image.Resampling.LANCZOS)

    # Keep unmasked pixels from the original. A slight blur softens mask edges.
    blend_mask = mask_l.filter(ImageFilter.GaussianBlur(radius=1.2))
    return Image.composite(result, src, blend_mask)
