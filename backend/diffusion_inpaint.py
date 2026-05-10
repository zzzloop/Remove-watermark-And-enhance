import os
from pathlib import Path

from PIL import Image, ImageFilter

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_ROOT / "models" / "diffusers"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"

_pipe = None


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

    if not torch.cuda.is_available():
        raise RuntimeError("生成式修复模型需要 CUDA 显卡；当前环境未检测到 CUDA。")

    os.environ.setdefault("HF_HOME", str(APP_ROOT / "models" / ".cache" / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(APP_ROOT / "models" / ".cache" / "huggingface" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(APP_ROOT / "models" / ".cache" / "torch"))

    print(f"[SD-Inpaint] 正在加载模型: {MODEL_ID}")
    print(f"[SD-Inpaint] 模型目录: {MODEL_DIR}")
    _pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=str(MODEL_DIR),
        torch_dtype=torch.float16,
        use_safetensors=False,
        safety_checker=None,
        requires_safety_checker=False,
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
