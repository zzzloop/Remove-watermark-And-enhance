import asyncio
import base64
import io
import json
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
import uvicorn

APP_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = APP_ROOT / "frontend"
MODEL_ROOT = APP_ROOT / "models"
CACHE_ROOT = APP_ROOT / ".cache"

MODEL_ROOT.mkdir(exist_ok=True)
CACHE_ROOT.mkdir(exist_ok=True)

# Keep model and package caches inside the project so the folder can be zipped.
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_ROOT / ".cache" / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(MODEL_ROOT / ".cache" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("PYTHONUTF8", "1")

sys.path.insert(0, str(BACKEND_DIR))

from diffusion_inpaint import inpaint_diffusion, model_status as diffusion_model_status
from enhance import enhance, get_download_progress as esrgan_progress
from lama_inpaint import get_download_progress as lama_progress, inpaint, inpaint_lama, inpaint_opencv


@asynccontextmanager
async def lifespan(app_: FastAPI):
    print("=" * 60)
    print("  去水印服务启动中")
    print("=" * 60)
    print(f"  项目目录: {APP_ROOT}")
    print(f"  模型目录: {MODEL_ROOT}")
    print("  模型将在首次处理时按需加载；如果神经模型不可用，会自动使用 OpenCV 兜底。")
    yield

app = FastAPI(title="去水印工具", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def base64_to_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))


def torch_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def get_gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"
    except Exception:
        return "Unknown"


@app.get("/")
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(
            index_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )
    return HTMLResponse("<h1>前端文件未找到，请确认 frontend/index.html 存在</h1>", status_code=404)


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "cuda_available": torch_available(),
        "gpu_name": get_gpu_name(),
        "model_root": str(MODEL_ROOT),
    }


@app.get("/api/models")
async def models():
    return {
        "models": [
            {
                "id": "opencv",
                "name": "快速修复 - OpenCV",
                "downloaded": True,
                "note": "无需下载，速度快，适合小水印和简单背景。",
            },
            {
                "id": "lama",
                "name": "LaMa - 本地模型",
                "downloaded": (MODEL_ROOT / "lama" / "big-lama.pt").exists(),
                "path": str(MODEL_ROOT / "lama"),
                "note": "速度较快；如果当前权重不兼容，会自动兜底到 OpenCV。",
            },
            diffusion_model_status(),
        ]
    }


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


async def read_payload(request: Request) -> dict:
    """Read JSON first, with form fallback for older frontends."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return await request.json()
    form = await request.form()
    return dict(form)


@app.get("/api/download-progress")
async def download_progress_stream():
    async def event_generator():
        while True:
            lama = lama_progress()
            esr = esrgan_progress()
            data = {
                "lama": lama,
                "esrgan": esr,
                "any_active": lama.get("active", False) or esr.get("active", False),
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            if not data["any_active"]:
                await asyncio.sleep(10)
                lama2 = lama_progress()
                esr2 = esrgan_progress()
                if not lama2.get("active", False) and not esr2.get("active", False):
                    final_data = {"lama": lama2, "esrgan": esr2, "any_active": False}
                    yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                    break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/remove")
async def remove_watermark(request: Request):
    t0 = time.time()
    try:
        payload = await read_payload(request)
        image = payload.get("image")
        mask = payload.get("mask")
        model = payload.get("model", "opencv")
        prompt = payload.get("prompt", "")
        if not image or not mask:
            return JSONResponse({"success": False, "error": "缺少 image 或 mask 参数"}, status_code=400)

        img = base64_to_image(image).convert("RGB")
        mask_img = base64_to_image(mask).convert("L")
        orig_w, orig_h = img.size

        if mask_img.size != img.size:
            mask_img = mask_img.resize(img.size, Image.Resampling.NEAREST)

        print(f"[API] 收到去水印请求: {orig_w}x{orig_h}, model={model}")
        if model == "opencv":
            result_img = inpaint_opencv(img, mask_img)
        elif model == "lama":
            try:
                result_img = inpaint_lama(img, mask_img)
            except Exception as exc:
                print(f"[LaMa] 模型不可用，自动改用 OpenCV: {exc}")
                result_img = inpaint_opencv(img, mask_img)
        elif model == "sd15":
            result_img = inpaint_diffusion(img, mask_img, prompt=prompt)
        else:
            return JSONResponse({"success": False, "error": f"未知模型: {model}"}, status_code=400)
        elapsed = time.time() - t0
        print(f"[API] 去水印完成，耗时 {elapsed:.2f}s")

        return JSONResponse(
            {
                "success": True,
                "result": f"data:image/png;base64,{image_to_base64(result_img, 'PNG')}",
                "time": round(elapsed, 2),
                "size": [orig_w, orig_h],
                "model": model,
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


@app.post("/api/enhance")
async def enhance_resolution(request: Request):
    t0 = time.time()
    try:
        payload = await read_payload(request)
        image = payload.get("image")
        scale = int(payload.get("scale", 2))
        enhancer = payload.get("enhancer", "general")
        if not image:
            return JSONResponse({"success": False, "error": "缺少 image 参数"}, status_code=400)

        if enhancer not in ("general", "anime"):
            return JSONResponse({"success": False, "error": "enhancer 仅支持 general 或 anime"}, status_code=400)

        if enhancer == "anime":
            scale = 4

        if scale not in (2, 4):
            return JSONResponse({"success": False, "error": "scale 仅支持 2 或 4"}, status_code=400)

        img = base64_to_image(image).convert("RGB")
        orig_w, orig_h = img.size
        print(f"[API] 收到增强请求: {orig_w}x{orig_h} -> {scale}x, enhancer={enhancer}")

        result_img = enhance(img, scale, enhancer)

        elapsed = time.time() - t0
        new_w, new_h = result_img.size
        print(f"[API] 增强完成: {orig_w}x{orig_h} -> {new_w}x{new_h}, 耗时 {elapsed:.2f}s")

        return JSONResponse(
            {
                "success": True,
                "result": f"data:image/png;base64,{image_to_base64(result_img, 'PNG')}",
                "time": round(elapsed, 2),
                "original_size": [orig_w, orig_h],
                "enhanced_size": [new_w, new_h],
                "enhancer": enhancer,
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n启动服务: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
