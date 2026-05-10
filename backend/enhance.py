"""
Real-ESRGAN 超分辨率增强模块
支持 2x / 4x 放大
自动下载模型权重到 ../models/realesrgan/
"""

import os
import sys
import threading
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "realesrgan")
os.makedirs(MODEL_DIR, exist_ok=True)

# Real-ESRGAN x2plus 模型 (通用，效果最好)
MODEL_URL_X2 = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
MODEL_PATH_X2 = os.path.join(MODEL_DIR, "RealESRGAN_x2plus.pth")

# Real-ESRGAN x4plus 模型
MODEL_URL_X4 = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH_X4 = os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth")

# Real-ESRGAN x4plus anime 6B 模型（动漫/插画/线稿）
MODEL_URL_ANIME_X4 = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
MODEL_PATH_ANIME_X4 = os.path.join(MODEL_DIR, "RealESRGAN_x4plus_anime_6B.pth")

# 全局下载进度（供 SSE 接口读取）
_esrgan_download_progress = {
    "active": False,
    "name": "Real-ESRGAN 超分模型",
    "percent": 0,
    "downloaded_mb": 0.0,
    "total_mb": 0.0,
    "status": "idle",
    "message": "",
}
_progress_lock = threading.Lock()


def get_download_progress():
    """获取当前下载进度（线程安全）"""
    with _progress_lock:
        return dict(_esrgan_download_progress)


def _set_progress(**kwargs):
    """更新下载进度（线程安全）"""
    with _progress_lock:
        _esrgan_download_progress.update(kwargs)


def _model_spec(model_name="general", scale=2):
    if model_name == "anime":
        if scale != 4:
            raise ValueError("动漫/插画增强模型仅支持 4x")
        return {
            "path": MODEL_PATH_ANIME_X4,
            "url": MODEL_URL_ANIME_X4,
            "label": "Real-ESRGAN Anime 4x 模型",
            "nb": 6,
            "scale": 4,
        }
    if scale == 2:
        return {
            "path": MODEL_PATH_X2,
            "url": MODEL_URL_X2,
            "label": "Real-ESRGAN 通用 2x 模型",
            "nb": 23,
            "scale": 2,
        }
    if scale == 4:
        return {
            "path": MODEL_PATH_X4,
            "url": MODEL_URL_X4,
            "label": "Real-ESRGAN 通用 4x 模型",
            "nb": 23,
            "scale": 4,
        }
    raise ValueError("仅支持 2x 或 4x 放大")


def download_model(scale=2, model_name="general", progress_callback=None):
    """下载 Real-ESRGAN 模型
    
    Args:
        scale: 放大倍数 (2 或 4)
        model_name: general 或 anime
        progress_callback: 可选回调，接收 (name, percent, downloaded_mb, total_mb, status)
    """
    import urllib.request

    spec = _model_spec(model_name, scale)
    model_path = spec["path"]
    model_url = spec["url"]
    label = spec["label"]

    if os.path.exists(model_path):
        print(f"[ESRGAN] 模型已存在: {model_path}")
        _set_progress(active=False, status="done", percent=100, message="模型已就绪")
        if progress_callback:
            progress_callback(label, 100, 0, 0, "done")
        return model_path

    print(f"[ESRGAN] 正在下载 {scale}x 模型到 {model_path} ...")
    print(f"[ESRGAN] 模型大小约 64MB，请耐心等待...")

    _set_progress(active=True, status="downloading", percent=0,
                  downloaded_mb=0.0, total_mb=0.0,
                  name=label)
    if progress_callback:
        progress_callback(label, 0, 0, 0, "downloading")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r[ESRGAN] 下载进度: {mb_down:.1f}MB / {mb_total:.1f}MB ({percent:.0f}%)", end="")
            sys.stdout.flush()
            _set_progress(
                percent=round(percent, 1),
                downloaded_mb=round(mb_down, 1),
                total_mb=round(mb_total, 1),
                message=f"{mb_down:.1f}MB / {mb_total:.1f}MB"
            )
            if progress_callback:
                progress_callback(label, round(percent, 1), round(mb_down, 1),
                                  round(mb_total, 1), "downloading")

    try:
        urllib.request.urlretrieve(model_url, model_path, progress_hook)
        print(f"\n[ESRGAN] 下载完成!")
        _set_progress(active=False, status="done", percent=100, message="下载完成")
        if progress_callback:
            progress_callback(label, 100, 0, 0, "done")
    except Exception as e:
        print(f"\n[ESRGAN] 下载失败: {e}")
        _set_progress(active=False, status="error", message=str(e))
        if progress_callback:
            progress_callback(f"Real-ESRGAN {scale}x 模型", 0, 0, 0, "error")
        raise RuntimeError(f"模型下载失败: {e}")

    return model_path


# ---------- Real-ESRGAN 中的 RRDB 网络 ----------

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block"""

    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale = 0.2

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.scale + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)
        self.scale = 0.2

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.scale + x


class RRDBNet(nn.Module):
    """Real-ESRGAN 的生成器网络"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(lambda: RRDB(nf, gc), nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # 上采样层
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if scale == 4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


# ---------- 全局模型缓存 ----------

_esrgan_models = {}
_device_esr = None


def get_device():
    """获取推理设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(scale=2, model_name="general"):
    """加载 Real-ESRGAN 模型（带缓存）"""
    global _esrgan_models, _device_esr

    spec = _model_spec(model_name, scale)
    cache_key = (model_name, spec["scale"])

    if cache_key in _esrgan_models:
        return _esrgan_models[cache_key], _device_esr

    _device_esr = get_device()

    # 下载模型
    model_path = download_model(scale, model_name)

    print(f"[ESRGAN] 正在加载 {spec['label']}...")
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=spec["nb"], gc=32, scale=spec["scale"])

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    model.load_state_dict(state_dict, strict=True)
    model = model.to(_device_esr)
    model.eval()

    _esrgan_models[cache_key] = model
    print(f"[ESRGAN] {spec['label']}加载完成，设备: {_device_esr}")

    return model, _device_esr


def _enhance_with_esrgan(image: Image.Image, scale: int = 2, model_name: str = "general") -> Image.Image:
    """
    超分辨率增强

    Args:
        image: PIL Image (RGB), 输入图片
        scale: 放大倍数，支持 2 或 4

    Returns:
        PIL Image (RGB), 放大后的图片
    """
    spec = _model_spec(model_name, scale)
    scale = spec["scale"]

    model, device = load_model(scale, model_name)

    original_w, original_h = image.size

    # 预处理
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # 处理大图
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    max_input = 800 if scale == 4 else 1200
    if max(h, w) > max_input:
        s = max_input / max(h, w)
        new_h, new_w = int(h * s), int(w * s)
        # 确保能被 2 整除
        new_h = new_h - new_h % 2
        new_w = new_w - new_w % 2
        img_tensor = F.interpolate(img_tensor, (new_h, new_w), mode="bilinear", align_corners=False)
        print(f"[ESRGAN] 图片过大，缩放到 {new_w}x{new_h} 后处理")

    # 确保尺寸能被 2 整除
    _, _, h, w = img_tensor.shape
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    # 推理
    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                result = model(img_tensor)
        else:
            result = model(img_tensor)

    # 去掉 padding
    if pad_h > 0:
        result = result[:, :, :result.shape[2] - pad_h * scale, :]
    if pad_w > 0:
        result = result[:, :, :, :result.shape[3] - pad_w * scale]

    # 转 PIL
    result = result.clamp(0, 1)
    result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result_np = (result_np * 255).clip(0, 255).astype(np.uint8)

    out_image = Image.fromarray(result_np)
    print(f"[ESRGAN] 增强完成: {original_w}x{original_h} -> {out_image.size[0]}x{out_image.size[1]}")

    return out_image


def enhance(image: Image.Image, scale: int = 2, model_name: str = "general") -> Image.Image:
    """
    Super-resolution with Real-ESRGAN when available.

    If the checkpoint cannot be downloaded or loaded, use high-quality Lanczos
    scaling so the button still gives a usable result in offline packages.
    """
    spec = _model_spec(model_name, scale)
    scale = spec["scale"]
    try:
        return _enhance_with_esrgan(image, scale, model_name)
    except Exception as exc:
        print(f"[ESRGAN] 模型增强失败，使用 Lanczos 兜底放大: {exc}")
        w, h = image.size
        return image.convert("RGB").resize((w * scale, h * scale), Image.Resampling.LANCZOS)
