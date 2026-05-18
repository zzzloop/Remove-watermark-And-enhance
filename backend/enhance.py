"""
Real-ESRGAN 超分辨率增强模块
支持 2x / 4x 放大
自动下载模型权重到 ../models/realesrgan/
"""

import os
import sys
import types
import threading
import time
import cv2
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

MODEL_URL_SWINIR_L_X4 = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
MODEL_PATH_SWINIR_L_X4 = os.path.join(MODEL_DIR, "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")

MODEL_URL_ULTRASHARP_X4 = "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth?download=true"
MODEL_PATH_ULTRASHARP_X4 = os.path.join(MODEL_DIR, "4x-UltraSharp.pth")

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


def _print_progress_bar(prefix, downloaded_mb, total_mb, percent):
    bar_width = 28
    filled = int(bar_width * max(0, min(100, percent)) / 100)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r{prefix} [{bar}] {percent:5.1f}%  {downloaded_mb:.1f}MB / {total_mb:.1f}MB", end="")
    sys.stdout.flush()


def _model_spec(model_name="general", scale=2):
    if model_name in ("swinir", "swinir_large"):
        return {
            "path": MODEL_PATH_SWINIR_L_X4,
            "url": MODEL_URL_SWINIR_L_X4,
            "label": "SwinIR-L Real-World 4x GAN 模型",
            "scale": 4,
            "arch": "swinir_large",
            "network": "SwinIR-L nearest+conv real-world GAN",
            "size_hint": "约 142MB",
            "min_bytes": 80 * 1024 * 1024,
        }
    if model_name in ("ultrasharp", "4x_ultrasharp"):
        return {
            "path": MODEL_PATH_ULTRASHARP_X4,
            "url": MODEL_URL_ULTRASHARP_X4,
            "label": "4x-UltraSharp 模型",
            "scale": 4,
            "arch": "spandrel_ultrasharp",
            "network": "spandrel ESRGAN 64nf 23nb",
            "size_hint": "约 67MB",
            "min_bytes": 30 * 1024 * 1024,
        }
    if model_name in ("realesrgan4", "general", "photo"):
        return {
            "path": MODEL_PATH_X4,
            "url": MODEL_URL_X4,
            "label": "R-ESRGAN 4x+ 通用模型",
            "nb": 23,
            "scale": 4,
            "arch": "realesrganer_rrdb",
            "network": "RealESRGANer + basicsr RRDBNet 23 blocks",
            "size_hint": "约 64MB",
            "min_bytes": 30 * 1024 * 1024,
        }
    if model_name in ("anime", "realesrgan_anime"):
        return {
            "path": MODEL_PATH_ANIME_X4,
            "url": MODEL_URL_ANIME_X4,
            "label": "R-ESRGAN 4x+ Anime6B 模型",
            "nb": 6,
            "scale": 4,
            "arch": "realesrganer_anime6b",
            "network": "RealESRGANer + basicsr RRDBNet 6 blocks",
            "size_hint": "约 18MB",
            "min_bytes": 8 * 1024 * 1024,
        }
    raise ValueError("增强模型仅支持 R-ESRGAN 4x+、R-ESRGAN 4x+ Anime6B、4x-UltraSharp 或 SwinIR-L 4x")


def _quarantine_model_file(model_path, reason):
    if not os.path.exists(model_path):
        return
    bad_path = f"{model_path}.bad-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"[ESRGAN] 模型文件疑似损坏，已移到: {bad_path}")
    print(f"[ESRGAN] 损坏原因: {reason}")
    try:
        os.replace(model_path, bad_path)
    except Exception:
        try:
            os.remove(model_path)
            print("[ESRGAN] 无法重命名损坏模型，已删除旧文件。")
        except Exception as exc:
            raise RuntimeError(f"无法处理损坏模型文件: {model_path}: {exc}") from exc


def _is_corrupt_model_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "pytorchstreamreader failed",
        "failed finding central directory",
        "failed reading zip archive",
        "not a zip archive",
        "unexpected eof",
        "pickle data was truncated",
        "ran out of input",
        "invalid load key",
        "invalid header",
        "bad magic number",
    )
    return any(marker in text for marker in markers)


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
    part_path = model_path + ".part"
    min_bytes = spec.get("min_bytes", 1024 * 1024)

    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        if file_size < min_bytes:
            _quarantine_model_file(model_path, f"文件太小: {file_size} bytes，预期至少 {min_bytes} bytes")
        else:
            print(f"[ESRGAN] 模型已存在: {model_path}")
            _set_progress(active=False, status="done", percent=100, message="模型已就绪")
            if progress_callback:
                progress_callback(label, 100, 0, 0, "done")
            return model_path

    print(f"[ESRGAN] 正在下载 {label} 到 {model_path} ...")
    print(f"[ESRGAN] 模型大小{spec.get('size_hint', '未知')}，请耐心等待...")

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
            _print_progress_bar("[ESRGAN] 下载进度", mb_down, mb_total, percent)
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
        if os.path.exists(part_path):
            os.remove(part_path)
        urllib.request.urlretrieve(model_url, part_path, progress_hook)
        file_size = os.path.getsize(part_path)
        if file_size < min_bytes:
            raise RuntimeError(f"模型下载不完整: {file_size} bytes，预期至少 {min_bytes} bytes")
        os.replace(part_path, model_path)
        print(f"\n[ESRGAN] 下载完成!")
        _set_progress(active=False, status="done", percent=100, message="下载完成")
        if progress_callback:
            progress_callback(label, 100, 0, 0, "done")
    except BaseException as e:
        print(f"\n[ESRGAN] 下载失败: {e}")
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except Exception:
                pass
        if not isinstance(e, Exception):
            _set_progress(active=False, status="error", message="下载已取消")
            raise
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


class OldRRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)
        self.scale = 0.2

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
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


class ESRGANOldNet(nn.Module):
    """ESRGAN/RRDBNet layout used by classic .pth upscalers like 4x-UltraSharp."""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super().__init__()
        trunk = make_layer(lambda: OldRRDB(nf, gc), nb)
        self.model = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True),
            nn.Sequential(trunk, nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True),
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.model[0](x)
        trunk = self.model[1](fea)
        fea = fea + trunk
        fea = self.lrelu(self.model[2](F.interpolate(fea, scale_factor=2, mode="nearest")))
        fea = self.lrelu(self.model[3](F.interpolate(fea, scale_factor=2, mode="nearest")))
        out = self.model[5](self.lrelu(self.model[4](fea)))
        return out


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN Anime6B generator architecture."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale

        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), self._make_activation(act_type, num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), self._make_activation(act_type, num_feat)]
        body += [nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)]

        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _make_activation(act_type, num_feat):
        if act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        if act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError(f"Unsupported SRVGG activation: {act_type}")

    def forward(self, x):
        out = self.upsampler(self.body(x))
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


# ---------- 全局模型缓存 ----------

_esrgan_models = {}
_device_esr = None
_torch_backend_inited = False


def _init_torch_backend():
    """初始化 torch 推理加速相关开关（只做一次）。

    目标：不改变结果的前提下，提升 CUDA 推理吞吐/首帧速度。
    """
    global _torch_backend_inited
    if _torch_backend_inited:
        return
    _torch_backend_inited = True

    try:
        # 允许 TF32（Ampere+）通常对超分这类卷积/矩阵计算提速明显，视觉差异可忽略。
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        # PyTorch 2.x：提升 matmul 精度策略（对性能有利）
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        # 不影响主流程
        pass


def get_device():
    """获取推理设备"""
    _init_torch_backend()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def apply_torchvision_compat_patch():
    try:
        from torchvision.transforms.functional import rgb_to_grayscale
    except Exception as exc:
        raise RuntimeError("torchvision 未正确安装，Real-ESRGAN Anime6B 无法加载。") from exc

    module_name = "torchvision.transforms.functional_tensor"
    if module_name not in sys.modules:
        compat_module = types.ModuleType(module_name)
        compat_module.rgb_to_grayscale = rgb_to_grayscale
        sys.modules[module_name] = compat_module


def load_model(scale=2, model_name="general"):
    """加载 Real-ESRGAN 模型（带缓存）"""
    global _esrgan_models, _device_esr

    spec = _model_spec(model_name, scale)
    cache_key = (model_name, spec["scale"])

    if cache_key in _esrgan_models:
        return _esrgan_models[cache_key], _device_esr

    _device_esr = get_device()

    for attempt in range(2):
        model_path = download_model(scale, model_name)
        try:
            if spec.get("arch") == "swinir_large":
                return load_swinir_model(model_path, spec)
            if spec.get("arch") == "spandrel_ultrasharp":
                return load_ultrasharp_model(model_path, spec)
            if spec.get("arch") == "realesrganer_rrdb":
                return load_realesrganer_model(model_path, spec, num_block=23, cache_name="realesrgan_x4plus")
            if spec.get("arch") == "realesrganer_anime6b":
                return load_realesrganer_model(model_path, spec, num_block=6, cache_name="anime6b_realesrganer")
            break
        except Exception as exc:
            if attempt == 0 and _is_corrupt_model_error(exc):
                _quarantine_model_file(model_path, str(exc))
                _set_progress(active=True, status="downloading", percent=0, message="模型文件损坏，正在重新下载...")
                continue
            raise

    print(f"[ESRGAN] 正在加载 {spec['label']} ({spec.get('network', 'RRDBNet')})...")
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


def load_realesrganer_model(model_path, spec, num_block, cache_name):
    global _esrgan_models, _device_esr

    cache_key = (cache_name, spec["scale"])
    if cache_key in _esrgan_models:
        return _esrgan_models[cache_key], _device_esr

    apply_torchvision_compat_patch()
    from basicsr.archs.rrdbnet_arch import RRDBNet as BasicSRRRDBNet
    from realesrgan import RealESRGANer

    print(f"[ESRGAN] 正在加载 {spec['label']} ({spec.get('network', 'RealESRGANer')})...")
    model = BasicSRRRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=num_block,
        num_grow_ch=32,
        scale=4,
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=(_device_esr.type == "cuda"),
        gpu_id=0 if _device_esr.type == "cuda" else None,
    )
    _esrgan_models[cache_key] = upsampler
    print(f"[ESRGAN] {spec['label']}加载完成，设备: {_device_esr}")
    return upsampler, _device_esr


def load_ultrasharp_model(model_path, spec):
    global _esrgan_models, _device_esr

    cache_key = ("spandrel_ultrasharp", spec["scale"])
    if cache_key in _esrgan_models:
        return _esrgan_models[cache_key], _device_esr

    import spandrel_extra_arches
    from spandrel import ImageModelDescriptor, ModelLoader

    spandrel_extra_arches.install()
    print(f"[UltraSharp] 正在加载 {spec['label']} ({spec.get('network', 'spandrel')})...")
    model = ModelLoader().load_from_file(str(model_path))
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError("4x-UltraSharp 不是可用的图像到图像超分模型。")
    model = model.to(_device_esr).eval()
    if _device_esr.type == "cuda":
        model = model.half()
    _esrgan_models[cache_key] = model
    print(f"[UltraSharp] {spec['label']}加载完成，设备: {_device_esr}")
    return model, _device_esr


def load_old_esrgan_model(model_path, spec):
    global _esrgan_models, _device_esr

    cache_key = ("old_esrgan", spec["label"])
    if cache_key in _esrgan_models:
        return _esrgan_models[cache_key], _device_esr

    print(f"[ESRGAN] 正在加载 {spec['label']} ({spec.get('network', 'ESRGAN/RRDBNet')})...")
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = _remap_old_esrgan_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(_device_esr)
    model.eval()
    _esrgan_models[cache_key] = model
    print(f"[ESRGAN] {spec['label']}加载完成，设备: {_device_esr}")
    return model, _device_esr


def _remap_old_esrgan_state_dict(state_dict):
    if not any(k.startswith("model.") for k in state_dict):
        return state_dict

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("module."):
            new_key = key[len("module."):]
        if new_key.startswith("model.0."):
            new_key = new_key.replace("model.0.", "conv_first.", 1)
        elif new_key.startswith("model.1.sub."):
            rest = new_key[len("model.1.sub."):]
            parts = rest.split(".", 1)
            if parts[0].isdigit():
                idx = int(parts[0])
                tail = parts[1] if len(parts) > 1 else ""
                if idx < 23:
                    tail = tail.replace("RDB1.", "rdb1.", 1).replace("RDB2.", "rdb2.", 1).replace("RDB3.", "rdb3.", 1)
                    new_key = f"RRDB_trunk.{idx}.{tail}"
                elif idx == 23:
                    new_key = f"trunk_conv.{tail}"
        elif new_key.startswith("model.3."):
            new_key = new_key.replace("model.3.", "upconv1.", 1)
        elif new_key.startswith("model.6."):
            new_key = new_key.replace("model.6.", "upconv2.", 1)
        elif new_key.startswith("model.8."):
            new_key = new_key.replace("model.8.", "HRconv.", 1)
        elif new_key.startswith("model.10."):
            new_key = new_key.replace("model.10.", "conv_last.", 1)
        remapped[new_key] = value
    return remapped


def load_swinir_model(model_path, spec):
    global _esrgan_models, _device_esr

    cache_key = ("swinir_large", spec["scale"])
    if cache_key in _esrgan_models:
        return _esrgan_models[cache_key], _device_esr

    from swinir_network import SwinIR

    print(f"[SwinIR] 正在加载 {spec['label']} ({spec.get('network', 'SwinIR')})...")
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="3conv",
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(_device_esr)
    model.eval()
    _esrgan_models[cache_key] = model
    print(f"[SwinIR] {spec['label']}加载完成，设备: {_device_esr}")
    return model, _device_esr


def _run_tiled(model, img_tensor, scale, tile=256, tile_overlap=32, label="增强模型"):
    _, _, h, w = img_tensor.size()
    tile = min(tile, h, w)
    tile = max(8, tile - tile % 8)
    if h <= tile and w <= tile:
        _set_progress(active=True, name=label, status="processing", percent=35, message="正在推理 1/1")
        output = model(img_tensor)
        _set_progress(active=True, name=label, status="processing", percent=92, message="正在整理结果")
        return output

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    output = torch.zeros(1, 3, h * scale, w * scale, device=img_tensor.device, dtype=img_tensor.dtype)
    weight = torch.zeros_like(output)

    total = max(1, len(h_idx_list) * len(w_idx_list))
    done = 0
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch)
            out_mask = torch.ones_like(out_patch)
            output[..., h_idx * scale:(h_idx + tile) * scale, w_idx * scale:(w_idx + tile) * scale].add_(out_patch)
            weight[..., h_idx * scale:(h_idx + tile) * scale, w_idx * scale:(w_idx + tile) * scale].add_(out_mask)
            done += 1
            percent = 20 + done * 72 / total
            _set_progress(active=True, name=label, status="processing",
                          percent=round(percent, 1),
                          message=f"正在推理分块 {done}/{total}")
    return output.div_(weight)


def _resize_to_target(image: Image.Image, target_scale: float) -> Image.Image:
    target_scale = max(0.0, min(float(target_scale), 8.0))
    if target_scale <= 0:
        return image.copy()
    w, h = image.size
    out_w = max(1, int(round(w * target_scale)))
    out_h = max(1, int(round(h * target_scale)))
    if (out_w, out_h) == image.size:
        return image
    return image.resize((out_w, out_h), Image.Resampling.LANCZOS)


def _run_with_progress_heartbeat(label: str, start_percent: float, end_percent: float, message_fn, work_fn):
    done = threading.Event()
    started = time.time()

    def heartbeat():
        tick = 0
        while not done.wait(3.0):
            tick += 1
            elapsed = int(time.time() - started)
            percent = min(end_percent, start_percent + tick * 2.5)
            _set_progress(
                active=True,
                name=label,
                status="processing",
                percent=round(percent, 1),
                message=message_fn(elapsed),
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return work_fn()
    finally:
        done.set()


def _enhance_with_realesrganer(image: Image.Image, upsampler, spec, target_scale: float) -> Image.Image:
    original_w, original_h = image.size
    label = spec["label"]
    target_w = max(1, int(round(original_w * target_scale)))
    target_h = max(1, int(round(original_h * target_scale)))
    _set_progress(
        active=True,
        name=label,
        status="processing",
        percent=35,
        message=f"GPU 正在推理，预计输出 {target_w}x{target_h}",
    )

    rgba = np.array(image.convert("RGBA"))
    alpha = rgba[:, :, 3] if (rgba[:, :, 3] < 255).any() else None
    img_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    old_tile = getattr(upsampler, "tile", 0)
    old_tile_pad = getattr(upsampler, "tile_pad", 10)
    upsampler.tile = 0
    upsampler.tile_pad = 10
    print(f"[ESRGAN] 开始快速整图推理: {original_w}x{original_h} -> {target_w}x{target_h}, tile=0, alpha={'yes' if alpha is not None else 'no'}")
    t_infer = time.time()
    try:
        output_bgr, _ = _run_with_progress_heartbeat(
            label,
            35,
            90,
            lambda elapsed: f"GPU 正在快速整图推理，tile=0，预计输出 {target_w}x{target_h}，已耗时 {elapsed}s",
            lambda: upsampler.enhance(img_bgr, outscale=float(target_scale)),
        )
        print(f"[ESRGAN] 快速整图推理完成，耗时 {time.time() - t_infer:.2f}s")
    except RuntimeError as exc:
        err = str(exc).lower()
        if "out of memory" not in err and "cuda" not in err:
            raise
        # 关键优化：
        # 之前这里为了切 tile 会重新 new RealESRGANer + 重新 load 权重，导致“有时特别慢”（大图触发 OOM 时尤为明显）。
        # 实际上 RealESRGANer 支持直接修改 tile/tile_pad 后复用同一个 upsampler，从而避免重复加载权重。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output_bgr = None
        last_exc = exc
        for tile in (800, 512, 400):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _set_progress(active=True, name=label, status="processing", percent=45, message=f"显存不足，改用分块推理 tile={tile}")
            print(f"[ESRGAN] 整图推理显存不足，尝试 tile={tile} (复用已加载模型，避免重复加载权重)")
            try:
                upsampler.tile = tile
                upsampler.tile_pad = 10
                t_tile = time.time()
                output_bgr, _ = _run_with_progress_heartbeat(
                    label,
                    45,
                    90,
                    lambda elapsed, tile=tile: f"正在分块推理 tile={tile}，预计输出 {target_w}x{target_h}，已耗时 {elapsed}s",
                    lambda: upsampler.enhance(img_bgr, outscale=float(target_scale)),
                )
                print(f"[ESRGAN] tile={tile} 分块推理完成，耗时 {time.time() - t_tile:.2f}s")
                break
            except RuntimeError as tile_exc:
                last_exc = tile_exc
                err = str(tile_exc).lower()
                if "out of memory" not in err and "cuda" not in err:
                    raise
        if output_bgr is None:
            raise last_exc
    finally:
        try:
            upsampler.tile = old_tile
            upsampler.tile_pad = old_tile_pad
        except Exception:
            pass

    _set_progress(active=True, name=label, status="processing", percent=95, message="正在整理结果")
    if alpha is not None:
        alpha_up = cv2.resize(alpha, (output_bgr.shape[1], output_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        output_bgra = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2BGRA)
        output_bgra[:, :, 3] = alpha_up
        output_rgb = cv2.cvtColor(output_bgra, cv2.COLOR_BGRA2RGBA)
    else:
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    out_image = Image.fromarray(output_rgb)
    print(f"[ESRGAN] RealESRGANer 增强完成: {original_w}x{original_h} -> {out_image.size[0]}x{out_image.size[1]}")
    return out_image


def _image_to_tensor_bgr(img_bgr: np.ndarray, device: torch.device, use_half: bool):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).unsqueeze(0).to(device)
    return tensor.half() if use_half else tensor


def _tensor_to_image_bgr(tensor: torch.Tensor):
    tensor = tensor.detach().float().clamp(0, 1).squeeze(0).cpu().numpy()
    img_rgb = np.transpose(tensor, (1, 2, 0))
    img_rgb = (img_rgb * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _run_spandrel_tiled(model, img_tensor, scale: int, tile: int, tile_pad: int, label: str):
    _, _, h, w = img_tensor.shape
    if tile <= 0:
        with torch.inference_mode():
            _set_progress(active=True, name=label, status="processing", percent=35, message="正在推理 1/1")
            return model(img_tensor)

    tile = min(tile, h, w)
    output = torch.zeros((1, 3, h * scale, w * scale), device=img_tensor.device, dtype=img_tensor.dtype)
    weight = torch.zeros_like(output)
    y_starts = list(range(0, h, tile))
    x_starts = list(range(0, w, tile))
    total = max(1, len(y_starts) * len(x_starts))
    done = 0

    with torch.inference_mode():
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)
                in_y0 = max(y0 - tile_pad, 0)
                in_x0 = max(x0 - tile_pad, 0)
                in_y1 = min(y1 + tile_pad, h)
                in_x1 = min(x1 + tile_pad, w)

                patch = img_tensor[:, :, in_y0:in_y1, in_x0:in_x1]
                out_patch = model(patch)
                crop_y0 = (y0 - in_y0) * scale
                crop_x0 = (x0 - in_x0) * scale
                crop_y1 = crop_y0 + (y1 - y0) * scale
                crop_x1 = crop_x0 + (x1 - x0) * scale
                out_crop = out_patch[:, :, crop_y0:crop_y1, crop_x0:crop_x1]

                oy0 = y0 * scale
                ox0 = x0 * scale
                oy1 = y1 * scale
                ox1 = x1 * scale
                output[:, :, oy0:oy1, ox0:ox1] += out_crop
                weight[:, :, oy0:oy1, ox0:ox1] += 1
                done += 1
                percent = 20 + done * 72 / total
                _set_progress(active=True, name=label, status="processing",
                              percent=round(percent, 1),
                              message=f"正在推理分块 {done}/{total}")
    return output / weight.clamp_min(1e-8)


def _enhance_with_ultrasharp(image: Image.Image, model, device, spec, target_scale: float) -> Image.Image:
    original_w, original_h = image.size
    label = spec["label"]
    rgba = np.array(image.convert("RGBA"))
    img_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = rgba[:, :, 3] if (rgba[:, :, 3] < 255).any() else None
    use_half = device.type == "cuda"
    img_tensor = _image_to_tensor_bgr(img_bgr, device, use_half)
    output_tensor = None
    last_exc = None
    for tile in (0, 800, 400):
        try:
            tile_text = "整图推理" if tile == 0 else f"分块推理 tile={tile}"
            _set_progress(active=True, name=label, status="processing", percent=30, message=f"正在{tile_text}")
            output_tensor = _run_spandrel_tiled(model, img_tensor, scale=4, tile=tile, tile_pad=16, label=label)
            if device.type == "cuda":
                torch.cuda.synchronize()
            break
        except RuntimeError as exc:
            last_exc = exc
            err = str(exc).lower()
            if "out of memory" not in err and "cuda" not in err:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[UltraSharp] {('整图' if tile == 0 else 'tile=' + str(tile))} 推理显存不足，尝试更小分块")
    if output_tensor is None:
        raise last_exc or RuntimeError("4x-UltraSharp 推理失败")
    output_bgr = _tensor_to_image_bgr(output_tensor)

    if alpha is not None:
        alpha_up = cv2.resize(alpha, (output_bgr.shape[1], output_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2BGRA)
        output_bgr[:, :, 3] = alpha_up
        out_array = cv2.cvtColor(output_bgr, cv2.COLOR_BGRA2RGBA)
    else:
        out_array = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    out_image = Image.fromarray(out_array)
    _set_progress(active=True, name=label, status="processing", percent=95, message="正在缩放到目标倍率")
    out_image = _resize_to_target(out_image, target_scale / 4.0)
    print(f"[UltraSharp] 增强完成: {original_w}x{original_h} -> {out_image.size[0]}x{out_image.size[1]}")
    return out_image


def _enhance_with_esrgan(image: Image.Image, target_scale: float = 4, model_name: str = "swinir_large") -> Image.Image:
    """
    超分辨率增强

    Args:
        image: PIL Image (RGB), 输入图片
        scale: 放大倍数，支持 2 或 4

    Returns:
        PIL Image (RGB), 放大后的图片
    """
    spec = _model_spec(model_name, target_scale)
    native_scale = spec["scale"]
    label = spec["label"]

    _set_progress(active=True, name=label, status="processing", percent=5, message="正在准备增强模型")
    model, device = load_model(native_scale, model_name)

    original_w, original_h = image.size

    if spec.get("arch") in ("realesrganer_rrdb", "realesrganer_anime6b"):
        return _enhance_with_realesrganer(image, model, spec, target_scale)
    if spec.get("arch") == "spandrel_ultrasharp":
        return _enhance_with_ultrasharp(image, model, device, spec, target_scale)

    # 预处理
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    if spec.get("arch") == "swinir_large":
        return _enhance_with_swinir(image, model, device, spec, target_scale)

    # 处理大图
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    max_input = 900
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
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                _set_progress(active=True, name=label, status="processing", percent=40, message="正在推理增强模型")
                result = model(img_tensor)
        else:
            _set_progress(active=True, name=label, status="processing", percent=40, message="正在推理增强模型")
            result = model(img_tensor)

    # 去掉 padding
    if pad_h > 0:
        result = result[:, :, :result.shape[2] - pad_h * native_scale, :]
    if pad_w > 0:
        result = result[:, :, :, :result.shape[3] - pad_w * native_scale]

    # 转 PIL
    result = result.clamp(0, 1)
    result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result_np = (result_np * 255).clip(0, 255).astype(np.uint8)

    out_image = Image.fromarray(result_np)
    _set_progress(active=True, name=label, status="processing", percent=95, message="正在缩放到目标倍率")
    out_image = _resize_to_target(out_image, target_scale / native_scale if native_scale else target_scale)
    print(f"[ESRGAN] 增强完成: {original_w}x{original_h} -> {out_image.size[0]}x{out_image.size[1]}")

    return out_image


def _enhance_with_swinir(image: Image.Image, model, device, spec, target_scale: float) -> Image.Image:
    original_w, original_h = image.size
    scale = spec["scale"]
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    window_size = 8
    _, _, h, w = img_tensor.shape
    h_pad = (h // window_size + 1) * window_size - h
    w_pad = (w // window_size + 1) * window_size - w
    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], 2)[:, :, :h + h_pad, :]
    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, :w + w_pad]

    result = None
    last_exc = None
    with torch.inference_mode():
        for tile in (800, 512, 400, 256):
            try:
                _set_progress(active=True, name=spec["label"], status="processing", percent=30, message=f"正在快速分块推理 tile={tile}")
                result = _run_tiled(model, img_tensor, scale=scale, tile=tile, tile_overlap=32, label=spec["label"])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                break
            except RuntimeError as exc:
                last_exc = exc
                err = str(exc).lower()
                if "out of memory" not in err and "cuda" not in err:
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[SwinIR] tile={tile} 推理显存不足，尝试更小分块")
        if result is None:
            raise last_exc or RuntimeError("SwinIR-L 推理失败")

    result = result[..., :original_h * scale, :original_w * scale]

    result = result.clamp(0, 1)
    result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result_np = (result_np * 255).clip(0, 255).astype(np.uint8)
    out_image = Image.fromarray(result_np)
    _set_progress(active=True, name=spec["label"], status="processing", percent=95, message="正在缩放到目标倍率")
    out_image = _resize_to_target(out_image, target_scale / scale if scale else target_scale)
    print(f"[SwinIR] 增强完成: {original_w}x{original_h} -> {out_image.size[0]}x{out_image.size[1]}")
    return out_image


def enhance(image: Image.Image, scale: float = 4, model_name: str = "swinir_large") -> Image.Image:
    """Run the selected real super-resolution backend and report failures clearly."""
    try:
        if float(scale) <= 0:
            _set_progress(active=False, status="done", percent=100, message="目标倍率为 0，保持原图")
            return image.convert("RGBA")
        result = _enhance_with_esrgan(image, float(scale), model_name)
        _set_progress(active=False, status="done", percent=100, message="增强完成")
        return result
    except Exception as exc:
        _set_progress(active=False, status="error", message=str(exc))
        raise
