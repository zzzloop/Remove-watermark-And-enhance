"""
LaMa (Large Mask Inpainting) 推理模块
支持自动下载模型权重到 ../models/lama/
"""

import os
import sys
import threading
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F

# 模型存放目录（在项目根目录下的 models/lama/）
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "lama")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
MODEL_PATH = os.path.join(MODEL_DIR, "big-lama.pt")
MODEL_PART_PATH = MODEL_PATH + ".part"

# 全局下载进度（供 SSE 接口读取）
_download_progress = {
    "active": False,
    "name": "LaMa 去水印模型",
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
        return dict(_download_progress)


def _set_progress(**kwargs):
    """更新下载进度（线程安全）"""
    with _progress_lock:
        _download_progress.update(kwargs)


def _print_progress_bar(prefix, downloaded_mb, total_mb, percent):
    bar_width = 28
    filled = int(bar_width * max(0, min(100, percent)) / 100)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r{prefix} [{bar}] {percent:5.1f}%  {downloaded_mb:.1f}MB / {total_mb:.1f}MB", end="")
    sys.stdout.flush()


def download_model(progress_callback=None):
    """下载 LaMa 模型权重
    
    Args:
        progress_callback: 可选回调，接收 (name, percent, downloaded_mb, total_mb, status)
    """
    import urllib.request
    
    if os.path.exists(MODEL_PATH):
        print(f"[LaMa] 模型已存在: {MODEL_PATH}")
        _set_progress(active=False, status="done", percent=100, message="模型已就绪")
        if progress_callback:
            progress_callback("LaMa 去水印模型", 100, 0, 0, "done")
        return
    
    print(f"[LaMa] 正在下载模型到 {MODEL_PATH} ...")
    print(f"[LaMa] 模型大小约 200MB，请耐心等待...")
    
    _set_progress(active=True, status="downloading", percent=0, downloaded_mb=0.0, total_mb=0.0)
    if progress_callback:
        progress_callback("LaMa 去水印模型", 0, 0, 0, "downloading")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            _print_progress_bar("[LaMa] 下载进度", mb_down, mb_total, percent)
            _set_progress(
                percent=round(percent, 1),
                downloaded_mb=round(mb_down, 1),
                total_mb=round(mb_total, 1),
                message=f"{mb_down:.1f}MB / {mb_total:.1f}MB"
            )
            if progress_callback:
                progress_callback("LaMa 去水印模型", round(percent, 1), round(mb_down, 1), round(mb_total, 1), "downloading")
    
    try:
        if os.path.exists(MODEL_PART_PATH):
            os.remove(MODEL_PART_PATH)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PART_PATH, progress_hook)
        os.replace(MODEL_PART_PATH, MODEL_PATH)
        print(f"\n[LaMa] 下载完成!")
        _set_progress(active=False, status="done", percent=100, message="下载完成")
        if progress_callback:
            progress_callback("LaMa 去水印模型", 100, 0, 0, "done")
    except BaseException as e:
        print(f"\n[LaMa] 下载失败: {e}")
        if os.path.exists(MODEL_PART_PATH):
            try:
                os.remove(MODEL_PART_PATH)
            except Exception:
                pass
        if not isinstance(e, Exception):
            _set_progress(active=False, status="error", message="下载已取消")
            raise
        print("[LaMa] 尝试备用下载链接（HuggingFace）...")
        _set_progress(status="downloading", message="切换备用源 (HuggingFace)...")
        if progress_callback:
            progress_callback("LaMa 去水印模型", 0, 0, 0, "downloading")
        try:
            from huggingface_hub import snapshot_download

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
                        _print_progress_bar("[LaMa] HF 下载进度", mb_down, mb_total, percent)
                        _set_progress(
                            percent=round(percent, 1),
                            downloaded_mb=round(mb_down, 1),
                            total_mb=round(mb_total, 1),
                            message=f"{mb_down:.1f}MB / {mb_total:.1f}MB"
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

            snapshot_download(
                repo_id="akhaliq/lama",
                allow_patterns=["big-lama.pt"],
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False,
                cache_dir=MODEL_DIR,
                tqdm_class=HfDownloadProgress,
            )
            print("[LaMa] 从 HuggingFace 下载完成!")
            _set_progress(active=False, status="done", percent=100, message="下载完成")
            if progress_callback:
                progress_callback("LaMa 去水印模型", 100, 0, 0, "done")
        except Exception as e2:
            _set_progress(active=False, status="error", message=str(e2))
            if progress_callback:
                progress_callback("LaMa 去水印模型", 0, 0, 0, "error")
            raise RuntimeError(f"模型下载失败: {e2}")


# ---------- LaMa 网络结构 ----------

def get_activation(kind="tanh"):
    if kind == "tanh":
        return nn.Tanh()
    if kind == "sigmoid":
        return nn.Sigmoid()
    if kind == "relu":
        return nn.ReLU()
    if kind == "leaky":
        return nn.LeakyReLU(0.2)
    return nn.Identity()


class SpectralNorm(nn.Module):
    """Spectral Normalization wrapper"""
    def __init__(self, module, name="weight", power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        self.u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        self.v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        self.u.data = self._l2normalize(self.u.data)
        self.v.data = self._l2normalize(self.v.data)
        self.w_bar = nn.Parameter(w.data, requires_grad=True)
    
    @staticmethod
    def _l2normalize(v, eps=1e-12):
        return v / (torch.norm(v) + eps)
    
    def _power_iteration(self, weight, u, v, n_power_iterations):
        for _ in range(n_power_iterations):
            v.data = self._l2normalize(torch.mv(weight.t().view(weight.shape[0], -1), u.data))
            u.data = self._l2normalize(torch.mv(weight.view(weight.shape[0], -1), v.data))
        sigma = torch.dot(u.data, torch.mv(weight.view(weight.shape[0], -1), v.data))
        return weight / sigma
    
    def forward(self, *args, **kwargs):
        weight = self._power_iteration(self.w_bar, self.u, self.v, self.power_iterations)
        setattr(self.module, self.name, weight)
        return self.module(*args, **kwargs)


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, pad_type="reflect", activation="leaky",
                 norm="none", sn=False):
        super().__init__()
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
            padding = 0
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
            padding = 0
        else:
            self.pad = nn.Identity()
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                         padding=padding, dilation=dilation)
        if sn:
            self.conv = SpectralNorm(conv)
        else:
            self.conv = conv
        
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        
        self.activation = get_activation(activation)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, pad_type="zero", activation="leaky",
                 norm="none", sn=False, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                         padding=padding, dilation=dilation)
        if sn:
            self.conv = SpectralNorm(conv)
        else:
            self.conv = conv
        
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        
        self.activation = get_activation(activation)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class FFCResnetBlock(nn.Module):
    """Fast Fourier Convolution Resnet Block"""
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv1 = Conv2dLayer(dim, dim, 3, padding=dilation, dilation=dilation,
                                  activation="leaky", norm="in")
        self.conv2 = Conv2dLayer(dim, dim, 3, padding=1, activation="leaky", norm="in")
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x + out


class SimpleLama(nn.Module):
    """简化版 LaMa 网络，加载 big-lama 预训练权重"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = Conv2dLayer(4, 64, 5, stride=2, padding=2, pad_type="reflect", activation="leaky", norm="bn")
        self.conv2 = Conv2dLayer(64, 128, 3, stride=1, padding=1, activation="leaky", norm="bn")
        self.conv3 = Conv2dLayer(128, 128, 3, stride=1, padding=1, activation="leaky", norm="bn")
        self.conv4 = Conv2dLayer(128, 256, 3, stride=2, padding=1, activation="leaky", norm="bn")
        self.conv5 = Conv2dLayer(256, 256, 3, stride=1, padding=1, activation="leaky", norm="bn")
        self.conv6 = Conv2dLayer(256, 256, 3, stride=1, padding=1, activation="leaky", norm="bn")
        
        # Res blocks
        self.res1 = FFCResnetBlock(256, dilation=1)
        self.res2 = FFCResnetBlock(256, dilation=2)
        self.res3 = FFCResnetBlock(256, dilation=4)
        self.res4 = FFCResnetBlock(256, dilation=8)
        self.res5 = FFCResnetBlock(256, dilation=16)
        self.res6 = FFCResnetBlock(256, dilation=1)
        self.res7 = FFCResnetBlock(256, dilation=1)
        self.res8 = FFCResnetBlock(256, dilation=1)
        self.res9 = FFCResnetBlock(256, dilation=1)
        
        # Decoder
        self.deconv1 = TransposeConv2dLayer(512, 128, 3, stride=1, padding=1, activation="leaky", norm="in")
        self.deconv2 = Conv2dLayer(256, 128, 3, stride=1, padding=1, activation="leaky", norm="in")
        self.deconv3 = Conv2dLayer(192, 64, 3, stride=1, padding=1, activation="leaky", norm="in")
        self.deconv4 = TransposeConv2dLayer(128, 64, 3, stride=1, padding=1, activation="leaky", norm="in")
        self.deconv5 = Conv2dLayer(128, 32, 3, stride=1, padding=1, activation="leaky", norm="in")
        self.deconv6 = Conv2dLayer(32, 3, 3, stride=1, padding=1, activation="tanh")
    
    def forward(self, image, mask):
        B, C, H, W = image.shape
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="reflect")
        
        masked_image = image * (1 - mask)
        x = torch.cat([masked_image, mask], dim=1)
        
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        
        r = self.res1(e6)
        r = self.res2(r)
        r = self.res3(r)
        r = self.res4(r)
        r = self.res5(r)
        r = self.res6(r)
        r = self.res7(r)
        r = self.res8(r)
        r = self.res9(r)
        
        d1 = self.deconv1(torch.cat([r, e6], dim=1))
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.deconv2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.deconv3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d4 = self.deconv4(d3)
        d4 = torch.cat([d4, x[:, :3]], dim=1)
        d5 = self.deconv5(d4)
        out = self.deconv6(d5)
        
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        
        return out


# ---------- 全局模型加载 ----------

_lama_model = None
_device = None


def get_device():
    """获取推理设备"""
    if torch.cuda.is_available():
        print("[LaMa] 使用 CUDA 推理")
        return torch.device("cuda")
    print("[LaMa] 使用 CPU 推理（较慢）")
    return torch.device("cpu")


def load_model():
    """加载或获取已加载的 LaMa 模型"""
    global _lama_model, _device
    
    if _lama_model is not None:
        return _lama_model, _device
    
    _device = get_device()
    
    download_model()
    
    print("[LaMa] 正在加载模型...")
    try:
        # big-lama.pt is usually a TorchScript archive. Opening the file first
        # avoids torch.jit.load path issues in non-ASCII folders on Windows.
        with open(MODEL_PATH, "rb") as f:
            _lama_model = torch.jit.load(f, map_location=_device)
        _lama_model.eval()
        print(f"[LaMa] TorchScript 模型加载完成，设备: {_device}")
        return _lama_model, _device
    except Exception as e:
        print(f"[LaMa] TorchScript 加载失败，尝试 state_dict: {e}")

    _lama_model = SimpleLama()
    
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    
    new_state = {}
    for k, v in state_dict.items():
        key = k.replace("model.", "").replace("generator.", "")
        new_state[key] = v
    
    try:
        _lama_model.load_state_dict(new_state, strict=True)
    except Exception as e:
        print(f"[LaMa] 严格加载失败: {e}")
        print("[LaMa] 尝试宽松加载...")
        _lama_model.load_state_dict(new_state, strict=False)
    
    _lama_model = _lama_model.to(_device)
    _lama_model.eval()
    print(f"[LaMa] 模型加载完成，设备: {_device}")
    
    return _lama_model, _device


def pad_to_modulo(img_tensor, mod=8):
    """将图片 padding 到 mod 的倍数"""
    _, _, h, w = img_tensor.shape
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return img_tensor


def _inpaint_with_lama(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    使用 LaMa 模型去除水印
    
    Args:
        image: PIL Image (RGB), 原始图片
        mask: PIL Image (L 或 RGB), 白色=水印区域（要修复的部分）
    
    Returns:
        PIL Image (RGB), 去水印后的图片
    """
    model, device = load_model()
    
    original_size = image.size
    
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) * 2.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    mask_img = mask.convert("L")
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    
    if mask_tensor.max() < 0.001:
        return image
    
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    max_size = 1536
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_tensor = F.interpolate(img_tensor, (new_h, new_w), mode="bilinear", align_corners=False)
        mask_tensor = F.interpolate(mask_tensor, (new_h, new_w), mode="nearest")
    
    _, _, h, w = img_tensor.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    
    # Keep LaMa in FP32. CUDA autocast can feed half tensors into float32
    # convolution bias parameters in this custom model and fail with:
    # "Input type (c10::Half) and bias type (float) should be the same".
    with torch.no_grad():
        result = model(img_tensor.float(), mask_tensor.float())
    
    result = result[:, :, :h, :w]
    
    if result.shape[2] != original_size[1] or result.shape[3] != original_size[0]:
        result = F.interpolate(result, (original_size[1], original_size[0]),
                               mode="bilinear", align_corners=True)
    
    result = result.clamp(-1, 1)
    result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result_np = ((result_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(result_np)


def _inpaint_with_opencv(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Reliable local fallback when the neural model is missing or incompatible."""
    try:
        import cv2

        img_np = np.array(image.convert("RGB"))
        mask_np = np.array(mask.convert("L"))
        mask_np = (mask_np > 8).astype(np.uint8) * 255
        if mask_np.max() == 0:
            return image

        radius = max(3, int(round(max(image.size) * 0.006)))
        radius = min(radius, 15)
        result_rgb = cv2.inpaint(img_np, mask_np, radius, cv2.INPAINT_TELEA)
        return Image.fromarray(result_rgb)
    except Exception:
        # Last-resort pure PIL fill. It is not as good as OpenCV, but keeps the app usable.
        base = image.convert("RGB")
        mask_l = mask.convert("L")
        blurred = base.filter(ImageFilter.GaussianBlur(radius=12))
        return Image.composite(blurred, base, mask_l)


def inpaint_opencv(image: Image.Image, mask: Image.Image) -> Image.Image:
    return _inpaint_with_opencv(image, mask)


def inpaint_lama(image: Image.Image, mask: Image.Image) -> Image.Image:
    return _inpaint_with_lama(image, mask)


def inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Remove a watermark using LaMa when available, with OpenCV/PIL fallback.

    The fallback is intentional: packaged copies should still work on machines
    where the LaMa checkpoint cannot be loaded because of Python/Torch version
    differences or missing GPU support.
    """
    if np.array(mask.convert("L")).max() < 1:
        return image

    try:
        return _inpaint_with_lama(image, mask)
    except Exception as exc:
        print(f"[LaMa] 神经模型推理失败，使用 OpenCV 兜底修复: {exc}")
        return _inpaint_with_opencv(image, mask)
