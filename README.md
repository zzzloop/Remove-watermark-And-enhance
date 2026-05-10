# 去水印工具

本项目是一个本地运行的图片去水印工具，包含前端页面、FastAPI 后端和本地模型目录。

## 一键启动

双击 `start.bat`。

启动器会先检测本机 conda 环境，并提示选择带 CUDA 版 PyTorch 的环境。建议选择列表中标记为“CUDA 可用 / 推荐”的环境；没有 CUDA 的环境只能运行基础 CPU/OpenCV 功能，AI 模型可能无法加载。

如果没有检测到 conda 环境，启动器只会给出说明并退出，不会自动下载或安装 conda。请先手动安装 Miniconda/Anaconda，并准备一个包含以下内容的环境：

- Python 3.10+
- PyTorch CUDA 版
- 可用的 NVIDIA 显卡驱动

依赖和模型缓存会尽量写入当前项目目录：

- `models/`
- `models/.cache/`
- `.cache/`

启动后会自动打开：

```text
http://127.0.0.1:8000
```

## 功能

- 上传 JPG / PNG / WebP 图片
- 涂抹水印区域生成遮罩
- 撤销、清空、画笔大小调整、涂抹/擦除切换
- 主体抠图：把主体抠出来并导出透明背景 PNG；优先使用 rembg/U2-Net，本地没有模型会自动下载到 `models/rembg/`；只有模型或依赖加载失败时才使用 OpenCV GrabCut 兜底；适合主体清晰的人像、商品和普通物体
- 可选择去水印模型：
  - 快速修复 - OpenCV：无需下载，速度最快
  - LaMa - 本地模型：首次使用时会确认或下载 `models/lama/big-lama.pt`，首次加载可能较慢
  - 生成式修复 - SD 1.5 Inpaint：首次使用时会自动下载到 `models/diffusers/`，模型较大、耗时较长；适合去大块水印，类似重绘效果；小水印通常优先用 LaMa
- 2x / 4x 超分增强，可直接增强原图，也可以增强去水印后的结果
  - 通用照片增强 - Real-ESRGAN：适合照片、真实图片和普通截图，支持 2x / 4x
  - 动漫插画增强 - Anime 6B：适合插画、线条图、二次元图片，固定 4x
- 原图/结果滑动对比
- 下载处理结果

## 说明

首次运行相关模型时可能需要自动下载，耗时较长，请等待终端下载进度完成：

- LaMa 会确认或下载 `models/lama/big-lama.pt`
- 生成式修复 - SD 1.5 Inpaint 会下载到 `models/diffusers/`
- rembg/U2-Net 主体抠图模型会下载到 `models/rembg/`

Real-ESRGAN 通用模型和 Anime 6B 模型在首次使用增强功能时会下载到 `models/realesrgan/`。如果模型加载失败，程序会自动使用 OpenCV、GrabCut 或 Lanczos 兜底，保证基础功能仍可使用。

## 退出

服务运行时，在启动窗口按 `Ctrl+C` 停止服务。如果 Windows 显示：

```text
Terminate batch job (Y/N)?
```

输入 `Y` 并按回车即可退出。
