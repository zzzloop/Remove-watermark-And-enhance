$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $AppRoot

$LibDir = Join-Path $AppRoot "python_libs"
$ModelsDir = Join-Path $AppRoot "models"
$CacheDir = Join-Path $AppRoot ".cache"
$TmpDir = Join-Path $CacheDir "tmp"
$LogFile = Join-Path $AppRoot "startup.log"
$Port = 8000

New-Item -ItemType Directory -Force $LibDir, $ModelsDir, $CacheDir, $TmpDir | Out-Null

$env:HF_HOME = Join-Path $ModelsDir ".cache\huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ModelsDir ".cache\huggingface\hub"
$env:TORCH_HOME = Join-Path $ModelsDir ".cache\torch"
$env:XDG_CACHE_HOME = $CacheDir
$env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"
$env:TEMP = $TmpDir
$env:TMP = $TmpDir
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$AppRoot\backend"
$RequiredNumpy = "2.2.2"
$RecommendedEnvName = "wmr310"
$RecommendedPython = "3.10.11"
$TorchIndexUrl = "https://download.pytorch.org/whl/cu118"
$TorchVersion = "2.7.1+cu118"
$TorchVisionVersion = "0.22.1+cu118"

function Write-Step($Text) {
    Write-Host $Text
    Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value $Text
}

function Write-UiLine {
    param(
        [string]$Text,
        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )
    Write-Host $Text -ForegroundColor $Color
    Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value $Text
}

function Write-UiBlank {
    Write-Host ""
    Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value ""
}

function Get-Conda-Envs {
    $envs = @()
    $condaExe = Get-Conda-Exe
    if ($condaExe) {
        try {
            $envLines = & $condaExe env list 2>$null
            foreach ($line in $envLines) {
                $trimmed = $line.Trim()
                if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
                $parts = $trimmed -split "\s+"
                if ($parts.Count -lt 2) { continue }
                $name = $parts[0]
                if ($name -eq "*") { continue }
                $envPath = $parts[$parts.Count - 1]
                if ($envPath -eq "*") { continue }
                $py = Join-Path $envPath "python.exe"
                if (Test-Path $py) {
                    $envs += [PSCustomObject]@{
                        Name = $name
                        Path = $envPath
                        Python = $py
                    }
                }
            }
        } catch {}
    }
    return $envs
}

function Get-Python-Info {
    param([string]$PythonExe)
    $script = "import sys, importlib.util; spec=importlib.util.find_spec('torch'); torch=__import__('torch') if spec else None; cuda=torch.cuda.is_available() if torch else False; dev=torch.cuda.get_device_name(0) if cuda else ('CPU' if torch else 'torch missing'); torch_s=torch.__version__ if torch else 'missing'; print('{}.{}.{}|{}|{}|{}'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro, torch_s, cuda, dev))"
    $out = & $PythonExe -c $script 2>$null
    if (-not $out) { return "unknown|missing|False|cannot run" }
    return $out
}

function Get-Conda-Exe {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        return $env:CONDA_EXE
    }
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda -and $conda.CommandType -eq "Application") { return $conda.Source }
    $condaExe = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($condaExe) { return $condaExe.Source }
    return $null
}

function Get-Conda-Env-Python {
    param([string]$Name)
    $envs = @(Get-Conda-Envs)
    foreach ($envItem in $envs) {
        if ($envItem.Name -eq $Name -and (Test-Path $envItem.Python)) {
            return $envItem.Python
        }
    }
    return $null
}

function Accept-Conda-TosIfUserAllows {
    param([string]$CondaExe)

    Write-Step ""
    Write-Step "Conda 需要先接受 Anaconda 官方频道服务条款，才能自动创建环境。"
    Write-Step "涉及频道："
    Write-Step "  - https://repo.anaconda.com/pkgs/main"
    Write-Step "  - https://repo.anaconda.com/pkgs/r"
    Write-Step "  - https://repo.anaconda.com/pkgs/msys2"
    $answer = Read-Host "是否接受以上 Anaconda 官方频道服务条款并继续创建推荐环境？输入 Y 继续，其他任意键取消"
    if ($answer -notin @("Y", "y")) {
        Write-Step "已取消自动接受服务条款。你也可以之后手动执行 conda tos accept 命令，再重新运行 start.bat。"
        return $false
    }

    $channels = @(
        "https://repo.anaconda.com/pkgs/main",
        "https://repo.anaconda.com/pkgs/r",
        "https://repo.anaconda.com/pkgs/msys2"
    )
    foreach ($channel in $channels) {
        $code = Run-Logged -Exe $CondaExe -ArgList @("tos", "accept", "--override-channels", "--channel", $channel)
        if ($code -ne 0) {
            Write-Step "[ERROR] 接受 Conda 服务条款失败: $channel"
            return $false
        }
    }
    Write-Step "Conda 服务条款已确认，继续创建推荐环境。"
    return $true
}

function New-RecommendedCondaEnv {
    $existing = Get-Conda-Env-Python -Name $RecommendedEnvName
    if ($existing) {
        Write-Step "使用推荐环境: $RecommendedEnvName ($existing)"
        return $existing
    }

    $condaExe = Get-Conda-Exe
    if (-not $condaExe) {
        Write-Step "[ERROR] 未检测到 conda，无法自动创建推荐环境。"
        return $null
    }

    Write-Step "正在创建推荐 conda 环境: $RecommendedEnvName"
    Write-Step "Python version: $RecommendedPython"
    Write-Step "这个环境专门用于本项目，避免 Python 3.13 导致 realesrgan/basicsr/torch 生态依赖安装失败。"
    Write-Step "将使用 conda-forge 创建环境，避免 Anaconda 默认频道 Terms of Service 阻塞。"
    $createArgs = @("create", "-n", $RecommendedEnvName, "python=$RecommendedPython", "-y", "--override-channels", "-c", "conda-forge")
    $code = Run-LoggedWithProgress -Exe $condaExe -ArgList $createArgs -Activity "Creating Python 3.10.11 Conda Environment"
    $created = Get-Conda-Env-Python -Name $RecommendedEnvName
    if ($created) {
        Write-Step "推荐环境已创建: $created"
        return $created
    }
    if ($code -ne 0) {
        Write-Step "conda-forge 创建失败，尝试使用默认 Anaconda 频道。若提示 Terms of Service，需要用户确认后继续。"
        $defaultCreateArgs = @("create", "-n", $RecommendedEnvName, "python=$RecommendedPython", "-y")
        $code = Run-LoggedWithProgress -Exe $condaExe -ArgList $defaultCreateArgs -Activity "Creating Python 3.10.11 Conda Environment"
        $created = Get-Conda-Env-Python -Name $RecommendedEnvName
        if ($created) {
            Write-Step "推荐环境已创建: $created"
            return $created
        }
    }
    if ($code -ne 0) {
        Write-Step "默认频道创建失败。若上方提示 Terms of Service，则需要先确认 Conda 官方频道服务条款。"
        if (Accept-Conda-TosIfUserAllows -CondaExe $condaExe) {
            $code = Run-LoggedWithProgress -Exe $condaExe -ArgList @("create", "-n", $RecommendedEnvName, "python=$RecommendedPython", "-y") -Activity "Creating Python 3.10.11 Conda Environment"
            $created = Get-Conda-Env-Python -Name $RecommendedEnvName
            if ($created) {
                Write-Step "推荐环境已创建: $created"
                return $created
            }
        }
    }
    if ($code -ne 0) {
        Write-Step "[ERROR] 推荐 conda 环境创建失败。可手动尝试: conda create -n $RecommendedEnvName python=$RecommendedPython -y --override-channels -c conda-forge"
        return $null
    }

    $created = Get-Conda-Env-Python -Name $RecommendedEnvName
    if (-not $created) {
        Write-Step "[ERROR] 推荐 conda 环境已创建但未找到 python.exe，请重新运行 start.bat。"
        return $null
    }
    return $created
}

function Select-Python {
    $envs = @(Get-Conda-Envs)
    if ($envs.Count -gt 0) {
        $displayEnvs = @($envs | Where-Object { $_.Name -ne $RecommendedEnvName })
        Write-UiLine "可用的 conda 环境：" Cyan
        Write-UiBlank
        Write-UiLine "请选择带 CUDA 版 PyTorch 的 conda 环境。" White
        Write-UiLine "  [推荐 0] 自动创建/使用 Python 3.10.11 项目专用环境，兼容 Real-ESRGAN、SwinIR、diffusers 等模型依赖。" Green
        Write-UiLine "  [完整体验] 适合增强放大、Anime6B、UltraSharp、SwinIR-L 和生成式修复模型。" Green
        Write-UiLine "  [网络提示] 首次选择 0 会下载 CUDA PyTorch 和项目依赖，建议网络稳定；下载较慢可开启代理后重试。" Yellow
        Write-UiLine "  [断点续传] 如果中途取消，下次重新运行仍可选 0，已下载的 conda/pip 缓存会尽量复用并继续完成安装。" Yellow
        Write-UiLine "  [普通环境] 标记为 CUDA 可用的已有环境可启动基础功能，但可能无法体验完整图片修复/增强功能。" DarkCyan
        Write-UiLine "  [无 CUDA] 会先尝试安装固定 CUDA PyTorch，避免重复下载多个 2GB+ wheel。" DarkYellow
        Write-UiBlank
        $recommendedPythonExe = Get-Conda-Env-Python -Name $RecommendedEnvName
        if ($recommendedPythonExe) {
            $recommendedStatus = if (Test-PythonDependencies -PythonExe $recommendedPythonExe) { "已创建，依赖已完成" } else { "已创建，依赖未完成，继续选 0" }
        } else {
            $recommendedStatus = "未创建"
        }
        Write-UiLine ("  [0] {0,-12} Python {1,-8} 最佳推荐  {2}  放大增强/AI模型最佳体验" -f $RecommendedEnvName, $RecommendedPython, $recommendedStatus) Green
        for ($i = 0; $i -lt $displayEnvs.Count; $i++) {
            $info = Get-Python-Info $displayEnvs[$i].Python
            $parts = $info -split "\|", 4
            $cudaLabel = if ($parts[2] -eq "True") { "CUDA 可用" } else { "无 CUDA" }
            $recommend = if ($parts[2] -eq "True") { "推荐" } else { "不推荐" }
            $featureNote = if ($parts[2] -eq "True") { "可能无法体验完整图片修复/增强功能" } else { "会尝试安装 CUDA PyTorch" }
            $rowColor = if ($parts[2] -eq "True") { [ConsoleColor]::Cyan } else { [ConsoleColor]::DarkYellow }
            Write-UiLine ("  [{0}] {1,-12} Python {2,-8} torch {3,-14} {4,-7} {5}  {6}  {7}" -f ($i + 1), $displayEnvs[$i].Name, $parts[0], $parts[1], $cudaLabel, $recommend, $parts[3], $featureNote) $rowColor
            Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value ("[{0}] {1} {2}" -f ($i + 1), $displayEnvs[$i].Name, $info)
        }
        Write-UiBlank
        $choice = Read-Host "请输入要使用的 conda 环境编号（推荐输入 0）"
        $idx = 0
        if ($choice -eq "0") {
            return New-RecommendedCondaEnv
        }
        if ([int]::TryParse($choice, [ref]$idx) -and $idx -ge 1 -and $idx -le $displayEnvs.Count) {
            return $displayEnvs[$idx - 1].Python
        }
        Write-Step "[ERROR] 环境编号无效。"
        return $null
    }

    $condaExe = Get-Conda-Exe
    if ($condaExe) {
        Write-Step "未列出已有 conda 环境，将尝试创建推荐 Python 3.10.11 环境。"
        return New-RecommendedCondaEnv
    }

    Write-Step "[ERROR] 没有检测到 conda 环境。"
    Write-Host ""
    Write-Host "这个软件的 AI 模型需要一个带 CUDA 版 PyTorch 的 Python/conda 环境。"
    Write-Host "请先手动安装 Miniconda 或 Anaconda，然后创建或选择一个满足以下条件的环境："
    Write-Host "  - Python $RecommendedPython"
    Write-Host "  - PyTorch CUDA 版"
    Write-Host "  - 已安装可用的 NVIDIA 显卡驱动"
    Write-Host ""
    Write-Host "启动器不会自动下载或安装 conda。"
    Write-Host "安装好 conda 和 CUDA 版 PyTorch 后，请重新运行 start.bat。"
    return $null
}

function Run-Logged {
    param([string]$Exe, [string[]]$ArgList)
    Write-Step ("> " + $Exe + " " + ($ArgList -join " "))
    & $Exe @ArgList
    return $LASTEXITCODE
}

function Join-ProcessArguments {
    param([string[]]$ArgList)
    $quoted = @()
    foreach ($arg in $ArgList) {
        if ($arg -match '[\s"]') {
            $quoted += '"' + ($arg.Replace('"', '\"')) + '"'
        } else {
            $quoted += $arg
        }
    }
    return ($quoted -join " ")
}

function Run-LoggedWithProgress {
    param([string]$Exe, [string[]]$ArgList, [string]$Activity)
    Write-Step ("> " + $Exe + " " + ($ArgList -join " "))

    $process = Start-Process -FilePath $Exe -ArgumentList (Join-ProcessArguments $ArgList) -NoNewWindow -PassThru
    $start = Get-Date
    while (-not $process.HasExited) {
        $elapsed = [int]((Get-Date) - $start).TotalSeconds
        $percent = ($elapsed * 3) % 100
        Write-Progress -Activity $Activity -Status ("运行中 {0}s；pip 解析依赖时不会提供文件大小，开始下载 wheel 后会显示真实下载进度。" -f $elapsed) -PercentComplete $percent
        Start-Sleep -Milliseconds 700
        $process.Refresh()
    }
    Write-Progress -Activity $Activity -Completed
    return $process.ExitCode
}

function Test-Dependencies {
    & $Python -c "import importlib.util; import fastapi, uvicorn, PIL, numpy, cv2, torch, timm, diffusers, transformers, accelerate, safetensors, spandrel, spandrel_extra_arches, einops; missing=[m for m in ('onnxruntime','rembg','realesrgan','ben2') if importlib.util.find_spec(m) is None]; raise SystemExit(1 if missing else 0)" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-PythonDependencies {
    param([string]$PythonExe)
    & $PythonExe -c "import importlib.util; import fastapi, uvicorn, PIL, numpy, cv2, torch, timm, diffusers, transformers, accelerate, safetensors, spandrel, spandrel_extra_arches, einops; missing=[m for m in ('onnxruntime','rembg','realesrgan','ben2') if importlib.util.find_spec(m) is None]; raise SystemExit(1 if missing else 0)" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Get-Torch-Cuda-Info {
    $script = "import importlib.util, sys; spec=importlib.util.find_spec('torch');`nif spec is None: print('missing|False|torch missing'); sys.exit(0)`nimport torch`nprint('{}|{}|{}'.format(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
    $out = & $Python -c $script 2>$null
    if (-not $out) { return "missing|False|cannot run torch" }
    return $out
}

function Test-TorchCuda {
    $info = Get-Torch-Cuda-Info
    $parts = $info -split "\|", 3
    return ($parts.Count -ge 2 -and $parts[1] -eq "True")
}

function Install-CudaTorch {
    Write-Step "当前环境没有可用的 CUDA 版 PyTorch，开始安装 CUDA 版 PyTorch。"
    Write-Step "如果安装失败，请换 Python 3.10-3.12 的 conda 环境，或手动安装匹配显卡驱动的 CUDA 版 PyTorch。"
    Write-Step "为避免重复下载多个 2GB+ 的 torch wheel，本启动器只安装一个 CUDA 版本。"
    Write-Step "默认选择 PyTorch CUDA 11.8 wheel: $TorchIndexUrl"
    Write-Step "CUDA 11.8 wheel 对 Python 3.10 更成熟，通常体积也比更新 CUDA wheel 更小。"
    Write-Step "固定版本: torch==$TorchVersion, torchvision==$TorchVisionVersion。"
    $indexUrl = $TorchIndexUrl
    if (Test-TorchCuda) {
        Write-Step "CUDA PyTorch is already ready: $(Get-Torch-Cuda-Info)"
        return $true
    }
    $code = Run-LoggedWithProgress -Exe $Python -ArgList @("-m", "pip", "install", "--verbose", "--progress-bar", "on", "--force-reinstall", "torch==$TorchVersion", "torchvision==$TorchVisionVersion", "numpy==$RequiredNumpy", "--index-url", $indexUrl, "--extra-index-url", "https://pypi.tuna.tsinghua.edu.cn/simple") -Activity "Installing CUDA PyTorch"
    if (Test-TorchCuda) {
        Write-Step "CUDA PyTorch is ready: $(Get-Torch-Cuda-Info)"
        return $true
    }
    Write-Step "[ERROR] 没有安装成功 CUDA 版 PyTorch。为避免重复下载，本次不会自动继续下载 cu126/cu124。"
    Write-Step "如确实需要其他 CUDA wheel，请手动安装后再启动，例如: python -m pip install torch --index-url https://download.pytorch.org/whl/cu126"
    return $false
}

function Test-NumpyVersion {
    $script = "import sys, numpy as np; req='$RequiredNumpy'; print(np.__version__); sys.exit(0 if np.__version__ == req else 1)"
    $out = & $Python -c $script 2>$null
    if ($LASTEXITCODE -eq 0) { return $true }
    $found = if ($out) { $out } else { "无法导入" }
    Write-Step "[ERROR] NumPy 版本不匹配。当前环境 NumPy: $found，项目要求: $RequiredNumpy。"
    Write-Step "请在当前 conda 环境中安装 numpy==$RequiredNumpy，或者切换到依赖版本正确的 conda 环境后重新启动。"
    return $false
}

function Repair-NumpyVersion {
    Write-Step "正在修复 NumPy 版本到 numpy==$RequiredNumpy。"
    $code = Run-LoggedWithProgress -Exe $Python -ArgList @("-m", "pip", "install", "--verbose", "--progress-bar", "on", "--force-reinstall", "numpy==$RequiredNumpy", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple") -Activity "Installing fixed NumPy"
    if ($code -ne 0) {
        $code = Run-LoggedWithProgress -Exe $Python -ArgList @("-m", "pip", "install", "--verbose", "--progress-bar", "on", "--force-reinstall", "numpy==$RequiredNumpy") -Activity "Installing fixed NumPy"
    }
    return (Test-NumpyVersion)
}

Clear-Content -LiteralPath $LogFile -ErrorAction SilentlyContinue
Write-UiLine "==============================================================" DarkGray
Write-UiLine "Watermark Remover" Cyan
Write-UiLine "App folder: $AppRoot" Gray
Write-UiLine "Models: $ModelsDir" Gray
Write-UiLine "Log: $LogFile" Gray
Write-UiLine "退出提示：需要停止服务时按 Ctrl+C；如果看到 'Terminate batch job (Y/N)?'，输入 Y。" Yellow
Write-UiLine "==============================================================" DarkGray

$Python = Select-Python
if (-not $Python) {
    Write-Step "[ERROR] No usable conda environment was selected."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Step "[1/3] Python: $Python"
& $Python --version 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Step "[2/3] Checking dependencies..."
$torchOk = Test-TorchCuda
if (-not $torchOk) {
    $torchOk = Install-CudaTorch
    if (-not $torchOk) {
        Read-Host "Press Enter to exit"
        exit 1
    }
}

$depsOk = Test-Dependencies

if (-not $depsOk) {
    Write-Step "Dependencies missing or version mismatch. Installing into selected Python environment..."
    Write-Step "Installing dependencies with progress. The activity bar moves while pip resolves packages; pip shows exact progress once wheel downloads start."

    $req = Join-Path $AppRoot "backend\requirements.txt"
    $torchConstraints = Join-Path $AppRoot "backend\torch-cu118-constraints.txt"
    Write-Step "Dependency install is constrained to torch==$TorchVersion and torchvision==$TorchVisionVersion to prevent pip from replacing CUDA torch."
    $code = Run-LoggedWithProgress -Exe $Python -ArgList @("-m", "pip", "install", "--verbose", "--progress-bar", "on", "--no-warn-script-location", "-r", $req, "-c", $torchConstraints, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "--extra-index-url", $TorchIndexUrl) -Activity "Installing Python dependencies"
    if ($code -ne 0) {
        $depsOk = Test-Dependencies
        if (-not $depsOk) {
            Write-Step "Tsinghua mirror failed. Trying default PyPI..."
            $code = Run-LoggedWithProgress -Exe $Python -ArgList @("-m", "pip", "install", "--verbose", "--progress-bar", "on", "--no-warn-script-location", "-r", $req, "-c", $torchConstraints, "--extra-index-url", $TorchIndexUrl) -Activity "Installing Python dependencies"
        }
    }
    $depsOk = Test-Dependencies
    if ($depsOk -and -not (Test-TorchCuda)) {
        Write-Step "[ERROR] Dependencies installed, but CUDA PyTorch was replaced or is unavailable."
        Write-Step "Reinstalling fixed CUDA torch once: torch==$TorchVersion, torchvision==$TorchVisionVersion."
        $torchOk = Install-CudaTorch
        $depsOk = (Test-Dependencies -and (Test-TorchCuda))
    }
    if (-not $depsOk) {
        Write-Step "[ERROR] Dependency installation failed."
        Test-NumpyVersion | Out-Null
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Step "Dependencies are ready."
}

if (-not (Test-NumpyVersion)) {
    if (Repair-NumpyVersion) {
        Write-Step "NumPy version is ready: $RequiredNumpy"
    } else {
    Read-Host "Press Enter to exit"
    exit 1
    }
}

Write-Step "[3/3] Starting server..."
& $Python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1 | Tee-Object -FilePath $LogFile -Append

$url = "http://127.0.0.1:$Port"
Write-Host ""
Write-Host "Open this URL if the browser does not open automatically:"
Write-Host $url
Write-Host ""

try {
    Start-Process $url -ErrorAction Stop | Out-Null
} catch {
    Write-Step "Browser auto-open was skipped. Open the URL manually if needed."
}
& $Python (Join-Path $AppRoot "backend\server.py")

Write-Host ""
Read-Host "Server stopped. Press Enter to exit"
