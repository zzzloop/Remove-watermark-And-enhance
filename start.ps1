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

function Write-Step($Text) {
    Write-Host $Text
    Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value $Text
}

function Get-Conda-Envs {
    $envs = @()
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        try {
            $envLines = & conda env list 2>$null
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

function Select-Python {
    $envs = @(Get-Conda-Envs)
    if ($envs.Count -gt 0) {
        Write-Step "可用的 conda 环境："
        Write-Host ""
        Write-Host "请选择带 CUDA 版 PyTorch 的 conda 环境。"
        Write-Host "建议选择标记为“CUDA 可用”的环境；“无 CUDA”的环境只能运行基础 CPU/OpenCV 功能，AI 模型可能无法加载。"
        Write-Host ""
        for ($i = 0; $i -lt $envs.Count; $i++) {
            $info = Get-Python-Info $envs[$i].Python
            $parts = $info -split "\|", 4
            $cudaLabel = if ($parts[2] -eq "True") { "CUDA 可用" } else { "无 CUDA" }
            $recommend = if ($parts[2] -eq "True") { "推荐" } else { "不推荐" }
            Write-Host ("  [{0}] {1,-12} Python {2,-8} torch {3,-14} {4,-7} {5}  {6}" -f ($i + 1), $envs[$i].Name, $parts[0], $parts[1], $cudaLabel, $recommend, $parts[3])
            Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value ("[{0}] {1} {2}" -f ($i + 1), $envs[$i].Name, $info)
        }
        Write-Host ""
        $choice = Read-Host "请输入要使用的 CUDA conda 环境编号"
        $idx = 0
        if ([int]::TryParse($choice, [ref]$idx) -and $idx -ge 1 -and $idx -le $envs.Count) {
            return $envs[$idx - 1].Python
        }
        Write-Step "[ERROR] 环境编号无效。"
        return $null
    }

    Write-Step "[ERROR] 没有检测到 conda 环境。"
    Write-Host ""
    Write-Host "这个软件的 AI 模型需要一个带 CUDA 版 PyTorch 的 Python/conda 环境。"
    Write-Host "请先手动安装 Miniconda 或 Anaconda，然后创建或选择一个满足以下条件的环境："
    Write-Host "  - Python 3.10+"
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

function Test-Dependencies {
    & $Python -c "import fastapi, uvicorn, PIL, numpy, cv2, torch, diffusers, transformers, accelerate, safetensors" *> $null
    return ($LASTEXITCODE -eq 0)
}

Clear-Content -LiteralPath $LogFile -ErrorAction SilentlyContinue
Write-Step "=============================================================="
Write-Step "Watermark Remover"
Write-Step "App folder: $AppRoot"
Write-Step "Models: $ModelsDir"
Write-Step "Log: $LogFile"
Write-Step "退出提示：需要停止服务时按 Ctrl+C；如果看到 'Terminate batch job (Y/N)?'，输入 Y。"
Write-Step "=============================================================="

$Python = Select-Python
if (-not $Python) {
    Write-Step "[ERROR] No usable conda environment was selected."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Step "[1/3] Python: $Python"
& $Python --version 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Step "[2/3] Checking dependencies..."
$depsOk = Test-Dependencies

if (-not $depsOk) {
    Write-Step "Dependencies missing or version mismatch. Installing into selected Python environment..."

    $req = Join-Path $AppRoot "backend\requirements.txt"
    $code = Run-Logged -Exe $Python -ArgList @("-m", "pip", "install", "--progress-bar", "on", "--no-warn-script-location", "-r", $req, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple")
    if ($code -ne 0) {
        $depsOk = Test-Dependencies
        if (-not $depsOk) {
            Write-Step "Tsinghua mirror failed. Trying default PyPI..."
            $code = Run-Logged -Exe $Python -ArgList @("-m", "pip", "install", "--progress-bar", "on", "--no-warn-script-location", "-r", $req)
        }
    }
    $depsOk = Test-Dependencies
    if (-not $depsOk) {
        Write-Step "[ERROR] Dependency installation failed."
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Step "Dependencies are ready."
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
