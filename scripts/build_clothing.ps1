param(
    [string]$Python = "python",
    [string]$CudaHome = $env:CUDA_HOME,
    [string]$CudaHeaders,
    [string]$CudaArch = "8.9",
    [switch]$SkipImportCheck
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$BuildDrive = $null

# Native CUDA tools on Windows need ASCII source and temporary paths.
if ($ProjectRoot -match '[^\x00-\x7F]') {
    $usedDrives = [IO.Directory]::GetLogicalDrives()
    foreach ($letter in @('Z', 'Y', 'X', 'W', 'V', 'U', 'T')) {
        if ($usedDrives -notcontains "${letter}:\") {
            $candidate = "${letter}:"
            & subst.exe $candidate $ProjectRoot
            if ($LASTEXITCODE -ne 0) { throw "Cannot map build drive $candidate" }
            $BuildDrive = $candidate
            $ProjectRoot = "${candidate}\"
            break
        }
    }
    if (-not $BuildDrive) { throw "No unused drive for native CUDA build" }
}

try {
    $Author = Join-Path $ProjectRoot "third_party/gaussian_wardrobe"
    $Runtime = Join-Path $ProjectRoot "runtime/clothing"
    $Extensions = Join-Path $Runtime "extensions"
    $WardrobeExtensions = Join-Path $Extensions "wardrobe"
    $StyleExtensions = Join-Path $Extensions "styleunet"
    $BuildRoot = Join-Path $Runtime "build"
    $LogRoot = Join-Path $Runtime "logs"
    if (-not (Test-Path -LiteralPath (Join-Path $Author "smplx/body_models.py"))) {
        throw "GaussianWardrobe source missing: $Author"
    }
    $pythonCommand = Get-Command $Python -ErrorAction Stop
    $Python = $pythonCommand.Source
    New-Item -ItemType Directory -Force -Path $Extensions, $WardrobeExtensions, $StyleExtensions, $BuildRoot, $LogRoot | Out-Null
    $env:TEMP = Join-Path $Runtime "temp"
    $env:TMP = $env:TEMP
    $env:TORCH_EXTENSIONS_DIR = Join-Path $Runtime "torch_extensions"
    New-Item -ItemType Directory -Force -Path $env:TEMP, $env:TORCH_EXTENSIONS_DIR | Out-Null
    $env:TORCH_CUDA_ARCH_LIST = $CudaArch
    $env:MAX_JOBS = "2"
    $env:DISTUTILS_USE_SDK = "1"
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:VSLANG = "1033"

    if (-not $CudaHome) {
        $nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
        $CudaHome = Split-Path (Split-Path $nvcc -Parent) -Parent
    }
    if (-not (Test-Path -LiteralPath (Join-Path $CudaHome "bin/nvcc.exe"))) {
        throw "nvcc missing; supply -CudaHome with the matching CUDA toolkit"
    }
    $env:CUDA_HOME = $CudaHome
    $env:CUDA_PATH = $CudaHome
    $env:CUB_HOME = Join-Path $CudaHome "include"
    $env:PATH = (Join-Path $CudaHome "bin") + ";" + $env:PATH

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
        if (-not (Test-Path -LiteralPath $vswhere)) { throw "MSVC discovery tool missing: $vswhere" }
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (-not $vs) { throw "MSVC C++ tools not found" }
        $vcvars = Join-Path $vs "VC/Auxiliary/Build/vcvars64.bat"
        $environmentLines = & cmd.exe /c "`"$vcvars`" >nul 2>&1 && set"
        if ($LASTEXITCODE -ne 0) { throw "MSVC environment initialization failed" }
        foreach ($line in $environmentLines) {
            if ($line -match '^([^=]+)=(.*)$') {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw "cl.exe unavailable" }
    $env:VSCMD_ARG_TGT_ARCH = "x64"
    if ($CudaHeaders) {
        $headers = (Resolve-Path -LiteralPath $CudaHeaders).Path
        $env:INCLUDE = "$headers;" + $env:INCLUDE
        $env:NVCC_PREPEND_FLAGS = "-I`"$headers`" " + $env:NVCC_PREPEND_FLAGS
    }

    & $Python -c "import sys,torch; print(sys.executable); print('torch',torch.__version__,'CUDA',torch.version.cuda)"
    if ($LASTEXITCODE -ne 0) { throw "Selected Python cannot import PyTorch" }

    function Invoke-ExtensionBuild([string]$Name, [string]$Source, [string]$Destination) {
        if (-not (Test-Path -LiteralPath (Join-Path $Source "setup.py"))) {
            throw "setup.py missing: $Source"
        }
        Push-Location -LiteralPath $Source
        try {
            $previousPreference = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            & $Python setup.py build_ext --build-lib $Destination --build-temp (Join-Path $BuildRoot $Name) 2>&1 |
                Tee-Object -FilePath (Join-Path $LogRoot "$Name.log")
            $buildCode = $LASTEXITCODE
            $ErrorActionPreference = $previousPreference
            if ($buildCode -ne 0) { throw "Extension build failed: $Name" }
        } finally {
            Pop-Location
        }
    }

    $RasterSource = Join-Path $Author "gaussians/diff_gaussian_rasterization_depth_alpha"
    $rasterConfig = Get-Content -LiteralPath (Join-Path $RasterSource "cuda_rasterizer/config.h") -Raw
    if ($rasterConfig -notmatch '#define\s+NUM_CHANNELS\s+6\b') {
        throw "GaussianWardrobe requires its six-channel rasterizer"
    }
    Invoke-ExtensionBuild "wardrobe_rasterizer" $RasterSource $WardrobeExtensions
    $RasterPackage = Join-Path $WardrobeExtensions "diff_gaussian_rasterization_depth_alpha"
    New-Item -ItemType Directory -Force -Path $RasterPackage | Out-Null
    Copy-Item -LiteralPath (Join-Path $RasterSource "diff_gaussian_rasterization_depth_alpha/__init__.py") -Destination (Join-Path $RasterPackage "__init__.py") -Force
    Invoke-ExtensionBuild "styleunet" (Join-Path $Author "network/styleunet") $StyleExtensions

    foreach ($name in @("posevocab_custom_ops", "root_finding")) {
        $source = Join-Path $Author "utils/$name"
        $copy = Join-Path $BuildRoot "source/$name"
        if (Test-Path -LiteralPath $copy) {
            $resolved = (Resolve-Path -LiteralPath $copy).Path
            $allowed = (Resolve-Path -LiteralPath $BuildRoot).Path.TrimEnd('\') + '\'
            if (-not $resolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Build-source cleanup outside intended root: $resolved"
            }
            Remove-Item -LiteralPath $resolved -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path $copy | Out-Null
        Get-ChildItem -LiteralPath $source -Force | Copy-Item -Destination $copy -Recurse -Force
        $setup = Join-Path $copy "setup.py"
        $text = Get-Content -LiteralPath $setup -Raw
        $before = '"cxx": ["-O2", "-I{}".format("./")]'
        $after = '"cxx": ["/O2", "/I{}".format("./")]'
        if (-not $text.Contains($before)) { throw "Unsupported MSVC setup flags in $source" }
        Set-Content -LiteralPath $setup -Value $text.Replace($before, $after) -NoNewline -Encoding utf8
        Invoke-ExtensionBuild $name $copy $Extensions
    }

    $required = @(
        (Join-Path $RasterPackage "_C*.pyd"),
        (Join-Path $StyleExtensions "upfirdn2d*.pyd"),
        (Join-Path $StyleExtensions "fused*.pyd"),
        (Join-Path $Extensions "posevocab_custom_ops*.pyd"),
        (Join-Path $Extensions "root_finding*.pyd")
    )
    foreach ($path in $required) {
        if (-not (Test-Path -Path $path)) { throw "Built extension missing: $path" }
    }
    if (-not $SkipImportCheck) {
        $check = "import pathlib,sys; sys.path[:0]=[r'$WardrobeExtensions',r'$StyleExtensions',r'$Extensions']; import torch; import diff_gaussian_rasterization_depth_alpha as d; import upfirdn2d,fused,posevocab_custom_ops,root_finding; assert pathlib.Path(r'$WardrobeExtensions') in pathlib.Path(d._C.__file__).parents; print('Wardrobe extensions imported:',d._C.__file__)"
        & $Python -c $check
        if ($LASTEXITCODE -ne 0) { throw "Built extension import check failed" }
    }
    Write-Host "Clothing extensions built under $Extensions"
} finally {
    if ($BuildDrive) { & subst.exe $BuildDrive /D }
}
