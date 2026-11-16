[CmdletBinding()]
param(
    [string]$Python = 'python',
    [Parameter(Mandatory = $true)][string]$CudaHome,
    [string]$WindowsSdk,
    [ValidateSet('rasterizer', 'hybrid', 'knn', 'all')]
    [string]$Component = 'all'
)

$ErrorActionPreference = 'Stop'
$Project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$OwnedDrive = $null
$OriginalEnvironment = @{}
Get-ChildItem Env: | ForEach-Object { $OriginalEnvironment[$_.Name] = $_.Value }
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Prepend-EnvironmentPath([string]$Name, [string[]]$Paths) {
    $Current = [Environment]::GetEnvironmentVariable($Name, 'Process')
    $Value = (@($Paths) + @($Current) | Where-Object { $_ }) -join ';'
    [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
}

function Invoke-LoggedBuild([string]$Name, [string]$WorkingDirectory, [string[]]$Arguments) {
    $Log = Join-Path $Logs "$Name.log"
    Push-Location -LiteralPath $WorkingDirectory
    try {
        & $PythonExe @Arguments *> $Log
        $Code = $LASTEXITCODE
        if ($Code -ne 0) {
            Get-Content -LiteralPath $Log -Tail 20 | Write-Host
            throw "$Name failed with exit code $Code. Log: $Log"
        }
    } finally {
        Pop-Location
    }
    Write-Host "$Name built under $Extensions; log: $Log"
}

try {
    $PythonExe = (Get-Command $Python -CommandType Application -ErrorAction Stop).Source
    $CudaRoot = (Resolve-Path -LiteralPath $CudaHome).Path
    if (-not (Test-Path -LiteralPath (Join-Path $CudaRoot 'bin/nvcc.exe'))) {
        throw "CudaHome must contain bin/nvcc.exe: $CudaRoot"
    }
    $SdkRoot = if ($WindowsSdk) { (Resolve-Path -LiteralPath $WindowsSdk).Path } else { $null }
    $BuildProject = $Project
    if ($Project -match '[^\x00-\x7F]') {
        $ExistingMappings = (& subst.exe) -join "`n"
        foreach ($Letter in @('Z', 'Y', 'X', 'W')) {
            $Drive = "${Letter}:"
            if ((Get-PSDrive -Name $Letter -ErrorAction SilentlyContinue) -or
                $ExistingMappings -match "(?m)^${Letter}:") { continue }
            & subst.exe $Drive $Project
            if ($LASTEXITCODE -ne 0) { throw "Could not map $Project to $Drive" }
            $OwnedDrive = $Drive
            $BuildProject = "${Drive}\"
            break
        }
        if (-not $OwnedDrive) { throw 'No unused Z/Y/X/W build drive is available.' }
    }

    $Author = Join-Path $BuildProject 'third_party/multi4d'
    $Runtime = Join-Path $BuildProject 'runtime/soft_objects'
    $Extensions = Join-Path $Runtime 'extensions'
    $Build = Join-Path $Runtime 'build'
    $Logs = Join-Path $Runtime 'logs'
    $Temp = Join-Path $Runtime 'temp'
    $TorchExtensions = Join-Path $Runtime 'torch_extensions'
    foreach ($Directory in @($Extensions, $Build, $Logs, $Temp, $TorchExtensions)) {
        New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    }

    $VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $VsRoot = & $VsWhere -latest -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $VsRoot) { throw 'MSVC C++ tools were not found.' }
    $VcVars = Join-Path ($VsRoot | Select-Object -First 1) 'VC/Auxiliary/Build/vcvars64.bat'
    $EnvironmentLines = & $env:ComSpec /d /s /c "call `"$VcVars`" >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) { throw 'vcvars64.bat failed.' }
    foreach ($Line in $EnvironmentLines) {
        if ($Line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
    if ($SdkRoot) {
        $Cpp = Join-Path $SdkRoot 'microsoft.windows.sdk.cpp/c'
        $IncludeRoot = Join-Path $Cpp 'Include'
        $VersionDirectory = Get-ChildItem -LiteralPath $IncludeRoot -Directory |
            Where-Object { $_.Name -match '^\d+(\.\d+)+$' } |
            Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
        if (-not $VersionDirectory) { throw "No SDK version found under $IncludeRoot" }
        $Includes = foreach ($Name in @('ucrt', 'shared', 'um', 'winrt')) {
            Join-Path $VersionDirectory.FullName $Name
        }
        $LibRoot = Join-Path $SdkRoot 'microsoft.windows.sdk.cpp.x64/c'
        Prepend-EnvironmentPath INCLUDE $Includes
        Prepend-EnvironmentPath LIB @(
            (Join-Path $LibRoot 'ucrt/x64'), (Join-Path $LibRoot 'um/x64')
        )
        Prepend-EnvironmentPath PATH @((Join-Path $Cpp "bin/$($VersionDirectory.Name)/x64"))
    }
    $env:CUDA_HOME = $CudaRoot
    $env:CUDA_PATH = $CudaRoot
    Prepend-EnvironmentPath PATH @((Join-Path $CudaRoot 'bin'))
    $env:TEMP = $Temp
    $env:TMP = $Temp
    $env:TORCH_EXTENSIONS_DIR = $TorchExtensions
    $env:MAX_JOBS = '1'
    $env:TORCH_CUDA_ARCH_LIST = '8.9'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:DISTUTILS_USE_SDK = '1'
    $env:VSCMD_ARG_TGT_ARCH = 'x64'
    $env:VSLANG = '1033'

    foreach ($Name in @('rasterizer', 'knn')) {
        if ($Component -notin @('all', $Name)) { continue }
        $SourceName = if ($Name -eq 'rasterizer') { 'diff-gaussian-rasterization' } else { 'simple-knn' }
        $Source = Join-Path $Author "submodules/$SourceName"
        Invoke-LoggedBuild $Name $Source @(
            'setup.py', 'build_ext', '--build-lib', $Extensions,
            '--build-temp', (Join-Path $Build $Name)
        )
        if ($Name -eq 'rasterizer') {
            $Package = Join-Path $Extensions 'diff_gaussian_rasterization'
            New-Item -ItemType Directory -Path $Package -Force | Out-Null
            Copy-Item -LiteralPath (Join-Path $Source 'diff_gaussian_rasterization/__init__.py') `
                -Destination (Join-Path $Package '__init__.py') -Force
        }
    }
    if ($Component -in @('all', 'hybrid')) {
        $Hybrid = Join-Path $Author 'diff_gaussian_rasterization_hybrid'
        $HybridBuild = Join-Path $Extensions 'hybrid'
        New-Item -ItemType Directory -Path $HybridBuild -Force | Out-Null
        $Code = @'
import sys
from pathlib import Path
from torch.utils.cpp_extension import load
root, build = map(Path, sys.argv[1:])
files = ["cuda_rasterizer/rasterizer_impl.cu", "cuda_rasterizer/forward.cu",
         "cuda_rasterizer/backward.cu", "rasterize_points.cu", "ext.cpp"]
load(name="diff_gaussian_rasterization_hybrid",
     sources=[str(root / name) for name in files],
     extra_include_paths=[str(root / "third_party/glm")],
     extra_cuda_cflags=["-g"], build_directory=str(build), verbose=True)
'@
        Invoke-LoggedBuild 'hybrid' $Hybrid @('-c', $Code, $Hybrid, $HybridBuild)
    }
} finally {
    if ($OwnedDrive) { & subst.exe $OwnedDrive /d }
    Get-ChildItem Env: | Where-Object {
        -not $OriginalEnvironment.ContainsKey($_.Name)
    } | ForEach-Object { [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process') }
    foreach ($Name in $OriginalEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $OriginalEnvironment[$Name], 'Process')
    }
}
