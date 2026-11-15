#Requires -Version 5.1
<#
.SYNOPSIS
Preview or permanently delete the explicitly approved legacy reconstruction files.
.EXAMPLE
    .\scripts\cleanup_legacy.ps1
.EXAMPLE
    .\scripts\cleanup_legacy.ps1 -Execute
#>
[CmdletBinding()]
param([switch]$Execute)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\', '/')
$separator = [IO.Path]::DirectorySeparatorChar
$comparison = [StringComparison]::OrdinalIgnoreCase
$workRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'work'))
$keepRoot = [IO.Path]::GetFullPath(
    (Join-Path $projectRoot 'work/mjj/two_person/data/quaternion_fit/prior_control')
)

foreach ($marker in @('pipeline.py', 'pyproject.toml', 'src/hoi')) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $marker))) {
        throw "Project marker missing: $marker. No deletion performed."
    }
}
if (-not (Test-Path -LiteralPath $keepRoot -PathType Container)) {
    throw "Protected directory missing: $keepRoot. No deletion performed."
}

$approvedPaths = @(
    'lhmpp/runs',
    'lhmpp/inputs',
    'scripts/hoi_paths.py',
    'scripts/avatar_rig.py',
    'scripts/mesh_renderer.py',
    'scripts/hybrid_compositor.py',
    'scripts/render_gtcam_video.py',
    'scripts/render_orbit_video.py',
    'scripts/compose_four_views.py'
) | ForEach-Object { [IO.Path]::GetFullPath((Join-Path $projectRoot $_)) }

function Assert-OrdinaryPath([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Link/reparse point encountered: $Path. No deletion performed."
    }
}

# Check links before descending; never follow a link outside the approved trees.
foreach ($ancestor in @($projectRoot, (Join-Path $projectRoot 'lhmpp'),
        (Join-Path $projectRoot 'scripts'))) {
    Assert-OrdinaryPath $ancestor
}
foreach ($root in @($workRoot) + $approvedPaths) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    Assert-OrdinaryPath $root
    if (Test-Path -LiteralPath $root -PathType Container) {
        $links = @(Get-ChildItem -LiteralPath $root -Force -Recurse -Attributes ReparsePoint)
        if ($links.Count -gt 0) {
            throw "Link/reparse point encountered: $($links[0].FullName). No deletion performed."
        }
    }
}

$deletePaths = [Collections.Generic.List[string]]::new()
function Add-WorkTargets([string]$Path) {
    $candidate = [IO.Path]::GetFullPath($Path)
    if ($candidate.Equals($keepRoot, $comparison)) { return }
    if ($keepRoot.StartsWith($candidate + $separator, $comparison)) {
        foreach ($child in Get-ChildItem -LiteralPath $candidate -Force) {
            Add-WorkTargets $child.FullName
        }
    } else {
        $deletePaths.Add($candidate)
    }
}
Add-WorkTargets $workRoot
foreach ($path in $approvedPaths) {
    if (Test-Path -LiteralPath $path) { $deletePaths.Add($path) }
}

# Validate the entire plan before the first removal.
foreach ($path in $deletePaths) {
    if (-not $path.StartsWith($projectRoot + $separator, $comparison)) {
        throw "Target outside project: $path. No deletion performed."
    }
    $insideWork = $path.StartsWith($workRoot + $separator, $comparison)
    if (-not $insideWork -and $approvedPaths -notcontains $path) {
        throw "Target outside approved scope: $path. No deletion performed."
    }
    if ($path.Equals($keepRoot, $comparison) -or
        $keepRoot.StartsWith($path + $separator, $comparison)) {
        throw "Protected directory or ancestor in removal plan: $path. No deletion performed."
    }
}

Write-Host "Project:  $projectRoot"
Write-Host "KEEP:     $keepRoot"
Write-Host 'Original data, assets, runtime, selected runs, reports and Git history are outside this plan.'
Write-Host "Removal targets: $($deletePaths.Count)"
foreach ($path in $deletePaths) { Write-Host "  REMOVE $path" }
if (-not $Execute) {
    Write-Host ''
    Write-Host 'PREVIEW ONLY. Run again with -Execute to permanently remove the listed targets.'
    return
}

function Get-ProtectedFingerprint {
    Get-ChildItem -LiteralPath $keepRoot -File -Force -Recurse |
        Sort-Object FullName |
        ForEach-Object {
            $_.FullName.Substring($keepRoot.Length) + '|' + $_.Length + '|' +
                (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
}

$protectedBefore = @(Get-ProtectedFingerprint)
$driveName = ([IO.Path]::GetPathRoot($projectRoot)).Substring(0, 1)
$freeBefore = (Get-PSDrive -Name $driveName).Free
$removed = 0
try {
    foreach ($path in $deletePaths) {
        Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
        $removed++
        Write-Host "Removed [$removed/$($deletePaths.Count)]: $path"
    }
} finally {
    $protectedAfter = @(Get-ProtectedFingerprint)
    if ($protectedBefore.Count -ne $protectedAfter.Count -or
        ($protectedBefore.Count -gt 0 -and
         @(Compare-Object $protectedBefore $protectedAfter).Count -ne 0)) {
        throw 'Protected prior_control fingerprint changed. Inspect it before continuing.'
    }
    Write-Host "Protected prior_control files verified: $($protectedAfter.Count)"
}

$remaining = @($deletePaths | Where-Object { Test-Path -LiteralPath $_ })
if ($remaining.Count -gt 0) { throw "Removal incomplete: $($remaining -join ', ')" }
$freedBytes = (Get-PSDrive -Name $driveName).Free - $freeBefore
Write-Host ('Completed. Free-space change: {0:N2} GiB ({1} bytes).' -f ($freedBytes / 1GB), $freedBytes)
