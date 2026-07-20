# SuperLocalMemory Windows installer
# Copyright (c) 2026 Varun Pratap Bhardwaj
# SPDX-License-Identifier: AGPL-3.0-or-later

[CmdletBinding()]
param(
    [ValidateSet("Install", "Upgrade", "Uninstall")]
    [string]$Action = "Install",

    [ValidateSet("Auto", "uv", "pipx")]
    [string]$ToolManager = "Auto",

    [ValidatePattern('^[0-9]+(?:\.[0-9]+){1,3}(?:(?:a|b|rc)[0-9]+)?$')]
    [string]$Version,

    [string]$Package,

    [switch]$DryRun,
    [switch]$NonInteractive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PackageName = "superlocalmemory"

function Resolve-ToolManager {
    param(
        [string]$RequestedManager,
        [string]$RequestedAction,
        [bool]$AllowMissing
    )

    if ($RequestedManager -ne "Auto") {
        $command = Get-Command -Name $RequestedManager -CommandType Application -ErrorAction SilentlyContinue
        if ($null -ne $command -or $AllowMissing) {
            return $RequestedManager
        }
        throw "$RequestedManager is not available on PATH."
    }

    $availableManagers = @(
        @("uv", "pipx") | Where-Object {
            $null -ne (Get-Command -Name $_ -CommandType Application -ErrorAction SilentlyContinue)
        }
    )

    if ($RequestedAction -eq "Install") {
        if ($availableManagers.Count -gt 0) {
            return $availableManagers[0]
        }
        throw @"
No supported isolated Python tool manager was found.
Install uv with: winget install --id astral-sh.uv -e
Or install pipx with: scoop install pipx
Then run this installer again. The installer will not install a tool manager for you.
"@
    }

    $owners = @(
        $availableManagers | Where-Object { Test-ManagerOwnsPackage -Manager $_ }
    )
    if ($owners.Count -gt 1) {
        throw "both uv and pipx own an installation; choose one with -ToolManager."
    }
    if ($owners.Count -eq 1) {
        return $owners[0]
    }
    throw "no isolated installation was found; run Install first or specify -ToolManager."
}

function Test-ManagerOwnsPackage {
    param([string]$Manager)

    $listing = if ($Manager -eq "uv") {
        & uv tool list 2>$null
    } else {
        & pipx list --short 2>$null
    }

    return [bool]($listing -match '^\s*superlocalmemory(?:\s|$)')
}

function Get-LifecycleArguments {
    param(
        [string]$Manager,
        [string]$RequestedAction,
        [string]$PackageSpec
    )

    if ($Manager -eq "uv") {
        switch ($RequestedAction) {
            "Install" { return @("tool", "install", $packageSpec) }
            "Upgrade" { return @("tool", "upgrade", $PackageName) }
            "Uninstall" { return @("tool", "uninstall", $PackageName) }
        }
    }

    switch ($RequestedAction) {
        "Install" { return @("install", $packageSpec) }
        "Upgrade" { return @("upgrade", $PackageName) }
        "Uninstall" { return @("uninstall", $PackageName) }
    }

    throw "Unsupported lifecycle request."
}

if ($Version -and $Package) {
    throw "-Version and -Package cannot be combined. Pass one exact install source."
}

if ($Package -and ($Package.StartsWith("-") -or $Package.Contains("`r") -or $Package.Contains("`n"))) {
    throw "-Package must be one package name, project directory, or wheel path."
}

if (($Version -or $Package) -and $Action -ne "Install") {
    throw "-Version and -Package are valid only with -Action Install."
}

$resolvedManager = Resolve-ToolManager `
    -RequestedManager $ToolManager `
    -RequestedAction $Action `
    -AllowMissing ($DryRun.IsPresent)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$projectMetadata = Join-Path $ProjectRoot "pyproject.toml"

$packageSpec = if ($Package) {
    $Package
} elseif ($Version) {
    "$PackageName==$Version"
} elseif (Test-Path -LiteralPath $projectMetadata -PathType Leaf) {
    $ProjectRoot
} else {
    $PackageName
}

$arguments = Get-LifecycleArguments `
    -Manager $resolvedManager `
    -RequestedAction $Action `
    -PackageSpec $packageSpec

$commandPreview = "$resolvedManager $($arguments -join ' ')"

if ($DryRun) {
    Write-Output "[dry-run] $commandPreview"
    Write-Output "No command was executed."
    return
}

Write-Output "SuperLocalMemory Windows installer"
Write-Output "Action: $Action"
Write-Output "Tool manager: $resolvedManager"
Write-Output "Command: $commandPreview"

& $resolvedManager @arguments
$managerExitCode = $LASTEXITCODE

if ($managerExitCode -ne 0) {
    throw "$resolvedManager failed with exit code $managerExitCode."
}

Write-Output "SuperLocalMemory $($Action.ToLowerInvariant()) completed successfully."
Write-Output "Run 'slm --help' to verify the command from a new terminal session."
