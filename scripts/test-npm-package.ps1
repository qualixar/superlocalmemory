# Verify the npm artifact without changing the machine's global npm state.

param()

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("slm-npm-pack-" + [guid]::NewGuid().ToString("N"))
$Tarball = $null
New-Item -ItemType Directory -Path $TempDir | Out-Null

try {
    $RequiredFiles = @(
        "package.json",
        "pyproject.toml",
        "bin/slm-npm",
        "scripts/postinstall.js",
        "scripts/preuninstall.js",
        "scripts/install.sh",
        "scripts/install.ps1",
        "README.md",
        "LICENSE",
        "NOTICE",
        "ATTRIBUTION.md"
    )

    foreach ($RelativePath in $RequiredFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $RepoDir $RelativePath) -PathType Leaf)) {
            throw "Missing required repository file: $RelativePath"
        }
    }

    if ((Test-Path -LiteralPath (Join-Path $RepoDir 'install.sh')) -or
        (Test-Path -LiteralPath (Join-Path $RepoDir 'install.ps1'))) {
        throw "Repository lifecycle installers must stay scoped under scripts/."
    }

    $Package = Get-Content -Raw -LiteralPath (Join-Path $RepoDir "package.json") | ConvertFrom-Json
    foreach ($Field in @("name", "version", "description", "author", "license", "repository", "bin")) {
        if (-not $Package.$Field) { throw "package.json is missing $Field" }
    }
    if ($Package.bin.slm -ne "./bin/slm-npm") { throw "Unexpected slm wrapper" }
    if ($Package.scripts.postinstall -ne "node scripts/postinstall.js") { throw "Unexpected postinstall" }

    Push-Location $RepoDir
    try {
        $PackResult = (npm pack --json --ignore-scripts --pack-destination $TempDir | Out-String) | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0) { throw "npm pack failed" }
    } finally {
        Pop-Location
    }

    if ($PackResult.Count -ne 1 -or -not $PackResult[0].filename) {
        throw "npm pack returned an unexpected result"
    }
    $Tarball = Join-Path $TempDir $PackResult[0].filename
    if (-not (Test-Path -LiteralPath $Tarball -PathType Leaf)) {
        throw "npm pack did not create the declared artifact"
    }

    $Contents = @(tar -tzf $Tarball)
    foreach ($PackagedPath in @("package/bin/slm-npm", "package/scripts/postinstall.js", "package/pyproject.toml")) {
        if ($PackagedPath -notin $Contents) { throw "Missing packaged runtime file: $PackagedPath" }
    }
    foreach ($ForbiddenPath in @("package/scripts/install.sh", "package/scripts/install.ps1", "package/install.sh", "package/install.ps1")) {
        if ($ForbiddenPath -in $Contents) { throw "Repository lifecycle installer leaked into npm artifact: $ForbiddenPath" }
    }

    node --test (Join-Path $RepoDir "tests/postinstall/test_npm_runtime_isolation.js")
    if ($LASTEXITCODE -ne 0) { throw "npm runtime-isolation test failed" }
    Write-Host "npm artifact contract verified: $($PackResult[0].filename)"
} finally {
    if ($Tarball -and (Test-Path -LiteralPath $Tarball)) {
        Remove-Item -LiteralPath $Tarball -Force
    }
    if (Test-Path -LiteralPath $TempDir) {
        Remove-Item -LiteralPath $TempDir -Force
    }
}
