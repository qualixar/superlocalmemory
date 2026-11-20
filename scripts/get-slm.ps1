# SuperLocalMemory one-line installer for 64-bit Windows (PowerShell 5.1 and 7).
#
#   irm https://raw.githubusercontent.com/qualixar/superlocalmemory/main/scripts/get-slm.ps1 | iex
#
# Start the pictures and documents download now (they are on by default and set up
# by themselves on 16 GB or more): `iex` cannot take
# arguments, so set an environment variable first, in the same window:
#
#   $env:SLM_MEDIA = "1"; irm https://raw.githubusercontent.com/qualixar/superlocalmemory/main/scripts/get-slm.ps1 | iex
#
# No Python and no Node are needed.  The script installs `uv` (Astral's tool
# installer) when it is missing, lets uv fetch its own Python 3.12 for
# SuperLocalMemory, then sets everything up, starts it and links the AI tools it
# finds.  It never needs administrator rights and never touches the system Python.
#
# Environment
#   SLM_VERSION=4.2.2     pin an exact release
#   SLM_MEDIA=1           also run `slm media enable --yes`
#   SLM_GET_SKIP_START=1  test hatch: skip starting the service, linking agents,
#                         turning on media and opening the dashboard (steps 5, 6,
#                         7 and 8) so nothing live is touched
#   SLM_GET_DRY_RUN=1     test hatch: print each command as "DRY-RUN: <command>"
#                         instead of running it (also lets the script run on a
#                         non-Windows PowerShell for testing)
#
# When run through `iex` this script never calls `exit`, so your PowerShell
# window stays open; a failure prints the reason and sets $LASTEXITCODE to 1.

$script:UvInstallUrl = 'https://astral.sh/uv/install.ps1'
$script:PackageName = 'superlocalmemory'
$script:DefaultPort = '8765'
$script:Step = 0
$script:Total = 7
$script:PathChanged = $false
$script:Partial = $false
$script:SlmExe = ''

function Test-On([string]$Value) {
    return ($Value -match '^(1|true|yes|on)$')
}

function Test-DryRun { return (Test-On $env:SLM_GET_DRY_RUN) }
function Test-SkipStart { return (Test-On $env:SLM_GET_SKIP_START) }

function Write-Step([string]$Text) {
    $script:Step++
    Write-Host ''
    Write-Host ("Step {0} of {1}: {2}" -f $script:Step, $script:Total, $Text)
}

function Write-Note([string]$Text) {
    Write-Host ("Note: {0}" -f $Text) -ForegroundColor Yellow
}

# Invoke-Uv 'tool install ...'  runs uv, or prints the command in dry-run mode.
function Invoke-Uv([string]$Arguments) {
    if (Test-DryRun) {
        Write-Host ("DRY-RUN: uv {0}" -f $Arguments)
        return 0
    }
    $parts = $Arguments -split ' '
    & uv @parts | Out-Host
    return $LASTEXITCODE
}

# Invoke-Slm 'setup --auto'  runs the installed slm by full path, or prints it.
function Invoke-Slm([string]$Arguments) {
    if (Test-DryRun) {
        Write-Host ("DRY-RUN: slm {0}" -f $Arguments)
        return 0
    }
    $parts = $Arguments -split ' '
    & $script:SlmExe @parts | Out-Host
    return $LASTEXITCODE
}

function Test-PathHas([string]$Dir) {
    $wanted = $Dir.TrimEnd('\', '/')
    foreach ($entry in ($env:Path -split ';')) {
        if ($entry.TrimEnd('\', '/') -ieq $wanted) { return $true }
    }
    return $false
}

function Add-ToPath([string]$Dir) {
    if (-not (Test-PathHas $Dir)) {
        $env:Path = "$Dir;$env:Path"
        $script:PathChanged = $true
    }
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# ---- Step 1 -----------------------------------------------------------------

function Confirm-Platform {
    $onWindows = $true
    if ($PSVersionTable.PSVersion.Major -ge 6) { $onWindows = [bool]$IsWindows }
    if (-not $onWindows) {
        if (Test-DryRun) {
            Write-Note 'not Windows; continuing only because this is a dry run.'
            return
        }
        throw 'this is not Windows. On macOS or Linux paste: curl -fsSL https://raw.githubusercontent.com/qualixar/superlocalmemory/main/scripts/get-slm.sh | sh'
    }
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw '32-bit Windows is not supported. SuperLocalMemory needs 64-bit Windows.'
    }
    $arch = $env:PROCESSOR_ARCHITEW6432
    if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
    if ($arch -notmatch '^(AMD64|ARM64)$') {
        throw "this processor type ($arch) is not supported. SuperLocalMemory needs 64-bit Windows."
    }
    Write-Host "This computer is supported (Windows $arch)."
}

# ---- Step 2 -----------------------------------------------------------------

function Find-UvDir {
    $candidates = @()
    if ($env:UV_INSTALL_DIR) {
        $candidates += $env:UV_INSTALL_DIR
        $candidates += (Join-Path $env:UV_INSTALL_DIR 'bin')
    }
    if ($env:XDG_BIN_HOME) { $candidates += $env:XDG_BIN_HOME }
    if ($env:XDG_DATA_HOME) { $candidates += (Join-Path $env:XDG_DATA_HOME '..\bin') }
    $candidates += (Join-Path $HOME '.local\bin')
    $candidates += (Join-Path $HOME '.cargo\bin')
    foreach ($dir in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $dir 'uv.exe')) { return $dir }
    }
    return $null
}

function Initialize-Uv {
    if (Test-Command 'uv') {
        Write-Host 'uv is already installed.'
        return
    }
    $found = Find-UvDir
    if ($found) {
        Add-ToPath $found
        Write-Host "uv is already installed (found in $found)."
        return
    }
    Write-Host 'Installing uv, the tool that fetches Python for you (no administrator rights needed)...'
    if (Test-DryRun) {
        Write-Host ('DRY-RUN: powershell -ExecutionPolicy ByPass -c "irm {0} | iex"' -f $script:UvInstallUrl)
        return
    }
    & powershell -ExecutionPolicy ByPass -c "irm $script:UvInstallUrl | iex" | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "the official uv installer failed (messages above). Check your internet connection and try again."
    }
    $found = Find-UvDir
    if (-not $found) {
        throw 'uv was installed but I cannot find it. Open a new PowerShell window and paste the line again.'
    }
    Add-ToPath $found
    $script:PathChanged = $true
    if (-not (Test-Command 'uv')) { throw 'uv still cannot be found after installing it.' }
}

# ---- Step 3 -----------------------------------------------------------------

function Get-ToolBinDir {
    $dir = ''
    if ((-not (Test-DryRun)) -or (Test-Command 'uv')) {
        try { $dir = (& uv tool dir --bin 2>$null | Out-String).Trim() } catch { $dir = '' }
    }
    if (-not $dir) {
        if ($env:XDG_BIN_HOME) { $dir = $env:XDG_BIN_HOME } else { $dir = Join-Path $HOME '.local\bin' }
    }
    return $dir
}

function Test-AlreadyInstalled {
    if (-not (Test-Command 'uv')) { return $false }
    $listing = (& uv tool list 2>$null | Out-String)
    return ($listing -match '(?m)^superlocalmemory ')
}

function Install-Slm {
    $toolBin = Get-ToolBinDir
    $code = 0
    if ($env:SLM_VERSION) {
        if ($env:SLM_VERSION -notmatch '^[A-Za-z0-9.+_-]+$') {
            throw "SLM_VERSION may only contain letters, digits and . + _ - (got '$($env:SLM_VERSION)')."
        }
        Write-Host "Installing SuperLocalMemory $($env:SLM_VERSION) (uv fetches Python 3.12 itself)..."
        $code = Invoke-Uv "tool install --python 3.12 --force $($script:PackageName)==$($env:SLM_VERSION)"
    }
    elseif (Test-AlreadyInstalled) {
        Write-Host 'SuperLocalMemory is already installed. Upgrading it...'
        $code = Invoke-Uv "tool upgrade $($script:PackageName)"
    }
    else {
        Write-Host 'Installing SuperLocalMemory (uv fetches Python 3.12 itself; this can take a few minutes)...'
        $code = Invoke-Uv "tool install --python 3.12 $($script:PackageName)"
    }
    if ($code -ne 0) {
        throw 'uv could not install SuperLocalMemory (messages above). Check your internet connection and try again.'
    }
    if (-not (Test-PathHas $toolBin)) {
        # update-shell first: uv skips the edit when the folder is already on PATH.
        $updated = Invoke-Uv 'tool update-shell'
        if ($updated -ne 0) {
            Write-Note "I could not update your PATH automatically. Add $toolBin to it yourself."
        }
        Add-ToPath $toolBin
    }
    $script:SlmExe = Join-Path $toolBin 'slm.exe'
    if ((-not (Test-DryRun)) -and (-not (Test-Path -LiteralPath $script:SlmExe))) {
        throw "the install finished but slm.exe is missing from $toolBin."
    }
}

# ---- Steps 4 to 8 -----------------------------------------------------------

function Initialize-Slm {
    if ((Invoke-Slm 'setup --auto') -ne 0) {
        throw "'slm setup --auto' failed (messages above). Run 'slm doctor' to see what is wrong."
    }
}

function Start-Slm {
    if ((Invoke-Slm 'serve start') -ne 0) {
        throw "SuperLocalMemory would not start (messages above). Run 'slm doctor' to see what is wrong."
    }
    Write-Host 'Loading the text model (the first time this downloads about 500 MB)...'
    if ((Invoke-Slm 'warmup') -ne 0) {
        throw "the text model did not load (messages above). Run 'slm doctor' to see what is wrong."
    }
}

function Join-Agents {
    if ((Invoke-Slm 'connect') -ne 0) {
        $script:Partial = $true
        Write-Note "I could not link every AI tool. Run 'slm connect' again later, or 'slm connect --list' to see what is supported."
    }
}

function Enable-Media {
    if ((Invoke-Slm 'media enable --yes') -eq 0) {
        Write-Host "Pictures and documents: the extra models download in the background. Check with 'slm media status'."
    }
    else {
        Write-Note 'Pictures and documents were not turned on (reason above). Text memory works fine without them.'
    }
}

function Get-DashboardUrl {
    $port = ''
    try {
        $toolRoot = (& uv tool dir 2>$null | Out-String).Trim()
        $py = Join-Path $toolRoot "$($script:PackageName)\Scripts\python.exe"
        if (Test-Path -LiteralPath $py) {
            $port = (& $py -c 'from superlocalmemory.cli.deployment import runtime_port; print(runtime_port())' 2>$null | Out-String).Trim()
        }
    } catch { $port = '' }
    if ($port -notmatch '^\d+$') {
        if ($env:SLM_DAEMON_PORT) { $port = $env:SLM_DAEMON_PORT } else { $port = $script:DefaultPort }
    }
    return "http://localhost:$port"
}

function Complete-Install {
    if (Test-DryRun) {
        [void](Invoke-Slm 'doctor --quick')
        [void](Invoke-Slm 'dashboard')
        return
    }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $report = (& $script:SlmExe doctor --quick 2>&1 | ForEach-Object { "$_" }) -join "`n"
    $doctorCode = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($doctorCode -ne 0) { Write-Note "the health check reported a problem. Run 'slm doctor' to see it." }
    $summary = ($report -split "`n" | Where-Object { $_ -match '^Summary:' } | Select-Object -Last 1)
    if (-not $summary) { $summary = ($report -split "`n" | Select-Object -Last 1) }
    Write-Host "Health check: $summary"
    $url = Get-DashboardUrl
    try {
        & $script:SlmExe dashboard 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Note 'I could not open the dashboard in your browser. Open it yourself:' }
    } catch {
        Write-Note 'I could not open the dashboard in your browser. Open it yourself:'
    }
    Write-Host "Dashboard: $url"
}

function Install-SuperLocalMemory {
    if (Test-On $env:SLM_MEDIA) { $script:Total = 8 }
    $wantMedia = ($script:Total -eq 8)

    Write-Host 'SuperLocalMemory installer'

    Write-Step 'checking your computer...'
    Confirm-Platform

    Write-Step 'getting uv (the tool that installs SuperLocalMemory)...'
    Initialize-Uv

    Write-Step 'installing SuperLocalMemory...'
    Install-Slm

    Write-Step 'setting up SuperLocalMemory on this computer...'
    Initialize-Slm

    if (Test-SkipStart) {
        Write-Step 'starting SuperLocalMemory (skipped: SLM_GET_SKIP_START)'
        Write-Step 'linking your AI tools (skipped: SLM_GET_SKIP_START)'
        if ($wantMedia) { Write-Step 'pictures and documents (skipped: SLM_GET_SKIP_START)' }
        Write-Step 'finishing (skipped: SLM_GET_SKIP_START)'
    }
    else {
        Write-Step 'starting SuperLocalMemory and loading the text model...'
        Start-Slm

        Write-Step 'linking the AI tools I can find (Claude, Cursor, Codex and others)...'
        Join-Agents

        if ($wantMedia) {
            Write-Step 'turning on pictures and documents...'
            Enable-Media
        }

        Write-Step 'finishing up...'
        Complete-Install
    }

    Write-Host ''
    if ($script:Partial) {
        Write-Host 'SuperLocalMemory is installed, but one step needs attention (see the note above).'
    }
    else {
        Write-Host 'Done. SuperLocalMemory is installed.'
    }
    if ($script:PathChanged) {
        Write-Host "Open a new PowerShell window so the 'slm' command works everywhere."
    }
    if ($script:Partial) { return 1 }
    return 0
}

function Start-SlmInstall {
    $exitCode = 0
    try {
        $exitCode = Install-SuperLocalMemory
    }
    catch {
        Write-Host ''
        Write-Host ("Sorry, the install stopped: {0}" -f $_.Exception.Message) -ForegroundColor Red
        $exitCode = 1
    }
    $global:LASTEXITCODE = $exitCode
    # Only a real script file may exit; under `irm | iex` exit would close the window.
    if ($PSCommandPath) { exit $exitCode }
}

Start-SlmInstall
