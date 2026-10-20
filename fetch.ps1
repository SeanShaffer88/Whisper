# Download the pinned upstream sources into vendor/ and verify SHA-256.
# Sources are NEVER committed; vendor/ is gitignored.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pins = @{}
Get-Content pins.txt | Where-Object { $_ -match '=' -and $_ -notmatch '^#' } | ForEach-Object {
    $k, $v = $_ -split '=', 2; $pins[$k.Trim()] = $v.Trim()
}
New-Item -ItemType Directory -Force vendor | Out-Null

function Fetch([string]$Url, [string]$File, [string]$Want) {
    $path = Join-Path vendor $File
    if (-not (Test-Path $path)) {
        Write-Host "downloading $Url"
        Invoke-WebRequest -Uri $Url -OutFile $path
    }
    $got = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $Want) {
        throw "SHA-256 MISMATCH for $File`n  pinned: $Want`n  actual: $got"
    }
    Write-Host "verified $File ($got)"
}

Fetch $pins['WHISPER_URL'] "whisper-$($pins['WHISPER_VERSION']).tar.gz" $pins['WHISPER_SHA256']
Fetch $pins['TEST_MODEL_URL'] $pins['TEST_MODEL_NAME'] $pins['TEST_MODEL_SHA256']

Remove-Item -Recurse -Force vendor/src-whisper, vendor/tmp-whisper -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force vendor/tmp-whisper | Out-Null
# Windows 10 1803+ ships tar.exe, which handles .tar.gz (Expand-Archive cannot).
tar -xzf "vendor/whisper-$($pins['WHISPER_VERSION']).tar.gz" -C vendor/tmp-whisper
Move-Item "vendor/tmp-whisper/$($pins['WHISPER_DIR'])" vendor/src-whisper
Remove-Item -Recurse -Force vendor/tmp-whisper
# Tar extraction preserves the release-date mtime, which cargo's
# build-script fingerprint can mistake for "unchanged" if a build ran
# while the tree was absent. Stamp the header so the probe re-runs.
(Get-Item vendor/src-whisper/include/whisper.h).LastWriteTime = Get-Date
Write-Host "sources ready in vendor/src-whisper; test model at vendor/$($pins['TEST_MODEL_NAME'])"
