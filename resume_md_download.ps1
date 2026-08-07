# resume_md_download.ps1
# MISATO MD.hdf5 resumable download via Zenodo /content (the REAL file endpoint).
#
# WHY /content (not ?download=1):
#   ?download=1 returns a 653-byte HTML stub (browser-only meta-refresh) for any plain GET
#   or full-range request; only a tiny bounded Range returns real bytes. Browsers follow the
#   meta-refresh to /content and download at ~3MB/s on a SINGLE connection.
# WHY curl (not aria2c -x4):
#   Zenodo throttles PER CONNECTION. aria2c -x4 opened 4 connections each capped ~27KiB
#   (aggregate 109KiB/s). A single curl connection gets the full ~3MB/s, like a browser.
# /content serves the raw file on plain GET and honors Range (206) -> resume works.

$misatoDir = "C:\Users\cc\WorkBuddy\2026-07-31-18-03-27\data\raw\misato"
if (-not (Test-Path $misatoDir)) { New-Item -ItemType Directory -Path $misatoDir -Force | Out-Null }
$out = Join-Path $misatoDir "MD.hdf5"
$url = "https://zenodo.org/api/records/7711953/files/MD.hdf5/content"
$expectedMd5 = "9bc6446922cd80e0f2f3f69349bf88ed"

Write-Host "Target: $out"
if (Test-Path $out) {
    $sz = (Get-Item $out).Length
    Write-Host "Found partial ($sz bytes) -> resuming (curl single connection, Range honored)."
    do { curl.exe -C - --retry 999 --retry-all-errors --retry-delay 5 -o $out $url } while ($LASTEXITCODE -ne 0)
} else {
    Write-Host "Fresh download via curl single connection (~3MB/s, matches browser speed)."
    do { curl.exe -r 0- --retry 999 --retry-all-errors --retry-delay 5 -o $out $url } while ($LASTEXITCODE -ne 0)
}

Write-Host ""
Write-Host "Download finished. Verify integrity:"
Write-Host "  certutil -hashfile `"$out`" MD5"
Write-Host "  Expected MD5: $expectedMd5"
