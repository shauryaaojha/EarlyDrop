# Renders a deck to PDF (one slide per page, final animation state) with headless Edge.
#   powershell -File deck/export_pdf.ps1                                   # PBL-I/II deck -> EarlyDrop_Deck.pdf
#   powershell -File deck/export_pdf.ps1 -Page pbl3.html -Out EarlyDrop_PBL3_Deck.pdf
param([string]$Page = "index.html", [string]$Out = "EarlyDrop_Deck.pdf")
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$edge = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) { Write-Error "Microsoft Edge not found"; exit 1 }
$url = "file:///" + $here.Replace('\', '/') + "/$Page" + "?print"
$outPath = Join-Path $here $Out
$profileDir = Join-Path $env:TEMP "earlydrop-pdf-profile"
& $edge --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=20000 --user-data-dir="$profileDir" --print-to-pdf="$outPath" $url 2>$null | Out-Null
Write-Host "Wrote $outPath"
