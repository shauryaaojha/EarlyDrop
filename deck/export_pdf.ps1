# Renders the deck to EarlyDrop_Deck.pdf (one slide per page, final animation state) with headless Edge.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$edge = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) { Write-Error "Microsoft Edge not found"; exit 1 }
$url = "file:///" + $here.Replace('\', '/') + "/index.html?print"
$out = Join-Path $here "EarlyDrop_Deck.pdf"
$profileDir = Join-Path $env:TEMP "earlydrop-pdf-profile"
& $edge --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=20000 --user-data-dir="$profileDir" --print-to-pdf="$out" $url 2>$null | Out-Null
Write-Host "Wrote $out"
