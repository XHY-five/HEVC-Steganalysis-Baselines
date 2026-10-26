# Provision the original Tar5Stego custom executables into THIS project.
# Reuse a LOCAL copy if available; otherwise download the exact repository files.
param(
  [string]$Tar5Path = 'E:\xhy\Tar5Stego',
  [switch]$WithRDCost
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$files = @(
  @{ source = 'C_Coder/x265.exe'; target = 'bin/C_Coder/x265.exe' },
  @{ source = 'Decoder/TAppDecoder.exe'; target = 'bin/Decoder/TAppDecoder.exe' }
)
if ($WithRDCost) {
  $files += @{ source='extractRDcostEncoder/x265.exe'; target='bin/extractRDcostEncoder/x265.exe' }
}
foreach ($item in $files) {
  $dest = Join-Path $root ($item.target.Replace('/', '\'))
  $folder = Split-Path -Parent $dest
  New-Item -ItemType Directory -Force -Path $folder | Out-Null
  if ((Test-Path $dest) -and (Get-Item $dest).Length -gt 100000) {
    Write-Host "[SKIP] Existing: $dest"; continue
  }
  $local = Join-Path $Tar5Path ($item.source.Replace('/', '\'))
  if (Test-Path $local) {
    Copy-Item -LiteralPath $local -Destination $dest -Force
    Write-Host "[COPY] $local -> $dest"
  } else {
    $url = 'https://raw.githubusercontent.com/XHY-five/Tar5Stego/main/' + $item.source
    Write-Host "[DOWNLOAD] $url"
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
  }
  $fs = [System.IO.File]::OpenRead($dest)
  try { $first = $fs.ReadByte(); $second = $fs.ReadByte() }
  finally { $fs.Dispose() }
  if ($first -ne 77 -or $second -ne 90 -or (Get-Item $dest).Length -lt 100000) {
    Remove-Item -LiteralPath $dest -Force
    throw "Downloaded/copied file is not a valid-size Windows PE executable: $dest"
  }
}
Write-Host '[OK] Wang now has its OWN independent local encoder/decoder binaries.'
