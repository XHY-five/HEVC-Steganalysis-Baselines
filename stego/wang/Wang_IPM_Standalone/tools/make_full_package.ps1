# Builds an offline standalone ZIP with the two custom Windows executables.
# External YUV files, cover data, and outputs are intentionally excluded.
param([string]$Tar5Path = 'E:\xhy\Tar5Stego')
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'setup_tools.ps1') -Tar5Path $Tar5Path
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$parent = Split-Path -Parent $root
$dst = Join-Path $parent 'Wang_IPM_Standalone_FULL.zip'
$stageParent = Join-Path $env:TEMP ('WangStage_' + [guid]::NewGuid().ToString('N'))
$stage = Join-Path $stageParent 'Wang_IPM_Standalone'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
try {
  foreach ($dir in @('algorithm','bin','tools','examples')) {
    Copy-Item -LiteralPath (Join-Path $root $dir) -Destination $stage -Recurse -Force
  }
  foreach ($file in @('run_pipeline.py','README.md','.gitignore')) {
    Copy-Item -LiteralPath (Join-Path $root $file) -Destination $stage -Force
  }
  foreach ($dir in @('input\cover_ipm','input\videos','outputs')) {
    New-Item -ItemType Directory -Path (Join-Path $stage $dir) -Force | Out-Null
  }
  if (Test-Path $dst) { Remove-Item -Force $dst }
  Compress-Archive -Path $stage -DestinationPath $dst -CompressionLevel Optimal
  Write-Host "[OK] Self-contained ZIP (including customized EXEs): $dst"
} finally {
  Remove-Item -LiteralPath $stageParent -Recurse -Force -ErrorAction SilentlyContinue
}
