$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $root
try { python (Join-Path $root 'algorithm\test_wang_ipm.py') }
finally { Pop-Location }
