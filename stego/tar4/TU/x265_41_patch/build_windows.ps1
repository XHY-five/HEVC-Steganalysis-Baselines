# Robust Windows build script for the TU-only x265 4.1 patch.
# Downloads the exact official x265 4.1 release archive, validates SHA-256,
# applies the local patch, and builds the x265 CLI with Visual Studio 2022.
param([switch]$ForceDownload)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$tar4 = (Resolve-Path (Join-Path $here '../..')).Path
$work = Join-Path $here 'build_work'
$archive = Join-Path $work 'x265_4.1.tar.gz'
$hashExpected = 'A31699C6A89806B74B0151E5E6A7DF65DE4B49050482FE5EBF8A4379D7AF8F29'
New-Item -ItemType Directory -Force -Path $work | Out-Null

function Test-OfficialArchive([string]$Path) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToUpperInvariant()
        return ($actual -eq $hashExpected)
    } catch { return $false }
}

$needDownload = $ForceDownload -or !(Test-OfficialArchive $archive)
if ($needDownload) {
    if (Test-Path -LiteralPath $archive) {
        Write-Warning 'Existing x265_4.1.tar.gz is incomplete or is an HTML redirect page; deleting it before retrying.'
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
    }

    # Direct mirror URLs from the VideoLAN download page. Do not use get.videolan.org
    # as a file URL: it may return an HTML interstitial page instead of the archive.
    $mirrors = @(
      'https://veronanetworks.mm.fcix.net/videolan-ftp/x265/x265_4.1.tar.gz',
      'https://mirror.clarkson.edu/videolan/x265/x265_4.1.tar.gz',
      'https://mirror.lstn.net/videolan/x265/x265_4.1.tar.gz',
      'https://southfront.mm.fcix.net/videolan-ftp/x265/x265_4.1.tar.gz',
      'https://github.com/Multicorewareinc/x265/releases/download/4.1/x265_4.1.tar.gz'
    )
    $ok = $false
    $curl = Get-Command 'curl.exe' -ErrorAction SilentlyContinue
    foreach ($url in $mirrors) {
      try {
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
        Write-Host "Downloading x265 4.1: $url"
        if ($curl) {
          & $curl.Source --location --fail --show-error --retry 2 --connect-timeout 20 --max-time 300 --output $archive $url
          if ($LASTEXITCODE -ne 0) { throw "curl.exe exited with code $LASTEXITCODE" }
        } else {
          Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing -TimeoutSec 300
        }
        if (Test-OfficialArchive $archive) {
          Write-Host 'SHA-256 verified: official x265 4.1 source archive.'
          $ok = $true
          break
        }
        $actual = if (Test-Path -LiteralPath $archive) { (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash } else { '<file missing>' }
        Write-Warning "Not the expected source archive (SHA-256: $actual); trying next mirror."
      } catch {
        Write-Warning $_.Exception.Message
      }
    }
    if (!$ok) {
      throw "Could not download the verified x265 4.1 archive. Download it in a browser from https://get.videolan.org/x265/x265_4.1.tar.gz (choose a direct mirror), save it as $archive, then rerun this script. Expected SHA-256: $hashExpected"
    }
}
if (!(Test-OfficialArchive $archive)) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash
    throw "x265_4.1.tar.gz SHA-256 mismatch: $actual; expected $hashExpected"
}

$extract = Join-Path $work 'source_download'
$sourceFiles = @()
if (Test-Path -LiteralPath $extract) {
    $sourceFiles = @(Get-ChildItem -Path $extract -Filter search.cpp -File -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match '[\\/]source[\\/]encoder[\\/]search\.cpp$' })
}
if ($sourceFiles.Count -ne 1) {
    if (Test-Path -LiteralPath $extract) { Remove-Item -LiteralPath $extract -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    & tar.exe -xzf $archive -C $extract
    if ($LASTEXITCODE -ne 0) { throw 'tar.exe extraction failed' }
    $sourceFiles = @(Get-ChildItem -Path $extract -Filter search.cpp -File -Recurse | Where-Object { $_.FullName -match '[\\/]source[\\/]encoder[\\/]search\.cpp$' })
}
if ($sourceFiles.Count -ne 1) { throw "Expected one x265 source/encoder/search.cpp, found $($sourceFiles.Count)" }
$src = (Resolve-Path (Join-Path $sourceFiles[0].Directory.FullName '../..')).Path
Write-Host "Source directory: $src"
python (Join-Path $here 'apply_patch.py') $src
if ($LASTEXITCODE -ne 0) { throw 'apply_patch.py failed' }
$build = Join-Path $work 'build_vs2022'
cmake -S (Join-Path $src 'source') -B $build -G 'Visual Studio 17 2022' -A x64 `
 -DENABLE_CLI=ON -DENABLE_SHARED=OFF -DENABLE_ASSEMBLY=OFF -DENABLE_TESTS=OFF
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed; install VS 2022 C++ Build Tools + Windows SDK + CMake' }
cmake --build $build --config Release --target x265 --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'x265 CLI compilation failed' }
$bins = @(Get-ChildItem -Path $build -Filter x265.exe -Recurse -File)
if ($bins.Count -eq 0) { throw 'No compiled x265.exe found in CMake output' }
$bin = $bins | Sort-Object Length -Descending | Select-Object -First 1
foreach ($subdir in @('C_Coder','extractencoder')) {
  $dest = Join-Path (Join-Path $tar4 $subdir) 'x265_TU.exe'
  Copy-Item $bin.FullName $dest -Force
  Write-Host "Built: $dest"
}
& (Join-Path $tar4 'C_Coder/x265_TU.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'Built binary failed --version smoke test' }
Write-Host '[SUCCESS] TU x265 binary compiled. Next: python run_TU.py --keyint 1 --frames 30 --ratio 0.3'
