#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="$HERE/build_work"
ARCHIVE="$WORK/x265_4.1.tar.gz"
mkdir -p "$WORK"
if [[ ! -f "$ARCHIVE" ]]; then
  curl -L --fail --retry 3 'https://get.videolan.org/x265/x265_4.1.tar.gz' -o "$ARCHIVE"
fi
echo 'a31699c6a89806b74b0151e5e6a7df65de4b49050482fe5ebf8a4379d7af8f29  '"$ARCHIVE" | sha256sum -c -
mkdir -p "$WORK/unpacked"
tar -xf "$ARCHIVE" -C "$WORK/unpacked"
mapfile -t files < <(find "$WORK/unpacked" -path '*/source/encoder/search.cpp')
[[ ${#files[@]} -eq 1 ]] || { echo 'Cannot uniquely locate x265 source' >&2; exit 1; }
SRC="$(cd "$(dirname "${files[0]}")/../.." && pwd)"
python3 "$HERE/apply_patch.py" "$SRC"
cmake -S "$SRC/source" -B "$WORK/build_linux" -DENABLE_SHARED=OFF -DENABLE_CLI=ON -DENABLE_ASSEMBLY=OFF
cmake --build "$WORK/build_linux" -j 4 --target x265
cp "$WORK/build_linux/x265" "$ROOT/C_Coder/x265_TU_linux"
cp "$WORK/build_linux/x265" "$ROOT/extractencoder/x265_TU_linux"
"$ROOT/C_Coder/x265_TU_linux" --version
