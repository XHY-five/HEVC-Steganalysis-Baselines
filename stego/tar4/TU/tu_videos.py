"""Discover Tar4 YUV clips, retaining original <name>_<WxH>_<fps>_segmentNNN.yuv naming."""
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Video:
    input: Path
    key: str
    width: int
    height: int
    fps: int
    frames: int


def discover(root: Path, requested_frames: int) -> list[Video]:
    if not root.is_dir():
        raise FileNotFoundError(
            f'Missing original YUV video directory: {root}')
    clips = []
    used = set()
    for path in sorted(root.rglob('*.yuv')):
        parts = path.stem.split('_')
        if len(parts) < 4 or not re.fullmatch(
                r'\d+x\d+', parts[1]) or not parts[2].isdigit():
            print('[skip] Unexpected filename:', path.name)
            continue
        w, h = map(int, parts[1].split('x'))
        fps = int(parts[2])
        if min(w, h, fps) <= 0 or w % 2 or h % 2:
            raise ValueError('YUV420 requires even dimensions and valid fps')
        key = f'{parts[0]}segment{parts[-1][-3:]}'
        if key in used:
            raise ValueError(f'Duplicate Tar4 clip key: {key}')
        used.add(key)
        frame_bytes = w * h * 3 // 2
        if path.stat().st_size % frame_bytes:
            raise ValueError(f'YUV420 file size mismatch: {path}')
        frames = min(requested_frames, path.stat().st_size // frame_bytes)
        if frames < 1:
            raise ValueError(f'No frames: {path}')
        clips.append(Video(path, key, w, h, fps, frames))
    if not clips:
        raise FileNotFoundError(f'No valid YUV clips found in {root}')
    return clips
