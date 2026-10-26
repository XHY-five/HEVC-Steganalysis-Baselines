"""Unit + smoke tests for wang_ipm.py; runs without x265."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import wang_ipm


def test_angle_mapping():
    assert wang_ipm.angle_bits(10, 10) == (0, 0)
    assert wang_ipm.angle_bits(10, 12) == (0, 1)
    assert wang_ipm.angle_bits(10, 14) == (1, 0)
    assert wang_ipm.angle_bits(10, 16) == (1, 1)
    assert wang_ipm.angle_bits(10, 18) == (0, 0)
    assert wang_ipm.angle_bits(2, 34) == (1, 1)
    for a in range(2, 35):
        for sym in [(0,0),(0,1),(1,0),(1,1)]:
            choices = [b for b in range(2, 35) if wang_ipm.angle_bits(a,b) == sym]
            assert choices, (a,sym)


def synthetic_cover(path):
    lines = []
    for fid in [1,2,3,4,5]:
        lines.append(f"Frame {fid} begin")
        for i in range(56):
            mode = [0,1,2,15,18,34,3,26][i % 8]
            # Make some mixed pairs, some DC/Planar pairs and some directional pairs.
            if i%8 in [0,1]: mode = [0,1][i%8]
            if i%8 in [2,3]: mode = [0, 18][i%8-2]  # intentionally mixed
            lines.append(f"CTU Address: {i//16} x={i*4},y={fid*12} paritionSize=3 depth=3 size=8x8 IPM={mode}")
            if i == 7:
                lines.append("CTU Address: 2 x=480,y=88 paritionSize=0 depth=2 size=16x16 IPM=1")
        lines.append(f"Frame {fid} end")
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def test_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        src=tmp/"cover.txt";dst=tmp/"stego.txt";retrieved=tmp/"recovered.bits"
        synthetic_cover(src)
        subprocess.run([sys.executable,wang_ipm.__file__,"embed","--cover",str(src),"--output",str(dst),"--payload-ratio","0.8"],check=True)
        subprocess.run([sys.executable,wang_ipm.__file__,"extract","--input",str(dst),"--metadata",str(dst)+".json","--output-bits",str(retrieved)],check=True)
        assert retrieved.read_bytes()==Path(str(dst)+".bits").read_bytes()
        meta=json.loads(Path(str(dst)+".json").read_text())
        assert meta["frame_stats"]["2"]["embedded_bits"]==0
        assert meta["frame_stats"]["5"]["embedded_bits"]>0
        a=wang_ipm.load_ipm(src);b=wang_ipm.load_ipm(dst)
        for fid in a:
            assert len(a[fid])==len(b[fid])
            if fid not in (1,5):
                assert all(x.ori==y.stego for x,y in zip(a[fid],b[fid]))
        assert any(x.ori!=y.stego for fid in (1,5) for x,y in zip(a[fid],b[fid]))


if __name__=="__main__":
    test_angle_mapping();test_roundtrip();print("ALL TESTS PASSED")
