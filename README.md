# HEVC Steganalysis Baselines

This repository provides eight HEVC steganalysis baselines with a shared data
partitioning and evaluation protocol. `cao`, `zhao`, and `shen` detect
intra-prediction-mode (IPM) steganography. The other methods analyze partition
structure (`cao-tu`, `dai-pu`), motion vectors (`zhai`), or prediction-error and
residual statistics (`cao264`, `dai`).

Selected Tar steganography algorithm sources are available under `stego/tar1/`,
`stego/tar2/`, `stego/tar4/`, and `stego/tar5/`. Tar3 embedding executables are
provided under `stego/tar3/`. Decoder executables and sample videos are not
included.

## Input format

The baselines consume UTF-8 text exported from an instrumented HEVC
encoder/decoder. Each frame begins with `Frame N begin`, followed by one block
record per line. For example:

```text
Frame 0 begin
CTU Address: 0 x=0,y=0 paritionSize=3 depth=3 size=8x8 IPM=15 TUdepth=1 mv_x=0 mv_y=0 peb=[-1.0,0.0,2.0,1.0]
```

Export the fields required by the method you run. Zhao additionally needs
candidate-mode costs and SATD ordering; the Tar1 embedding algorithm needs
per-block candidate modes and distortion costs. Prediction-error methods need
real decoded residual data. Some methods also use a calibrated export.

List samples in a CSV manifest with this header:

```csv
sample_id,label,split,tar,payload,frame,original,calibrated
```

`label` is `0` for cover and `1` for stego. See
[`examples/manifest_template.csv`](examples/manifest_template.csv) for an
example. Check prepared inputs with:

```bash
python tools/audit_input_domains.py --manifest path/to/manifest.csv --method all
```
