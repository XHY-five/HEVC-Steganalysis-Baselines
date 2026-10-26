# HEVC Steganalysis Baselines

This project provides implementations of eight HEVC video steganalysis
baselines spanning several embedding and feature domains. Only `cao`, `zhao`,
and `shen` are IPM-domain detectors. The remaining methods analyze other codec
domains: CU/PU/TU partition structure (`cao-tu`), motion vectors (`zhai`),
prediction-error/residual statistics (`cao264` and `dai`), or PU partition maps
(`dai-pu`). All methods follow a unified data partitioning and evaluation
protocol. Instrumented HM/x265 syntax exports provide a common interchange
format, but sharing that input container does not make every method an IPM
steganalysis method.

## Published steganography sources

The repository also includes selected source code for generating stego inputs.
These files are separate from the eight steganalysis baselines:

| Source | Included code | External inputs |
| --- | --- | --- |
| `stego/tar1/generate_embed_plan.py` | Candidate filtering and matrix coding for an IPM modification plan | Cover IPM TXT and per-block candidate TXT |
| `stego/wang/Wang_IPM_Standalone/algorithm/wang_ipm.py` | Wang angle-difference IPM embedding and extraction | Cover IPM TXT; optional candidate RD-cost JSON |
| `stego/liu/HM_diamond/source/` | HM source tree containing the diamond-coding modification | An HM-compatible build environment and input video |

The HM source retains its original upstream C++ layout and license in
`stego/liu/HM_diamond/COPYING`. Build directories, prebuilt EXEs, videos,
generated TXT files, and experiment launchers are excluded. The locally
available `tar3` scripts orchestrate external STC/x265 binaries, while the
`yang` script invokes an encoder whose embedding implementation is available
here only as a binary; neither supplies a standalone core algorithm source.
They are therefore excluded from this source release.

To use the Python algorithms, install their dependencies and supply the input
files described below. Tar1 uses `pandas`; Wang uses the Python standard
library. No encoder or decoder executable is bundled.

## External stego/encoder output contract

An external data-generation pipeline must produce, for every sample:

1. A cover or stego HEVC bitstream (`.265`, `.hevc`, or `.bin`).
2. A decoded `original` syntax TXT matching the format in the next section.
3. When required by the selected method, a calibrated/recompressed bitstream
   and its decoded `calibrated` syntax TXT.
4. One row in the experiment manifest identifying the sample, class label,
   split, payload, frame, and TXT paths.

The filenames and directory layout are not significant; the manifest is the
authoritative mapping. Cover and stego samples must be generated with
comparable codec settings (resolution, frame count, frame rate, QP, GOP and
chroma/bit-depth settings) so the classifier does not learn encoder-setting
differences instead of steganographic traces.

The Tar1 and Wang algorithms write an IPM modification plan for a user-supplied
instrumented encoder. Its plain UTF-8 record format is:

```text
Frame 1 begin
CTU Address: 4 x=272,y=24 paritionSize=3 depth=3 size=8x8 oriIPM=26 stegIPM=25
CTU Address: 4 x=276,y=24 paritionSize=3 depth=3 size=8x8 oriIPM=20 stegIPM=21
Frame 1 end
```

`oriIPM` is the original intra-prediction mode and `stegIPM` is the requested
embedded mode at the same spatial block. The steganalysis programs read the
decoded syntax TXT described below, not this pre-encoding plan.

### Tar1 candidate input

`generate_embed_plan.py` reads the cover IPM TXT shown below and a separate
candidate TXT with one `name=value` record per candidate PU. Required fields
are `Frame`, `CTU`, `x`, `y`, `depth`, `paritionSize`, `coverIPM`, `coverBit`,
`D0`, `D1`, `IPM_0`, and `IPM_1`. For example:

```text
Frame=1 CTU=0 x=0 y=0 depth=3 paritionSize=3 coverIPM=15 coverBit=1 D0=4.2 D1=4.8 IPM_0=14 IPM_1=15
```

Four candidate records must form each eligible 8×8 region; the algorithm
groups records by frame, CTU, CU coordinates, depth, and partition size. It
selects a payload fraction by minimum `D0`/`D1`, then writes the IPM plan
above. Set `cover_root`, `cand_root`, `stego_root`, and `payload` in the script's
`__main__` block for a local run. `Cencoder.py` and `StegoDecoder.py` are
local codec launchers and are not required to read or generate the plan.

### Wang cover and optional RD-cost input

Wang reads the decoded cover IPM TXT shown below, including `Frame N begin/end`
and the `CTU Address`, `x`, `y`, `paritionSize`, `depth`, `size`, and `IPM`
fields. For eligible 4×4 PUs, the current exporter represents the containing
CU as `paritionSize=3 depth=3 size=8x8`. To embed:

```bash
python stego/wang/Wang_IPM_Standalone/algorithm/wang_ipm.py embed --cover cover_ipm.txt --output stego_ipm.txt --payload-ratio 0.3
```

The script also writes `.json` metadata and `.bits` next to the output TXT.
Optional `--ddg-dir` files are named `frame_N_ddg.json` and contain nodes with
`ctu_addr`, `x`, `y`, `w=4`, `h=4`, and a `candidates` object mapping IPM indices
to RD costs. Without complete RD costs, the script reports when it uses a
nearest-IPM fallback; this does not reproduce the paper's RD-optimal choice.

Each sample normally has two files:

- `original`: syntax exported while decoding the input bitstream.
- `calibrated`: syntax exported after the method's calibration/recompression
  procedure. Methods that do not use calibration may only consume `original`.

## Decoded syntax TXT format

The file is plain UTF-8 text. Start each decoded frame with:

```text
Frame 0 begin
```

Then emit one block record per line. Fields are written as whitespace-separated
`name=value` pairs. Lists use comma-separated values inside square brackets.
The parser also accepts `paritionSize` (the spelling used by some HM
instrumentation) as an alias for `partSize`.

```text
CTU Address: 0 x=0,y=0 paritionSize=3 depth=3 size=8x8 IPM=15 TUdepth=1 mv_x=0 mv_y=0 peb=[-1.0,0.0,2.0,1.0]
CTU Address: 0 x=8,y=0 paritionSize=0 depth=2 size=16x16 IPM=5 TUdepth=0 mv_x=1 mv_y=-1 peb=[0.0,1.0,-1.0,0.0]
```

Required field meanings:

| Field | Meaning |
| --- | --- |
| `Frame N begin` | Selects frame index `N` for following records. |
| `CTU Address` / `CTU` | CTU index containing the block. |
| `x`, `y` | Top-left block coordinates in luma pixels. |
| `size` | Block dimensions, for example `8x8`. |
| `depth` | CU quadtree depth. |
| `partSize` / `paritionSize` | HM partition-size enum; this code expects `3` for `NxN`. |
| `IPM` / `coverIPM` | Integer intra-prediction mode. |
| `TUdepth` / `tu_depth` | Transform-unit depth. |
| `mv_x`, `mv_y` | Motion-vector components required by the Zhai method. |
| `peb` | Prediction-error/residual samples as a numeric list. |

Records must cover the complete decoded frame without missing spatial regions
for methods that construct PU or prediction-error maps. `original` and
`calibrated` files must use matching frame indices and coordinates wherever a
method compares their records.

The eight methods operate in the following detection domains and require these
exported fields:

| Method | Detection domain | Required exported fields |
| --- | --- | --- |
| `cao`, `shen` | IPM domain | `depth`, `partSize`, `IPM`; both original and calibrated TXT |
| `zhao` | IPM/SATD candidate domain | Candidate-mode format described below; both original and calibrated TXT |
| `cao-tu` | CU/PU/TU partition structure | `depth`, `partSize`, `TUdepth` |
| `zhai` | Motion-vector domain | `depth`, `partSize`, `mv_x`, `mv_y` |
| `dai-pu` | PU partition-map domain | `depth`, `partSize`, `IPM` (IPM is exported context, not the detector's target domain) |
| `dai` | Prediction-error domain | Real `peb`/prediction-error data, with `IPM` as exported context |
| `cao264` | PEB/residual-statistics domain | Real `peb` data and auxiliary `IPM`; both original and calibrated TXT |

Do not synthesize prediction-error values from IPM differences. `dai` and
`cao264` require actual decoded residual/prediction-error data.

### Zhao candidate-mode TXT

Zhao additionally requires all 35 intra-mode costs and their ascending SATD
order for each block. Emit one record per line:

```text
Frame=0 CTU=0 x=0 y=0 depth=3 partSize=3 size=8x8 coverIPM=15 coverBit=1 MPM=[0,1,2] modeCosts=[14.2,13.9,15.1] SATD_sorted=[1,0,2]
```

In real input, `modeCosts` and `SATD_sorted` must each contain all candidate
modes used by the implementation (35 modes for the current HEVC setup).
`modeCosts[i]` is the cost of mode `i`; `SATD_sorted` lists mode indices from
lowest to highest cost.

## Manifest format

Point the analysis code to exported TXT files through a CSV manifest:

```csv
sample_id,label,split,tar,payload,frame,original,calibrated
clip01_cover,0,train,Tar3,0.3,0,logs/cover/clip01.txt,logs/cover_cal/clip01.txt
clip01_stego,1,train,Tar3,0.3,0,logs/stego/clip01.txt,logs/stego_cal/clip01.txt
```

`label` is `0` for cover and `1` for stego. `split` is normally `train` or
`test`. `frame` selects one frame from a multi-frame TXT file. Paths may be
absolute or relative to the process working directory. A complete template is
available at `examples/manifest_template.csv`.

Validate a prepared dataset before running experiments:

```bash
python tools/audit_input_domains.py --manifest path/to/manifest.csv --method all
```

## Project Structure

- `cao/`, `shen/`, `zhao/`: IPM-domain steganalysis methods.
- `cao-tu/`: CU/PU/TU partition-structure steganalysis.
- `zhai/`: Motion-vector-domain steganalysis.
- `cao264/`: Prediction-error-block/residual-statistics steganalysis.
- `dai/`: Prediction-error-domain deep steganalysis.
- `dai-pu/`: PU partition-map deep steganalysis.
- `stego/tar1/`, `stego/wang/`, `stego/liu/`: Selected embedding algorithm sources and their input contracts above.
- `common/`: Shared modules for data processing and evaluation.
- `tools/`: Input validation and experimental utilities (no decoder included).
- `tests/`: Basic functionality tests.
