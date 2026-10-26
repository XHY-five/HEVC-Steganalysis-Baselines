# HEVC Steganalysis Baselines

This project provides implementations of eight video steganalysis methods for evaluating their detection performance against HEVC Intra Prediction Mode (IPM) steganography. The implementations utilize HM/x265 decoding information for feature extraction, model training, and performance evaluation, following a unified data partitioning and experimental protocol.

## Codec and steganography implementation policy

No HEVC encoder, decoder, steganographic embedder, codec wrapper, or related
binary is distributed in this repository. This includes method-specific
implementations such as those formerly kept under `stego/`. Users must provide
their own implementations and produce UTF-8 text files in the interfaces
described below. The published code performs steganalysis only; it does not
embed data, encode YUV video, or decode HEVC bitstreams.

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

If an external embedder exchanges an IPM modification plan with an instrumented
encoder, use this plain UTF-8 record format (the former `stego/wang` interface):

```text
Frame 1 begin
CTU Address: 4 x=272,y=24 paritionSize=3 depth=3 size=8x8 oriIPM=26 stegIPM=25
CTU Address: 4 x=276,y=24 paritionSize=3 depth=3 size=8x8 oriIPM=20 stegIPM=21
Frame 1 end
```

`oriIPM` is the original intra-prediction mode and `stegIPM` is the requested
embedded mode at the same spatial block. This modification-plan format is only
a contract for a user-supplied embedder/encoder; the steganalysis programs read
the decoded syntax TXT described below.

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

The eight methods require these fields:

| Method | Required decoded fields |
| --- | --- |
| `cao`, `shen` | `depth`, `partSize`, `IPM`; both original and calibrated TXT |
| `cao-tu` | `depth`, `partSize`, `TUdepth` |
| `zhai` | `partSize`, `mv_x`, `mv_y` |
| `dai-pu` | `depth`, `partSize`, `IPM` |
| `dai` | `IPM`, plus real `peb`/prediction-error data |
| `cao264` | `IPM`, plus real `peb` data; both original and calibrated TXT |
| `zhao` | Candidate-mode format described below; both original and calibrated TXT |

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

- `cao/`, `shen/`, `zhao/`: IPM-based steganalysis methods.
- `cao264/`, `cao-tu/`, `zhai/`: Statistical feature-based steganalysis methods.
- `dai/`, `dai-pu/`: Deep learning-based steganalysis methods.
- `common/`: Shared modules for data processing and evaluation.
- `tools/`: Input validation and experimental utilities (no decoder included).
- `tests/`: Basic functionality tests.
