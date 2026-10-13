# HEVC Steganalysis Baselines

This project provides implementations of eight video steganalysis methods for evaluating their detection performance against HEVC Intra Prediction Mode (IPM) steganography. The implementations utilize HM/x265 decoding information for feature extraction, model training, and performance evaluation, following a unified data partitioning and experimental protocol.

## Project Structure

- `cao/`, `shen/`, `zhao/`: IPM-based steganalysis methods.
- `cao264/`, `cao-tu/`, `zhai/`: Statistical feature-based steganalysis methods.
- `dai/`, `dai-pu/`: Deep learning-based steganalysis methods.
- `common/`: Shared modules for data processing and evaluation.
- `tools/`: Bitstream parsing and experimental utilities.
- `tests/`: Basic functionality tests.