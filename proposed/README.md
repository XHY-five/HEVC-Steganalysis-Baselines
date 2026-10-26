# Proposed Method

This directory contains our dual-branch IPM/coefficient steganalysis code:
feature-map generation, dataset loading, model definition, and training scripts
for the three payload settings (`01`, `03`, `05`).

Install the packages in `requirements.txt`. Prepare decoded IPM and transform-
coefficient TXT files, then set the input/output paths in the corresponding
generation and training scripts. Model weights, datasets, the HM decoder, and
third-party EfficientNet source are not included. Place the EfficientNet-B1 and
B3 pretrained weights under `efficientnet_path/` if you want pretrained
initialization; otherwise the model uses random initialization.
