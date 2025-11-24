# SC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17625710.svg)](https://doi.org/10.5281/zenodo.17625710) ![GitHub License](https://img.shields.io/github/license/pinto0309/SC) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/sc)

Ultrafast sitting classification. 32x24 pixels is sufficient for estimating the state of the whole human body.

https://github.com/user-attachments/assets/635773d8-3826-45fd-ac33-e51fe3695176

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|115 KB|0.8923|0.13 ms|[Download](https://github.com/PINTO0309/SC/releases/download/onnx/sc_p_32x24.onnx)|
|N|176 KB|0.9076|0.24 ms|[Download](https://github.com/PINTO0309/SC/releases/download/onnx/sc_n_32x24.onnx)|
|T|279 KB|0.8935|0.31 ms|[Download](https://github.com/PINTO0309/SC/releases/download/onnx/sc_t_32x24.onnx)|
|S|494 KB|0.9168|0.39 ms|[Download](https://github.com/PINTO0309/SC/releases/download/onnx/sc_s_32x24.onnx)|
|C|875 KB|0.9265|0.47 ms|[Download](https://github.com/PINTO0309/SC/releases/download/onnx/sc_c_32x24.onnx)|

## Data sample

|1|2|3|4|
|:-:|:-:|:-:|:-:|
|<img width="24" height="32" alt="image" src="https://github.com/user-attachments/assets/1e0ba9c7-9020-43f2-a3e8-adcad1cbc283" />|<img width="24" height="32" alt="image" src="https://github.com/user-attachments/assets/c5abc97e-ae56-4e30-8be2-7f4883e41083" />|<img width="24" height="32" alt="image" src="https://github.com/user-attachments/assets/ee23ffe4-bfe9-4e43-984a-b7fa563af29c" />|<img width="24" height="32" alt="image" src="https://github.com/user-attachments/assets/fb2c5736-e22f-4933-a0e1-5b04084bede5" />|

## Setup

```bash
git clone https://github.com/PINTO0309/SC.git && cd SC
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Inference

```bash
uv run python demo_sc.py \
-pm sc_c_32x24.onnx \
-v 0 \
-ep cuda \
-dlr -dnm -dgm -dhm -dhd

uv run python demo_sc.py \
-pm sc_c_32x24.onnx \
-v 0 \
-ep tensorrt \
-dlr -dnm -dgm -dhm -dhd
```

## Dataset Preparation

- AVA Actions Download (v2.2) - CC BY 4.0 license
  https://research.google.com/ava/download.html

```bash
uv run python 01_download_ava_videos.py
```
```bash
uv run python 03_renumber_files.py
uv run python 04_data_prep.py

uv run python 04_data_prep.py \
--timestamp-stride 32 \
--image-dir ./data/images \
--annotation-file ./data/annotation.txt \
--histogram-file ./data/image_size_hist.png \
--class-ratio-file ./data/class_ratio.png \
--balance-classes \
--dry-run

uv run python 04_data_prep.py \
--timestamp-stride 1 \
--image-dir ./data/images \
--annotation-file ./data/annotation.txt \
--histogram-file ./data/image_size_hist.png \
--class-ratio-file ./data/class_ratio.png \
--balance-classes

python 04_data_prep.py \
--timestamp-stride 1 \
--image-dir ./data/images \
--annotation-file ./data/annotation.txt \
--histogram-file ./data/image_size_hist.png \
--class-ratio-file ./data/class_ratio.png \
--balance-classes \
--resume-existing \
--start-index 198727
```

<img width="400" alt="class_ratio_merged" src="https://github.com/user-attachments/assets/df607706-088e-4c23-9be1-64af8558a8cf" />
<img width="400" alt="image_size_hist_merged" src="https://github.com/user-attachments/assets/1db80a4e-7b19-4636-a72d-012f7827c3ac" />

```bash
uv run python 05_make_parquet.py \
--embed-images \
--overwrite
```
```bash
uv run python 06_merge_parquet.py dataset1.parquet dataset2.parquet \
--output dataset.parquet \
--overwrite
```
```bash
uv run python 06_merge_parquet.py dataset1.parquet dataset2.parquet \
--output dataset.parquet \
--overwrite
```

## Training Pipeline

- Use the images located under `dataset/output/002_xxxx_front_yyyyyy` together with their annotations in `dataset/output/002_xxxx_front.csv`.
- Every augmented image that originates from the same `still_image` stays in the same split to prevent leakage.
- The training loop relies on `BCEWithLogitsLoss` plus class-balanced `pos_weight` to stabilise optimisation under class imbalance; inference produces sigmoid probabilities. Use `--train_resampling weighted` to switch on the previous `WeightedRandomSampler` behaviour, or `--train_resampling balanced` to physically duplicate minority classes before shuffling.
- Training history, validation metrics, optional test predictions, checkpoints, configuration JSON, and ONNX exports are produced automatically.
- Per-epoch checkpoints named like `sc_epoch_0001.pt` are retained (latest 10), as well as the best checkpoints named `sc_best_epoch0004_f1_0.9321.pt` (also latest 10).
- The backbone can be switched with `--arch_variant`. Supported combinations with `--head_variant` are:

  | `--arch_variant` | Default (`--head_variant auto`) | Explicitly selectable heads | Remarks |
  |------------------|-----------------------------|---------------------------|------|
  | `baseline`       | `avg`                       | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, you need to adjust the height and width of the feature map so that they are divisible by `--token_mixer_grid` (if left as is, an exception will occur during ONNX conversion or inference). |
  | `inverted_se`    | `avgmax_mlp`                | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, it is necessary to adjust `--token_mixer_grid` as above. |
  | `convnext`       | `transformer`               | `avg`, `avgmax_mlp`, `transformer`, `mlp_mixer` | For both heads, the grid must be divisible by the feature map (default `3x2` fits with 30x48 input). |
- The classification head is selected with `--head_variant` (`avg`, `avgmax_mlp`, `transformer`, `mlp_mixer`, or `auto` which derives a sensible default from the backbone).
- Pass `--rgb_to_yuv_to_y` to convert RGB crops to YUV, keep only the Y (luma) channel inside the network, and train a single-channel stem without modifying the dataloader.
- Alternatively, use `--rgb_to_lab` or `--rgb_to_luv` to convert inputs to CIE Lab/Luv (3-channel) before the stem; these options are mutually exclusive with each other and with `--rgb_to_yuv_to_y`.
- Mixed precision can be enabled with `--use_amp` when CUDA is available.
- Resume training with `--resume path/to/sc_epoch_XXXX.pt`; all optimiser/scheduler/AMP states and history are restored.
- Loss/accuracy/F1 metrics are logged to TensorBoard under `output_dir`, and `tqdm` progress bars expose per-epoch progress for train/val/test loops.

Baseline depthwise-separable CNN:

```bash
SIZE=32x24
uv run python -m sc train \
--data_root data/dataset.parquet \
--output_dir runs/sc_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_ratio 0.9 \
--val_ratio 0.1 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant baseline \
--seed 42 \
--device auto \
--use_amp
```

Inverted residual + SE variant (recommended for higher capacity):

```bash
SIZE=32x24
uv run python -m sc train \
--data_root data/dataset.parquet \
--output_dir runs/sc_is_s_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_ratio 0.9 \
--val_ratio 0.1 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--seed 42 \
--device auto \
--use_amp
```

ConvNeXt-style backbone with transformer head over pooled tokens:

```bash
SIZE=32x24
uv run python -m sc train \
--data_root data/dataset.parquet \
--output_dir runs/sc_convnext_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_ratio 0.9 \
--val_ratio 0.1 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant convnext \
--head_variant transformer \
--token_mixer_grid 2x2 \
--seed 42 \
--device auto \
--use_amp
```

- Outputs include the latest 10 `sc_epoch_*.pt`, the latest 10 `sc_best_epochXXXX_f1_YYYY.pt` (highest validation F1, or training F1 when no validation split), `history.json`, `summary.json`, optional `test_predictions.csv`, and `train.log`.
- After every epoch a confusion matrix and ROC curve are saved under `runs/sc/diagnostics/<split>/confusion_<split>_epochXXXX.png` and `roc_<split>_epochXXXX.png`.
- `--image_size` accepts either a single integer for square crops (e.g. `--image_size 48`) or `HEIGHTxWIDTH` to resize non-square frames (e.g. `--image_size 64x48`).
- Add `--resume <checkpoint>` to continue from an earlier epoch. Remember that `--epochs` indicates the desired total epoch count (e.g. resuming `--epochs 40` after training to epoch 30 will run 10 additional epochs).
- Launch TensorBoard with:
  ```bash
  tensorboard --logdir runs/sc
  ```

### ONNX Export

```bash
uv run python -m sc exportonnx \
--checkpoint runs/sc_is_s_32x24/sc_best_epoch0049_f1_0.9939.pt \
--output sc_s.onnx \
--opset 17
```

- The saved graph exposes `images` as input and `prob_pointing` as output (batch dimension is dynamic); probabilities can be consumed directly.
- After exporting, the tool runs `onnxsim` for simplification and rewrites any remaining BatchNormalization nodes into affine `Mul`/`Add` primitives. If simplification fails, a warning is emitted and the unsimplified model is preserved.

## Arch

<img width="300" alt="sc_p_32x24" src="https://github.com/user-attachments/assets/1b0d74b7-ceca-49ae-832d-9ffff80f6945" />

## Ultra-lightweight classification model series
1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License
6. [HSC: Happy smile classifier](https://github.com/PINTO0309/HSC) - MIT License
7. [WHC: Waving Hand Classification](https://github.com/PINTO0309/WHC) - MIT License

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025sc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/SC},
  month     = {11},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17625710},
  url       = {https://github.com/PINTO0309/sc},
  abstract  = {Ultrafast sitting classification.},
}
```

## Acknowledgements
- AVA Actions Download (v2.2) - CC BY 4.0 License
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34: Apache 2.0 License
  ```bibtex
  @software{DEIMv2-Wholebody34,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: body, adult, child, male, female, body_with_wheelchair, body_with_crutches, head, front, right-front, right-side, right-back, back, left-back, left-side, left-front, face, eye, nose, mouth, ear, collarbone, shoulder, solar_plexus, elbow, wrist, hand, hand_left, hand_right, abdomen, hip_joint, knee, ankle, foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34},
    year={2025},
    month={10},
    doi={10.5281/zenodo.17625710}
  }
  ```
- https://github.com/PINTO0309/bbalg: MIT License
