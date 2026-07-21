# OmniSTVG
Official PyTorch implementation for **OmniSTVG: Toward Spatio-Temporal Omni-Object Video Grounding**. \
Jiali Yao*, Xin Gu*, Xinran Deng, Mengrui Dai, Bing Fan, Zhipeng Zhang, Yan Huang, Heng Fan†, Libo Zhang† \
(*Equal contribution, †Equal advising and corresponding authors)
## Abstract
We introduce spatio-temporal omni-object video grounding, dubbed **OmniSTVG**, a new STVG task aiming to localize spatially and temporally *all* targets mentioned in the textual query within videos. Compared to classic STVG locating only a single target, OmniSTVG enables localization of not only an arbitrary number of text-referred targets but also their interacting counterparts in the query from the video, making it more flexible and practical in real scenarios for comprehensive understanding. In order to facilitate exploration of OmniSTVG, we propose **BOSTVG**, a large-scale benchmark dedicated to OmniSTVG. Specifically, BOSTVG contains 10,018 videos with 10.2M frames and covers a wide selection of 287 classes from diverse scenarios. Each sequence, paired with a free-form textual query, encompasses a varying number of targets ranging from 1 to 10. To ensure high quality, each video is manually annotated with meticulous inspection and refinement. To our best knowledge, BOSTVG, to date, is the first and the largest benchmark for OmniSTVG. To encourage future research, we present a simple yet effective approach, named **OmniTube**, which, drawing inspiration from Transformer-based STVG methods, is specially designed for OmniSTVG and demonstrates promising results. By releasing BOSTVG, we hope to go beyond classic STVG by locating every object appearing in the query for more comprehensive understanding, opening up a new direction for STVG.

## BOSTVG
The BOSTVG benchmark contains 10,018 videos, 10.2M frames, 287 object classes, and 24,175 annotated target objects. It is publicly available on Hugging Face: https://huggingface.co/datasets/Jelly3000/OmniSTVG
## OmniTube
### Data Preparation
Prepare the following layout under the repository root:

```text
data/
  bostvg/
    annos/
      train.json
      test.json
    videos/
      <video files referenced by the annotations>

model_zoo/
  pretrained_resnet101_checkpoint.pth
  swin_tiny_patch244_window877_kinetics400_1k.pth
  roberta-base/
    config.json
    merges.txt
    pytorch_model.bin
    tokenizer_config.json
    tokenizer.json
    vocab.json
```

Notes:

- `model_zoo/` is intentionally excluded from git. Please download or copy pretrained weights locally before training or testing.
- The default config is [experiments/omnistvg.yaml](experiments/omnistvg.yaml).
- The default dataset root is `data/bostvg`.
- RoBERTa is loaded from `model_zoo/roberta-base/`, not from the network at runtime.

The annotation JSON files are expected to provide the fields used by [datasets/bostvg.py](datasets/bostvg.py), including `width`, `height`, `img_num`, `fps`, `st_frame`, `end_frame`, `st_time`, `ed_time`, `caption`, and `bbox`.

Dataset caches are created automatically on first use:

```text
data/bostvg/data_cache/
  bostvg-train-input.cache
  bostvg-train-anno.cache
  bostvg-test-input.cache
  bostvg-test-anno.cache
```


### Installation
The code requires Python 3.9+ and a CUDA-enabled PyTorch environment.

Install PyTorch and torchvision first according to your CUDA version from the official PyTorch installation guide, then install the remaining dependencies:

```bash
conda create -n omnistvg python=3.11 -y
conda activate omnistvg

# Example only; choose the command that matches your CUDA version.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

The dataloader reads videos through `ffmpeg`, so the system binary must also be available:

```bash
ffmpeg -version
```

### Training

The helper script launches distributed training by default:

```bash
bash run_bostvg.sh
```

Use `GPUS` to change the number of GPUs:

```bash
GPUS=8 bash run_bostvg.sh
```

For single-GPU training:

```bash
GPUS=1 bash run_bostvg.sh
```

Alternatively, run:

```bash
python scripts/train_net.py \
  --config-file experiments/omnistvg.yaml \
  OUTPUT_DIR model_output \
  TENSORBOARD_DIR model_output/tensorboard
```

Checkpoints and logs are written to `OUTPUT_DIR`.

### Evaluation

Evaluate a trained checkpoint with:

```bash
python scripts/test_net.py \
  --config-file experiments/omnistvg.yaml \
  OUTPUT_DIR model_output \
  MODEL.WEIGHT model_output/model_final.pth
```

Or use the helper script:

```bash
MODE=test \
GPUS=1 \
OUTPUT_DIR=model_output \
MODEL_WEIGHT=model_output/model_final.pth \
bash run_bostvg.sh
```

If prediction saving is enabled, results are written to:

```text
<OUTPUT_DIR>/test_results.json
```

### Configuration

YACS config values can be overridden from the command line after the config file:

```bash
python scripts/train_net.py \
  --config-file experiments/omnistvg.yaml \
  DATA_DIR /path/to/bostvg \
  OUTPUT_DIR /path/to/output \
  MODEL.WEIGHT model_zoo/pretrained_resnet101_checkpoint.pth
```

Common options:

- `DATA_DIR`: BOSTVG dataset root.
- `OUTPUT_DIR`: checkpoints, logs, copied config, and evaluation outputs.
- `TENSORBOARD_DIR`: TensorBoard log directory.
- `MODEL.WEIGHT`: initialization checkpoint for training or trained checkpoint for testing.
- `DATALOADER.NUM_WORKERS`: dataloader worker count.
- `SOLVER.MAX_EPOCH`: number of training epochs.

## Citation
If you find our work inspiring or use our codebase in your research, please cite:

```bibtex
@inproceedings{
  yao2026omnistvg,
  title={OmniSTVG: Toward Spatio-Temporal Omni-Object Video Grounding},
  author={Jiali Yao and Xin Gu and Xinran Deng and Mengrui Dai and Bing Fan and Zhipeng Zhang and Yan Huang and Heng Fan and Libo Zhang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=azcQJtcYTE}
}
