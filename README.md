# Attention-Guided Energy Optimization for Label-Aligned Anomaly Generation

Official implementation of **Attention-Guided Energy Optimization for Label-Aligned Anomaly Generation**.

This repository focuses on **anomaly image-mask pair synthesis** for downstream segmentation tasks. Our method addresses the **mask label drift** problem in diffusion-based generation, where the generated anomalous region does not accurately align with the provided ground-truth mask.

Instead of relying on an additional frozen segmentation model, we directly leverage the diffusion model’s **cross-attention maps** and introduce an **Attention Discrepancy Maximisation (ADM)** energy function during sampling. This improves the alignment between the generated anomalous region and the GT mask, producing higher-quality synthetic data for downstream anomaly segmentation.


---

## Highlights

- **Segmentor-free label alignment**
  - Replaces the auxiliary segmentation model used in prior pipelines.
  - Directly guides generation using the diffusion model’s attention maps.

- **ADM energy-guided sampling**
  - Encourages high attention activation inside the target region.
  - Suppresses attention activation outside the target region.
  - Mitigates mask label drift during generation.

- **ControlNet-based framework**
  - Uses the **GT mask** as the conditional input.
  - Built on top of the **ControlNet** framework

## Overview

Given a ground-truth mask `M`, our framework uses `M` as the conditional input to a diffusion model and generates a synthetic image `x` containing anomalous content inside the mask region.

The key challenge in this task is **mask label drift**, where the generated anomalous region deviates from the provided mask. To solve this, we optimize the denoising process by introducing an **attention-guided energy function** based on the diffusion model’s U-Net attention maps.

During sampling:

1. The GT mask is provided as the control signal.
2. The diffusion model predicts noise step by step.
3. Cross-attention maps are extracted from the U-Net.
4. The proposed **ADM energy** is computed from the GT mask and the attention map.
5. The energy gradient guides the sampling trajectory toward better label alignment.
6. The final generated image stays more consistent with the input mask.

Our method is designed to refine **object location and shape** during sampling, especially in the **contour generation stage**, where early structural errors can propagate to the final output.

---

## Method

### Overall Pipeline

The overall pipeline consists of two stages:

#### 1. Synthetic Sample Generation

Generate anomaly images conditioned on GT masks using a ControlNet-based diffusion model with ADM-guided sampling.

#### 2. Downstream Training

Use the generated image-mask pairs to augment the original dataset and train downstream segmentation models.

### Method Figure

<p align="center">
  <img src="assets/method.png" width="95%" alt="Method figure">
</p>

> **Method overview.** The GT mask is passed into the diffusion model as the condition input. During denoising, cross-attention maps are extracted from the U-Net, and the proposed **Attention Discrepancy Maximisation (ADM)** energy is used to guide sampling. This encourages the model to focus on the target region and mitigates mask label drift without requiring an additional pretrained segmentor.

### ADM Energy

The proposed **Attention Discrepancy Maximisation (ADM)** energy is designed to align attention activation with the GT mask.

It includes two complementary components:

- **Region-level discrepancy term**
  - Encourages high attention values inside the target region.
  - Encourages low attention values outside the target region.

- **Abnormal-attention correction term**
  - Handles strong area imbalance between target and non-target regions.
  - Further suppresses abnormal high/low attention values.

Together, these terms improve the consistency between the generated anomalous area and the GT mask.

---

## Repository Structure

A typical repository layout is as follows:

```text
.
├── annotator/
├── cldm/
├── ldm/
├── models/
│   ├── attnloss.yaml
│   ├── cldm_step.yaml
│   ├── cldm_v15.yaml
│   ├── cldm_v21.yaml
│   ├── myinference.yaml
│   ├── mymodel.yaml
│   └── pretrain.yaml
├── docs/
├── environment.yaml
├── gen.py
├── rotate_mask.py
├── share.py
└── README.md
```

> You may adjust this section if your final repository structure is slightly different.

---

## Environment

This project follows the ControlNet environment setup.

### 1. Clone the repository

```bash
git clone https://github.com/Bbinzz/Attention-Guided-Energy-Optimization.git
cd Attention-Guided-Energy-Optimization
```

### 2. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate control
```

### 3. Install dependencies manually if needed

If `environment.yaml` is not sufficient for your platform, make sure the following dependencies are available:

- Python 3.8+
- PyTorch
- torchvision
- pytorch-lightning
- opencv-python
- numpy
- Pillow
- einops
- omegaconf
- transformers
- safetensors
- gradio

> The environment configuration is intended to follow the original **ControlNet** setup.

---

## Pretrained Weights

Before training or inference, please prepare the required pretrained weights for the Stable Diffusion / ControlNet backbone.

A common layout is:

```text
pretrained_weights/
├── control_sd15_ini.ckpt
├── v1-5-pruned.ckpt
└── other_required_weights...
```

Please update your config files accordingly.

---

## Dataset

Organize your dataset into image-mask pairs.

### Directory Layout

```text
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Notes

- Each mask should indicate the anomaly region.
- The mask is used as the **conditional control signal** during generation.
- The generated image and the input mask together form a synthetic training pair.

### Supported Task Scenario

This repository is designed for **anomaly segmentation data synthesis**, especially for domains where:

- annotated samples are scarce
- anomaly appearance is diverse
- label consistency between generated image and mask is critical

### Example Categories

You can adapt the framework to categories such as:

- polyp segmentation
- medical anomaly regions
- industrial defect regions
- other sparse-data anomaly segmentation tasks

---

## Training

The training pipeline follows the general **ControlNet training paradigm**, while the generation process is further improved by the proposed attention-guided strategy.

### Training Goal

Train a mask-conditioned diffusion model that can generate anomaly images aligned with the provided GT masks.

### Recommended Configs

The repository includes several config files under `models/`, such as:

- `mymodel.yaml`
- `pretrain.yaml`
- `cldm_v15.yaml`
- `cldm_step.yaml`

You can modify dataset paths, checkpoint paths, and training hyperparameters in these files.

### Example Training Command

```bash
python tutorial_train.py \
  --config models/mymodel.yaml \
  --train_data ./data/train \
  --val_data ./data/val \
  --batch_size 4 \
  --learning_rate 1e-5
```

If your training entry script is different, replace the script name accordingly.

### Training Notes

- The baseline model is **ControlNet**.
- The GT mask is used as the conditioning input.
- The purpose of training is to build a strong conditional anomaly generator.
- The proposed method focuses on improving **sampling-time guidance** rather than relying on an extra segmentation model.

---

## Inference

Inference is performed with `gen.py`.

The inference pipeline is intended to follow your local `gen.py` script and applies the proposed **ADM energy guidance** during sampling.

### Inputs

Typical inputs include:

- model config
- model checkpoint
- input mask
- text prompt
- output directory
- image resolution
- number of samples
- DDIM steps
- guidance scale
- random seed

### Example Inference Command

```bash
python gen.py \
  --config models/myinference.yaml \
  --ckpt path/to/your_checkpoint.ckpt \
  --mask path/to/mask.png \
  --prompt "polyp" \
  --save_dir outputs \
  --num_samples 1 \
  --image_resolution 512 \
  --ddim_steps 50 \
  --scale 9.0 \
  --seed 1234
```

> Please adjust the argument names above to match the exact implementation of your local `gen.py`.






## Pretrained Models

You can list released checkpoints here.

| Model | Backbone | Resolution | Dataset | Link | Notes |
|------|----------|-----------:|--------|------|------|
| AGEO-base | ControlNet / SD1.5 | 512 | Polyp / Custom | Coming soon | Main checkpoint |
| AGEO-large | ControlNet / SD1.5 | 512 | Polyp / Custom | Coming soon | Higher-quality version |

> Replace `Coming soon` with actual download links after release.

---

## Quick Start

### Train

```bash
python tutorial_train.py --config models/mymodel.yaml
```

### Generate

```bash
python gen.py \
  --config models/myinference.yaml \
  --ckpt path/to/your_checkpoint.ckpt \
  --mask path/to/mask.png \
  --prompt "polyp" \
  --save_dir outputs
```



## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{attention_guided_energy_optimization_2026,
  title={Attention-Guided Energy Optimization for Label-Aligned Anomaly Generation},
  author={Anonymous},
  journal={CVPR Findings},
  year={2026}
}
```

---

## Contact

If you have any questions, feel free to open an issue or contact the authors.

```text
Email: your_email@example.com
```

---

## Acknowledgements

This project is built upon the excellent **ControlNet** framework and related diffusion-model repositories. We sincerely thank the open-source community for their contributions.

---

## License

Please specify the license for this repository here.

For example:

```text
MIT License
```
