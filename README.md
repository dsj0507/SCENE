# SCENE

SCENE is a research project for open-vocabulary semantic segmentation in
remote-sensing imagery.

> **Release status**
>
> This is a preliminary repository. The complete core implementation, detailed
> method description, dataset preparation instructions,
> and reproducibility documentation will be released after
> the paper is published.

## Preparation

Run all commands from the repository root.

The environment should contain PyTorch, Detectron2, OpenCLIP, and the remaining
dependencies imported by the project. If the environment is already activated,
you can make the scripts use its Python executable:

```bash
export PYTHON_CMD=python
```

Place the required pretrained resources under `pretrained/`, or update their
paths in the configuration:

```text
pretrained/
├── RemoteCLIP-ViT-B-32.pt
├── sam3.pt
└── bpe_simple_vocab_16e6.txt.gz
```

Dataset locations are registered under `SCENE/data/datasets/`. Adjust the
corresponding registration file when your dataset root differs from the local
layout expected by the repository.

## Training

The training wrapper accepts a configuration file, requested GPU count, output
directory, and optional Detectron2 configuration overrides:

```bash
bash run.sh CONFIG_FILE NUM_GPUS OUTPUT_DIR [OPTS]
```

Train the ViT-B model on LandDiscover 50K with two GPUs:

```bash
bash run.sh \
  configs/vitb_384_LD50K_v2.yaml \
  2 \
  output_vitb_384_LD50K_v2
```

Train the ViT-L model:

```bash
bash run.sh \
  configs/vitl_336_LD50K_v2.yaml \
  2 \
  output_vitl_336_LD50K_v2
```

## Evaluation

Evaluate a trained checkpoint with:

```bash
bash eval.sh CONFIG_FILE NUM_GPUS RUN_OUTPUT_DIR [OPTS]
```

For example:

```bash
bash eval.sh \
  configs/vitb_384_LD50K_v2.yaml \
  1 \
  output_vitb_384_LD50K_v2/latest
```


Please check the repository releases and commit history for future updates.
