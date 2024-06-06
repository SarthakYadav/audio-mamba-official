# audio-mamba-official
This is the official repository for our paper ["Audio Mamba: Selective State Spaces for Self-Supervised Audio Representations"](https://arxiv.org/abs/2406.02178), set to appear in Proc. INTERSPEECH 2024. 

# Contents
* [Pre-trained weights for the default Self-Supervised Audio Mamba (SSAM) configurations](https://drive.google.com/drive/folders/1l6_8e1TFEfnobwUMIbA7-aKw2FwlBhho?usp=drive_link)
* Our local copy of [hear-eval-kit](external_sources/hear-eval-kit) for easy downstream reproducibility. Original can be found [here](https://github.com/hearbenchmark/hear-eval-kit)
* [Feature extraction API](hear_api) compatible with the [hear-eval-kit](https://github.com/hearbenchmark/hear-eval-kit) format for extracting features.
* Code used to train the SSAM models.
* Helper code to [extract features](extract_features.sh) and [run downstream experiments](downstream_experiments.sh) on provided pre-trained models

---

# Setup

## Environment
* Required: `cuda 11.x` or newer, `cudnn 8.2` or newer.
* Create a new conda environment with `python 3.10` or later.
* Requires `torch 2.1.2` or newer.

Follow these steps
```shell
conda create -n mamba-env python=3.10 -y
conda activate mamba-env

pip install -r requirements.txt

# install hear-eval-kit specific requirements
pip install -r hear-eval-kit/requirements.txt

# install hear-eval-kit, WITHOUT AUTO DEPS
cd external_sources/hear-eval-kit && pip install --no-deps . && cd -

# install causal-conv1d (optional)
pip install causal-conv1d>=1.2.0

# install mamba-ssm
pip install mamba-ssm==1.1.3
```

## Get 16000 Hz data from hear
* Follow https://hearbenchmark.com/hear-tasks.html to get data. By default, data on HEAR's zenodo page is 48000 Hz.
* We recommend downloading data directly from HEAR's [GCS bucket](gs://hear2021-archive/tasks/), where you can find preprocessed 16000 Hz data.
* Extract all the files to a folder `$TASKS_DIR`

## Get pretrained weights

* Pre-trained weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1l6_8e1TFEfnobwUMIbA7-aKw2FwlBhho?usp=drive_link)
* Download the entire folder and export that folder as `$PT_MAMBA_MODEL_DIR`

## Extract features

```shell
export PT_MAMBA_MODEL_DIR=/path/to/pretrained_weights
./extract_features.sh $TASKS_DIR $OUTPUT_DIR
```
where TASKS_DIR is the directory where you extracted tasks from HEAR-2021 to, and OUTPUT_DIR is the base directory where output features will be stored. The given script will extract features from SSAST and SSAM Tiny configurations, you can change it as you need.
This also prepares a `todo_audioset` directory in OUTPUT_DIR, which is setting up for downstream classification on 10 seeds.

## Run downstream experiments

After extracting features, to run downstream experiment on a specific config, use the following command:
```shell
./downstream_experiments.sh ssam_tiny_200_16x4 $OUTPUT_DIR
```

This will run downstream experiments on all the extracted features for the tiny SSAM configuration on 10 random seeds.

## Get results
Finally, you can run the following script to get results of downstream experiments of the two models

```shell
python stats_aggregation_v2.py --base_dir ${OUTPUT_DIR}/todo_audioset --output_dir ${OUTPUT_DIR}/parsed_results
```

---

# Extracting features on your own audio file
The [hear_api](hear_api) can be used to extract features from your own audio files.

```python
import torchaudio

from hear_api import RuntimeSSAST
from importlib import import_module
config = import_module("configs.ssam_tiny_200_16x4").get_config()
ssam = RuntimeSSAST(config, "path/to/pretrained/weights").cuda()

# alternatively just use the following if you have the paths setup right
# ssam = import_module("configs.ssam_tiny_200_16x4").load_model().cuda()

x, sr = torchaudio.load("path/to/audio.wav")
x = x.cuda()
o = ssam.get_scene_embeddings(x)

```

---

# Pretraining
Pretraining code is included in the release. Any model configuration (for instance, `ssam_tiny_200_16x4`) was trained with the following command:
```shell
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --config configs.poc.ssam_tiny_200_16x4 --workdir $EXP_DIR/ssam_tiny_200_16x4_4x256_fp16_r1 --precision float16 --print_freq 50 --num_workers 16 --no_wandb
```
We use a `torchdata` based datapipe for data loading, operating on precomputed log melspectrogram features stored in webdataset archive(s). You can adapt the data loading for your own use case.

---