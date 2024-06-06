import ml_collections
import os
DATASET_DIR = os.environ.get("DATASETS_BASE_DIR", "")


def get_config():
    config = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()
    config.model.arch = "ssast_small"
    config.model.type = "ssast"
    config.model.num_classes = 1000     # needed for dataset parsing. Is not used.
    config.model.model_args = {
        "mask_ratio": 0.8,
        "img_size": (200, 80),
        "patch_size": (4, 16),
        "mask_ratio": 0.5,
        "frequency_first": False
    }
    config.model.patch_embed_args = ml_collections.ConfigDict()


    config.opt = ml_collections.ConfigDict()
    config.opt.optimizer = "Adamw"
    config.opt.learning_rate = 1.5e-4
    config.opt.weight_decay = 0.05
    config.opt.schedule = "warmupcosine"
    config.opt.warmup_epochs = 10
    config.opt.momentum = 0.9

    config.log_every_steps = 100
    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.data = ml_collections.ConfigDict()
    config.data.train_dirs = [
        os.path.join(DATASET_DIR, "audioset_logmelspec_webdataset/balanced_train"),
        os.path.join(DATASET_DIR, "audioset_logmelspec_webdataset/unbalanced_train"),
        os.path.join(DATASET_DIR, "audioset_logmelspec_webdataset/unbalanced_train_2"),
    ]
    config.data.train_samples = [
        18988,
        1766912,
        265408
    ]
    config.data.val_dir = os.path.join(DATASET_DIR, "audioset_logmelspec_webdataset/eval")
    config.data.val_samples = 17408
    config.data.clip_duration = 2.
    config.data.num_frames = 200
    config.data.dataset_name = "audioset"

    config.batch_size = 256     # per gpu
    config.shuffle_buffer_multiplier = 250
    config.half_precision = False
    config.num_epochs = 100

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "ssast-pytorch-mamba"

    return config
