import os
import sys
from hear_api.runtime import RuntimeSSAST
import ml_collections
from importlib import import_module
PT_MAMBA_MODEL_DIR = os.environ.get("PT_MAMBA_MODEL_DIR")


config_path = "configs.ssast_base_200_16x4"
precision = "float16"
RUN_ID = 1
model_path = os.path.join(PT_MAMBA_MODEL_DIR, f"ssast_base_200_16x4_4x256_fp16_r{RUN_ID}")


def load_model(model_path=model_path, config=import_module(config_path).get_config()):
    model = RuntimeSSAST(config=config, weights_dir=model_path, precision=precision)
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
