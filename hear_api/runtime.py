import os
import glob
import torch
# import tensorflow as tf
import torch.nn as nn
from .feature_helper import get_timestamps
import sys

sys.path.append("..")
from src import utilities
from src.data.features import LogMelSpec
torch.set_float32_matmul_precision("high")
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class RuntimeSSAST(nn.Module):
    def __init__(self, config, weights_dir, precision="float32") -> None:
        super().__init__()
        self.config = config
        self.local_batch_size = 1

        model = utilities.get_model(self.config)
        ckpts = glob.glob(os.path.join(weights_dir, "checkpoints", "*.pth"))
        ckpts = sorted(ckpts, key=lambda x:int(x.replace(".pth","").split("-")[-1]))
        print("loading checkpoint -> {}".format(ckpts[-1]))
        checkpoint = torch.load(ckpts[-1], map_location="cpu")
        model.load_state_dict(checkpoint['model'])

        self.model = model
        self.model.eval()

        precision = config.get("precision", precision)
        print(f"!! Using {precision} precision !!")

        self.autocast_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        self.autocast_enabled = True if "16" in precision else False
        print("autocast_enabled:", self.autocast_enabled)
        print("autocast_dtype:", self.autocast_dtype)
        # frequency_first = config.model.model_args.get("frequency_first", False)

        self.input_size = self.model.img_size
        self.grid_size = self.model.grid_size()
        self.patch_size = self.model.patch_size()
        self.sample_rate = 16000
        self.embed_dim = self.model.embed_dim
        self.frequency_first = self.model.frequency_first
        self.log_mel_spec = LogMelSpec(flip_ft=not self.frequency_first)
        self._cpu_mel_spec = "AMD" in torch.cuda.get_device_name()
        if self._cpu_mel_spec:
           print("Running on an AMD device, extracting features on the CPU")

    def to_feature(self, batch_audio):
        if self._cpu_mel_spec:
            self.log_mel_spec = self.log_mel_spec.cpu()
            batch_audio = batch_audio.cpu()
        x = self.log_mel_spec(batch_audio)
        if self._cpu_mel_spec:
            x = x.cuda()
        return x

    def encode(self, lms):
        x = lms
        # print("lms shape:", lms.shape)
        if self.frequency_first:
            patch_fbins = self.grid_size[0]
            unit_frames = self.input_size[1]
            cur_frames = x.shape[-1]
        else:
            patch_fbins = self.grid_size[1]
            unit_frames = self.input_size[0]
            cur_frames = x.shape[-2]
        # print(f"patch_fbins: {patch_fbins}, unit_frames: {unit_frames}, cur_frames: {cur_frames}")
        embed_d = self.embed_dim

        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            if self.frequency_first:
                x = torch.nn.functional.pad(x, (0, pad_frames))
            else:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_frames))
        # print("!!!!!!!!!!!!!!!!!! padded x.shape:", x.shape)
        if self.frequency_first:
            r =  x.shape[-1] // unit_frames
        else:
            r =  x.shape[-2] // unit_frames
        embeddings = []
        with torch.no_grad():
            for i in range(r):
                if self.frequency_first:
                    sub_x = x[..., i*unit_frames:(i+1)*unit_frames]
                else:
                    sub_x = x[:, :, i*unit_frames:(i+1)*unit_frames, :]
                # print("\t sub_x.shape", sub_x.shape)
                with torch.autocast(device_type="cuda", 
                                    dtype=self.autocast_dtype,
                                    enabled=self.autocast_enabled):
                    emb = self.model.forward_features(sub_x)
                # print("emb shape:", emb.shape)
                embeddings.append(emb)

        x = torch.hstack(embeddings)
        # print("stacked embeddings shape:", x.shape)
        pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
        if pad_emb_frames > 0:
            x = x[:, :-pad_emb_frames] # remove padded tail
        return x

    def audio2feats(self, audio):
        self.model.eval()
        x = self.to_feature(audio)
        x = self.encode(x)
        return x
    
    def get_scene_embeddings(self, audio):
        x = self.audio2feats(audio)        
        x = torch.mean(x, dim=1)
        return x

    def get_timestamp_embeddings(self, audio):
        x = self.audio2feats(audio)
        ts = get_timestamps(self.sample_rate, audio, x)
        return x, ts
