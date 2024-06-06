import os
import json
import torch
import functools
from . import models
from absl import logging
from ml_collections import ConfigDict


def get_model(config:ConfigDict, device=None, precision=None):
    model_cls = getattr(models, config.model.arch)
    model_args = config.model.get("model_args", {})
    return model_cls(**model_args.to_dict())


def write_config_to_json(workdir, config):
    config_path = os.path.join(workdir, "config.json")
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    if os.path.exists(config_path):
        logging.info(f"config file {config_path} exists.. Not overwriting.")
        return
    with open(config_path, "w") as fd:
        json.dump(config.to_dict(), fd)


class Prefetcher():
    def __init__(self, loader,
                 device,
                 input_key="audio",
                 target_key="label"):
        self.loader = iter(loader)
        self.device = device
        self.inp_key = input_key
        self.tgt_key = target_key
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            inp, label = next(self.loader)
            self.next_input = inp
            self.next_target = label
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            # self.next_input = self.next_input.cuda(non_blocking=True)
            # self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


class PrefetcherTorch():
    def __init__(self, loader,
                 device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            rec = next(self.loader)
            # print("in preload, rec[0].shape", rec[0].shape)
            self.next_input = rec[0]
            self.next_target = rec[1]
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            # self.next_input = self.next_input.cuda(non_blocking=True)
            # self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target
