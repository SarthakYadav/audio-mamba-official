import webdataset as wds
import functools
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import pytorch_lightning as pl
from ml_collections import ConfigDict
from . import dataset_helper


class AudioSetData(pl.LightningDataModule):
    def __init__(self, config: ConfigDict, batch_size: int) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
    
    def make_loader(self, train=False):
        urls, samples = dataset_helper.get_data_dirs(self.config.data.train_dirs, self.config.data.train_samples)
        parser_fn = functools.partial(
            dataset_helper.audio_parser, 
            clip_duration=self.config.data.clip_duration,
            crop_type="random" if train else "center"
        )
        record_parser_fn = functools.partial(
            dataset_helper.parse_record,
            audio_parser_fn=parser_fn,
            label_parser_fn=dataset_helper.placeholder_label_transform
        )

        num_dataset_instances = torch.distributed.get_world_size()

    
        # spec = dataset_helper.LogMelSpec(num_frames=self.config.data.num_frames)
        dataset = (
            wds.WebDataset(urls, handler=wds.ignore_and_continue, shardshuffle=True)
            .shuffle(1000).map(record_parser_fn)
            .to_tuple("audio.flac label")
            .batched(self.batch_size, partial=False)
            # .map_tuple(spec, dataset_helper.identity)
            # .with_length(samples//self.batch_size)
        )
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )
        loader.length = samples // self.batch_size
        if train:
            loader = loader.ddp_equalize(samples // self.batch_size)
        return loader

    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.make_loader(train=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.make_loader(train=False)
