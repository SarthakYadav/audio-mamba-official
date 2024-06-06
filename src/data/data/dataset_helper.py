import functools
import os
import glob
import ml_collections
import webdataset as wds
import torch
import torchaudio
from .parsing_utilities import audio_parser, parse_record
from .features import LogMelSpec


def placeholder_label_transform(record):
    return torch.Tensor([0.])


def identity(x):
    return x


def get_data_dirs(dirs, samples):
    # config.data.train_dirs
    if isinstance(dirs, str):
        dirs = [dirs]
    files = []
    for dir in dirs:
        files.extend(glob.glob(os.path.join(dir, "*.tar")))

    if isinstance(samples, int):
        num_samples = samples
    else:
        num_samples = sum(samples)
    return files, num_samples


# def prepare_dataset(train_dirs, train_samples):


def split_urls_on_nodes(urls):
    """Split urls_ correctly per accelerator node
    :param urls:
        num_replicas: total number of nodes
            usually corresponds to the world size
        rank: rank of current dataset instance
    :return: slice of urls_
    """

    num_replicas = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    urls_this = urls[rank::num_replicas]
    
    return urls_this


def split_urls_on_workers(urls):
    """Split urls_ correctly per accelerator node
    :param urls:
    :return: slice of urls_
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers

        return urls[wid::num_workers]
    else:
        return urls



def decode_td(item):
        key, value = item
        if key.endswith(".flac") or key.endswith(".label") or key.endswith(".duration"):
            return key, value.read()
        else:
            return key, value


def decode_np_tar(item):
    key, value = item
    if key.endswith(".pyd"):
        return key, value.read()
    else:
        return key, value
    

def fix_keys_for_tp(record):
    key = record['__key__']
    actual_key = key.split("/")[-1]
    actual_url = "/".join(key.split("/")[:-1])
    result = {
        "__key__": actual_key,
        "__url__": actual_url
    }
    for k, v in record.items():
        if k.startswith("."):
            result[k[1:]] = v
    return result


def return_data(data):
    return data['audio.flac'], data['label']


# def prepare_dataset(config: ml_collections.ConfigDict):
#     tr_files, tr_samples = get_data_dirs(config.data.train_dirs, config.data.train_samples)
#     val_files, val_samples = get_data_dirs(config.data.val_dir, config.data.val_samples)

#     tr_parser_fn = functools.partial(audio_parser, clip_duration=config.data.clip_duration, crop_type=config.data.get("tr_crop_type", "random"))
#     tr_record_parser_fn = functools.partial(parse_record, audio_parser_fn=tr_parser_fn, label_parser_fn=placeholder_label_transform)

#     val_parser_fn = functools.partial(audio_parser, clip_duration=config.data.clip_duration, crop_type=config.data.get("val_crop_type", "center"))
#     val_record_parser_fn = functools.partial(parse_record, audio_parser_fn=val_parser_fn, label_parser_fn=placeholder_label_transform)

#     batch_size = config.batch_size
#     shuffle_buffer = config.get("shuffle_buffer", 1000)

#     # spec = LogMelSpec(num_frames=config.data.num_frames)

#     tr_dataset = (
#         wds.WebDataset(tr_files, handler=wds.ignore_and_continue, shardshuffle=True)
#         .shuffle(shuffle_buffer).map(tr_record_parser_fn)
#         .to_tuple("audio.flac label")
#         # .batched(batch_size, partial=False)
#         # .map_tuple(spec, identity)
#         .with_length(tr_samples)# //config.batch_size)
#     )

#     val_dataset = (
#         wds.WebDataset(val_files, handler=wds.ignore_and_continue)
#         .map(val_record_parser_fn)
#         .to_tuple("audio.flac label")
#         # .batched(batch_size, partial=False)
#         # .map_tuple(spec, identity)
#         .with_length(val_samples)#//config.batch_size)
#     )

#     return tr_dataset, val_dataset
