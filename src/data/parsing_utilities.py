import numpy as np
import torch
import soundfile as sf
from io import BytesIO
SR = 16000


def load_audio_from_buffer(buffer, start_offset=0, frames=-1):
    with BytesIO(buffer) as fd:
        x, sr = sf.read(fd, frames=frames, start=start_offset, dtype='float32')
    return x


def audio_parser(record, clip_duration=None, crop_type="random", transforms=None):
    audio = record['audio.flac']
    duration = record.get("duration", None)
    if duration is not None:
        duration = int(duration)

    if clip_duration is not None:
        desired_samples = int(SR * clip_duration)
    else:
        desired_samples = 0
    
    start_offset = 0
    frames = -1
    x = None
    if duration is None:
        x = load_audio_from_buffer(audio)
        duration = len(x)
    
    if duration > desired_samples:
        # cropped read
        if crop_type == "random":
            start_offset = np.random.randint(0, duration-desired_samples-1)
        elif crop_type == "center":
            start_offset = (duration-desired_samples) // 2
        else:
            raise ValueError("Unknown crop type: {}, accepted values are [random,center]")
        
        frames = desired_samples
    try:
        if x is None:
            # seeked reading seemed prone to psf_seek errors
            # given that the entire audio file is already in memory, it's probably not worth the trouble
            x = load_audio_from_buffer(audio)
        
    except Exception as e:
        print("-----------------------------------")
        print("Error loading audio from buffer:", e)
        # print("record:", record)
        print(record['__key__'], record['__url__'])
        print("-----------------------------------")
        raise e
    
    try:
        x = x[start_offset:start_offset+frames]
    except Exception as e:
        print("-----------------------------------")
        print("Error cropping audio:", e)
        print("record:", record)
        print("start_offset:", start_offset, "frames:", frames)
        print("-----------------------------------")
        raise e

    if len(x) < desired_samples:
        tile_size = (desired_samples // x.shape[0]) + 1
        x = np.tile(x, tile_size)[:desired_samples]
    
    # convert to torch tensor
    x = torch.from_numpy(x)

    # apply transforms, if any
    if transforms is not None:
        x = transforms(x)

    return x


def parse_record(record, audio_parser_fn, label_parser_fn=None):
    # print("record keys:", record.keys())
    result = {
        "__key__": record['__key__'],
        "__url__": record['__url__'],
    }
    
    result['audio.flac'] = audio_parser_fn(record)
    result['label'] = record['label'] if label_parser_fn is None else label_parser_fn(record)
    return result


def np_array_loader_helper(buffer):
    with BytesIO(buffer) as fd:
       data = np.load(fd, allow_pickle=True)
    return data


def np_spec_parser(record, req_num_frames, crop_type="random", transforms=None, flip_ft=False, normalize=True):
    spec = np_array_loader_helper(record['data.pyd'])
    duration = spec.shape[0]

    if normalize:
       mean = np.mean(spec, keepdims=True)
       std = np.std(spec, keepdims=True)
       spec = (spec - mean) / (std + 1e-8)

    start_offset = 0
    if duration > req_num_frames:
        # cropped read
        if crop_type == "random":
            start_offset = np.random.randint(0, duration-req_num_frames)
        elif crop_type == "center":
            start_offset = (duration-req_num_frames) // 2
        else:
            raise ValueError("Unknown crop type: {}, accepted values are [random,center]")
    spec = spec[start_offset:start_offset+req_num_frames]
    
    if spec.shape[0] < req_num_frames:
        tile_size = (req_num_frames // spec.shape[0]) + 1
        spec = np.tile(spec, tile_size)[:req_num_frames]
    
    spec = torch.from_numpy(spec)
    
    # apply transforms, if any
    if transforms is not None:
        spec = transforms(spec)
    
    if flip_ft:
       spec = spec.transpose(1, 0)    

    spec = spec.unsqueeze(0)
    
    return spec


def np_label_parser(record, num_classes=527):
    tgt = np_array_loader_helper(record['target.pyd'])
    output = np.sum(np.eye(num_classes)[tgt], axis=0)
    output = torch.from_numpy(output)
    return output


def numpy_record_parser(record, numpy_spec_parser_fn, label_parser_fn=np_label_parser):
    result = {
        "__key__": record['__key__'],
        "__url__": record['__url__'],
    }

    result['audio.flac'] = numpy_spec_parser_fn(record)
    result['label'] = np_array_loader_helper(record['target.pyd']) if label_parser_fn is None else label_parser_fn(record)
    return result
