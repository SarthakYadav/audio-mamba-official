import torch
import torch.nn as nn
import torchaudio


class LogMelSpec(nn.Module):
    def __init__(
        self, 
        sr=16000,
        n_mels=80,
        n_fft=400,
        win_len=400,
        hop_len=160,
        f_min=50.,
        f_max=8000.,
        normalize=True,
        flip_ft=True,
        num_frames=None
    ) -> None:
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, win_length=win_len, hop_length=hop_len,
            f_min=f_min, f_max=f_max,
            n_mels=n_mels, power=2.
        )
        self.normalize = normalize
        self.flip_ft = flip_ft
        if num_frames is not None:
            self.num_frames = int(num_frames)
        else:
            self.num_frames = None

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.melspec(x)
        x = (x + torch.finfo().eps).log()
        if self.num_frames is not None:
            x = x[:, :, :self.num_frames]
    
        # print("in LogMelSpec, x.shape after melspec", x.shape)
        if self.normalize:
            mean = torch.mean(x, [1, 2], keepdims=True)
            # print("in LogMelSpec, mean.shape", mean.shape)
            std = torch.std(x, [1, 2], keepdims=True)
            x = (x - mean) / (std + 1e-8)
        
        if self.flip_ft:
            x = x.transpose(-2, -1)
            # print("in LogMelSpec, x.shape after flip_ft", x.shape)
        x = x.unsqueeze(1)
        # print("in LogMelSpec, x.shape after normalization", x.shape)
        return x

# log_mel_spec = LogMelSpec()