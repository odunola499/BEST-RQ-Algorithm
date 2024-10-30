import torch
from torchaudio import transforms as T
from torch import nn
from einops import rearrange


# whisper's feature extractor parameter instead?
def normalize_features(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True)
    return (features - mean) / (std + 1e-10)


class FeatureExtractor:
    def __init__(self,
                 n_mels = 80,
                 sample_rate = 16000,
                 win_length = 960,
                ):
        hop_length =  win_length // 4
        n_fft = 2 * win_length

        self.feature_extractor = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft
            ),
            T.AmplitudeToDB()
        )
        self.feature_extractor.requires_grad = False

    def __call__(self, x: torch.Tensor):
        features = self.feature_extractor(x)
        features = rearrange(features, "b c n -> b n c")
        features = normalize_features(features)
        return features

if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    audio = torch.randn(8, 24000)
    output = feature_extractor(audio)
    print(output.shape)