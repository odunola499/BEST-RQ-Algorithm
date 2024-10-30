import torch
from torch import nn
from .feed_forward import FeedForwardModule
from .spec_augment import SpecAugment
from .modules import ResidualConnectionModule
from .attention import MultiHeadedSelfAttention
from .convolution import ConformerConvModule
from .subsampling import ConvSubsampling
import torchaudio

class ConformerBlock(nn.Module):
    def __init__(self,
                 encoder_dim:int = 512,
                 num_attention_heads:int = 8,
                 feed_forward_expansion_factor:int = 4,
                 conv_expansion_factor:int = 2,
                 feed_forward_dropout_p:float = 0.1,
                 attention_dropout_p:float = 0.1,
                 conv_dropout_p: float = 0.1,
                 conv_kernel_size:int = 31,
                 half_step_residual:bool = True
                 ):
        super(ConformerBlock, self).__init__()

        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttention(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self,inputs:torch.Tensor):
        return self.sequential(inputs)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        num_heads: int = 8,
        encoder_dim: int = 512,
        num_layers: int = 12,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        subsampling_factor: int = 4,
        half_step_residual: bool = True,
        grad_ckpt_batchsize: int = 4,
    ):
        super().__init__()
        self.grad_ckpt_batchsize = grad_ckpt_batchsize

        self.conv_subsampling = ConvSubsampling(
            input_dim=input_dim,
            feat_out=encoder_dim,
            conv_channels=encoder_dim,
            subsampling_factor=subsampling_factor,
        )
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.Dropout(p=dropout)
        )
        module_list = [
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                feed_forward_dropout_p=dropout,
                attention_dropout_p=dropout,
                conv_dropout_p=dropout,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual,
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.Sequential(*module_list)

    def forward(self, x, lengths = None):
        x, lengths = self.conv_subsampling(x, lengths)
        x = self.input_projection(x)
        x = self.layers(x)
        return x


if __name__ == "__main__":
    model = ConformerEncoder()
    spect_func = torchaudio.transforms.Spectrogram(n_fft=159)
    #feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-large')
    num_params = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {num_params}")

    audio = torch.randn(2, 3000, 80)
    shape = torch.LongTensor([16000])
    #print(audio.shape)

    spect = spect_func(audio)
    length = spect.shape[0]
    print(spect.shape)
    output = model(audio)
    print(output.shape)





