from torch import nn
import torch
from encoder.model import ConformerEncoder
from best_rq.quantizer import RQMasking, Quantizer
from feature_extractor import FeatureExtractor
import yaml

class ConformerEncoderForPretrain(nn.Module):
    def __init__(self, config, train = True):
        super(ConformerEncoderForPretrain, self).__init__()

        encoder_config = config.get('encoder_params', None)
        feature_config = config.get('feature_params', None)
        rq_config = config.get('rq_params', None)
        mask_config = config.get('mask_params', None)

        self.encoder = ConformerEncoder(**encoder_config)
        self.feature_extractor = FeatureExtractor(**feature_config)
        self.rq_masker = RQMasking(**mask_config)
        self.quantizer = Quantizer(**rq_config)
        self.loss_fn = nn.CrossEntropyLoss()

        if train:
            self.output_module = nn.Sequential(
                nn.Conv1d((667 // 4) + 1,667, 1, 1, padding = 'same'),
                nn.LayerNorm(encoder_config['encoder_dim']),
                nn.Linear(encoder_config['encoder_dim'], rq_config['codebook_size'])
            )

    @torch.no_grad()
    def get_features(self, inputs: torch.Tensor):
        assert inputs.ndim == 2, 'mono audio'
        features = self.feature_extractor(inputs)
        return features

    @torch.no_grad()
    def get_labels(self, features:torch.Tensor): # takes in features
        labels = self.quantizer(features)
        return labels

    @torch.no_grad()
    def get_masked_features(self, features:torch.Tensor): # takes in features
        masked_features, indices = self.rq_masker(features)
        return masked_features

    def forward(self, inputs: torch.Tensor, compute_loss = True):
        features = self.get_features(inputs)
        print(features.shape)
        if compute_loss:
            labels = self.get_labels(features)
            masked_features = self.get_masked_features(features)
            output = self.encoder(masked_features)
            logits = self.output_module(output)
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
            loss = self.loss_fn(logits, labels)
            return loss
        return self.encoder(features)


if __name__ == '__main__':
    config = yaml.safe_load(open('config.yml'))
    conformer = ConformerEncoderForPretrain(config)
    num_params = sum([p.numel() for p in conformer.parameters()])
    print(f"Number of parameters: {num_params}")

    array = torch.randn(8, 10 * 16_000)
    loss = conformer(array, True)
    print(loss)



