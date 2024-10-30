import torch
from torch import nn
from torch.nn import functional as F

from vector_quantize_pytorch.random_projection_quantizer import RandomProjectionQuantizer
from einops import rearrange
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def compute_effective_mask_ratio(mask_indices):
    return mask_indices.float().mean().item()

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

def set_eos_id(t: torch.Tensor, eos_id: int, pad_id: int):
    eos_indices = ((t == pad_id).cumsum(dim = -1) == 0).sum(dim = -1, keepdim = True).long()

    batch_range = torch.arange(t.shape[0], device = t.device, dtype = torch.long)
    batch_range = rearrange(batch_range, '... -> ... 1')

    t = F.pad(t, (0, 1), value = pad_id)
    t[batch_range, eos_indices] = eos_id
    return t

class Quantizer(nn.Module):
    def __init__(self,
                 n_mels = 80,
                 codebook_size = 1000,
                 codebook_dim = 512,
                 norm = False):
        super().__init__()
        self.rpq = nn.Sequential(
            nn.LayerNorm(n_mels, elementwise_affine = False),
            RandomProjectionQuantizer(dim = n_mels,
                                      codebook_size = codebook_size,
                                      codebook_dim = codebook_dim,
                                      norm = norm)
        )
        self.rpq.requires_grad = False
        self.pad_id = 0
        self.eos_id = codebook_size + 1
        self.mask_prob = 0.6

    #good mel and corrupted mel both get codebooks

    def forward(self,features:torch.Tensor):
        with torch.no_grad():
            labels = self.rpq(features)
            #labels = torch.where(labels != self.pad_id, labels + 1, self.pad_id)
        return labels

class RQMasking(nn.Module):
    def __init__(self,
                 mask_prob=0.15,           # Probability of starting a mask sequence
                 mask_length=4,            # Length of mask sequence
                 mask_value=0.0,           # Value to use for masking
                 batch_first=True):        # Whether batch dimension is first
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_value = mask_value
        self.batch_first = batch_first

    def forward(self, x, mask_indices=None):
        batch_size, sequence_length, feature_dim = x.shape

        if mask_indices is None:
            # Initialize mask starting points
            mask_starts = torch.bernoulli(
                torch.ones(batch_size, sequence_length) * self.mask_prob
            ).bool().to(x.device)

            # Ensure we don't start new masks too close to the end
            mask_starts[:, -self.mask_length + 1:] = False

            # Create the full mask by extending each starting point
            mask_indices = torch.zeros(batch_size, sequence_length,
                                       dtype=torch.bool, device=x.device)

            # For each mask start position, mask the next mask_length frames
            for i in range(self.mask_length):
                mask_indices = mask_indices | torch.roll(mask_starts, shifts=i, dims=1)

        masked_x = x.clone()
        masked_x[mask_indices] = self.mask_value
        return masked_x, mask_indices



if __name__ == "__main__":
    quantizer = Quantizer(n_mels = 80, codebook_size = 1000, codebook_dim = 512, norm = False)
    features = torch.randn(8, 101, 80)
    masker = RQMasking()
    masked_features, mask_indices = masker(features)
    print(f"Masked features: {masked_features.shape}")
    print(f"Mask indices: {mask_indices.shape}")

    tokens = quantizer(features)
    print(tokens.shape)









    




