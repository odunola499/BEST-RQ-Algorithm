import math
import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention


def relative_positional(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx)

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model = 512,
                 num_heads = 16,
                 dropout_p = 0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.n_head = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        B,T,C = inputs.size()
        q, k, v = self.c_attn(inputs).split(self.d_model, dim = 2)
        k.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B,T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attn_score = flex_attention(q, k, v, score_mod = relative_positional)
        attn_score = attn_score.transpose(1, 2).contiguous().view(B, T, C)

        y = self.dropout(self.out_proj(attn_score))


        return y


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self,
                 d_model = 512,
                 num_heads = 16,
                 dropout_p = 0.1):
        super(MultiHeadedSelfAttention, self).__init__()
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs)
        return outputs




