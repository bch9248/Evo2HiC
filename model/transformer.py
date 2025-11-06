import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, max_relative_position=16, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.max_rel_pos = max_relative_position

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * max_relative_position + 1, self.head_dim))

    def _relative_positions(self, seq_len):
        # [seq_len, seq_len] relative distance matrix
        range_vec = torch.arange(seq_len, device=self.rel_pos_emb.device)
        rel_pos = range_vec[None, :] - range_vec[:, None]  # [T, T]
        rel_pos = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        return rel_pos

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, T, nhead, head_dim]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # [B, nhead, T, head_dim]

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [B, nhead, T, T]

        # Relative position contribution
        rel_pos = self._relative_positions(T)  # [T, T]
        rel_emb = self.rel_pos_emb[rel_pos]    # [T, T, head_dim]
        rel_scores = torch.einsum('bnth,tsh->bnts', q, rel_emb)  # [B, nhead, T, T]

        attn_scores = (attn_scores + rel_scores) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, nhead, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)

class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, max_relative_position=16):
        super().__init__()
        self.self_attn = RelativeSelfAttention(d_model, nhead, max_relative_position, dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # Self-attention + residual + norm
        x2 = self.self_attn(x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        # Feedforward + residual + norm
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x