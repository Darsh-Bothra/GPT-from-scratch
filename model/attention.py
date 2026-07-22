import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.Key = nn.Linear(in_features=embedding_dim, out_features=attention_dim, bias=False)
        self.Query = nn.Linear(in_features=embedding_dim, out_features=attention_dim, bias=False)
        self.Value = nn.Linear(in_features=embedding_dim, out_features=attention_dim, bias=False)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places

        # 1.
        K = self.Key(embedded)
        Q = self.Query(embedded)
        V = self.Value(embedded)

        # 2. 
        ctx_len, attn_dim = K.shape[1], K.shape[2]
        attn_score = (Q @ torch.transpose(K, 1, 2)) / (attn_dim ** 0.5)

        # 3.
        mask = torch.tril(torch.ones(ctx_len, ctx_len))
        scores = attn_score.masked_fill(mask == 0, float('-inf'))

        # 4. 
        attn_weights = nn.functional.softmax(scores, dim=-1)

        # 5.
        ans = attn_weights @ V

        return torch.round(ans, decimals=4)