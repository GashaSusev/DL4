import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
    

def test():
    batch_size = 64
    seq_len = 256
    hidden_dim = 512
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    custom_norm = RMSNorm(hidden_dim)
    torch_norm = nn.RMSNorm(hidden_dim)
    
    out_custom = custom_norm(x)
    out_torch = torch_norm(x)
    
    print("the norm is the same:", torch.allclose(out_custom, out_torch, atol=1e-6))
    max_diff = torch.max(torch.abs(out_custom - out_torch))
    print(f"Maximum diff: {max_diff:.6f}")


test() 