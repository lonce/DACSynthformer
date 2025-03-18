import torch
import torch.nn as nn
import torch.nn.functional as F




class MultiheadAttentionWithRoPE(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, rotary_positional_embedding, verbose=0, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.rotary_positional_embedding = rotary_positional_embedding
        self.verbose=verbose
        if verbose > 0 :
            print(f"Initializing MultiheadAttentionWithRoPE with embed_dim={embed_dim}, and num_heads={num_heads}")
            
            
    def forward(self, query, key, value, **kwargs):
        if self.verbose > 0 :
            print(f"MultiheadAttentionWithRoPE with query.shape={query.shape}, key.shape={key.shape}, value.shape={value.shape}")

        # Manually compute Q, K, V projections
        q = F.linear(query, self.in_proj_weight[:self.embed_dim], self.in_proj_bias[:self.embed_dim])
        k = F.linear(key, self.in_proj_weight[self.embed_dim:2*self.embed_dim], self.in_proj_bias[self.embed_dim:2*self.embed_dim])
        v = F.linear(value, self.in_proj_weight[2*self.embed_dim:], self.in_proj_bias[2*self.embed_dim:])

        if self.verbose > 0 :
            print(f"MultiheadAttentionWithRoPE with q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
            print(f"MultiheadAttentionWithRoPE with q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}")
            
        # Apply Rotary Positional Encoding (RoPE) to Q and K
        q = self.rotary_positional_embedding(q)
        k = self.rotary_positional_embedding(k)

        if self.verbose > 0 :
            print(f"After rotary coding, q.shape={q.shape}, k.shape={k.shape}")

            
        # Use the parent's forward method for attention computation
        return super().forward(q, k, v, **kwargs)