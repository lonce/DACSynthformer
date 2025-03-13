import torch
import torch.nn as nn

from .MultiheadAttentionWithRoPE import MultiheadAttentionWithRoPE



class ConditioningEmbedding(nn.Module):
    """
    Embed the conditioning vector before injecting it with FiLM
    """
    def __init__(self, cond_dim, embed_dim, hidden_dim=64):
        """
        Maps the full conditioning vector (pitch, loudness, instrument) into an embedding.

        Args:
            cond_dim (int): Total size of the conditioning vector (float + one-hot).
            embed_dim (int): Output embedding dimension for FiLM.
            hidden_dim (int): Hidden size for MLP processing.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),  # Map raw inputs to hidden space
            nn.ReLU(), 
            nn.Linear(hidden_dim, embed_dim)  # Map to final embedding space
        )

    def forward(self, cond):
        return self.mlp(cond)  # Shape: (batch, embed_dim)





"""
In this version of the Transformer, the positional encoding is applied dynamically to the query and key vectors within each Transformer block, as required for Rotary Positional Encoding (RoPE). The redundant RoPE application to the initial embedding has been removed. All other logic, comments, and verbosity remain unchanged.
"""

# Rotary Positional Embedding
# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(self, dim, max_len=1024):
#         super().__init__()
#         self.dim = dim // 2
#         inv_freq = 1.0 / (10000 ** (torch.arange(0.0, self.dim, 2.0) / self.dim))
#         t = torch.arange(max_len).unsqueeze(1)
#         freqs = t * inv_freq.unsqueeze(0)
#         self.register_buffer('sinusoid', torch.cat([freqs.sin(), freqs.cos()], dim=-1))

#     def forward(self, qk):
#         #print(f"------- In Rotary forward, x.shape is ={x.shape}")
#         n, seq_len, d = qk.shape
#         sinusoid = self.sinusoid[:seq_len, :].to(qk.device)
#         sinusoid = sinusoid.repeat_interleave(2, dim=1)  # Ensure sinusoid covers all dimensions
#         sin_part, cos_part = sinusoid[:, :d//2], sinusoid[:, d//2:]

#         qk_sin = qk[:, :, :d//2] * sin_part - qk[:, :, d//2:] * cos_part
#         qk_cos = qk[:, :, :d//2] * cos_part + qk[:, :, d//2:] * sin_part

#         return torch.cat((qk_sin, qk_cos), dim=-1)


# ------------------------------------------------------------------------------
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, self.dim, 2.0) / self.dim))
        t = torch.arange(max_len).unsqueeze(1)
        freqs = t * inv_freq.unsqueeze(0)
        sinusoid = torch.cat([freqs.sin(), freqs.cos()], dim=-1)

        # Precompute interleaved sin/cos parts separately
        sinusoid = sinusoid.repeat_interleave(2, dim=1)
        self.register_buffer('sin_part', sinusoid[:, :dim//2])
        self.register_buffer('cos_part', sinusoid[:, dim//2:])

    def forward(self, qk):
        n, seq_len, d = qk.shape
        sin_part = self.sin_part[:seq_len, :].to(qk.device)
        cos_part = self.cos_part[:seq_len, :].to(qk.device)

        qk_sin = qk[:, :, :d//2] * sin_part - qk[:, :, d//2:] * cos_part
        qk_cos = qk[:, :, :d//2] * cos_part + qk[:, :, d//2:] * sin_part

        return torch.cat((qk_sin, qk_cos), dim=-1)
# ------------------------------------------------------------------------------

# # Multiembedding Layer
# class MultiEmbedding(nn.Module):
#     def __init__(self, vocab_size, per_token_embed_size, num_tokens):
#         super().__init__()
#         self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, per_token_embed_size) for _ in range(num_tokens)])

#     def forward(self, x):
#         # x shape: (batch, seq_len, num_tokens)
#         embeddings = [self.embeddings[i](x[:, :, i]) for i in range(len(self.embeddings))]
#         #print(f"------- In MultiEmbedding will return a vector of shape  ={torch.cat(embeddings, dim=-1).shape}")
#         return torch.cat(embeddings, dim=-1)  # Concatenate embeddings along the last dimension


# This version allows us to pass either integer tokens (in which case nn.Embedding is used for efficiency) OR 
# vectors of floating point data (eg mel spectrograms - one "token") OR multiple tokens of n-D floaating point values (e.g. 
# DAC latent space codes) that would be embedded independently and concatenated.  


class MultiEmbedding(nn.Module):
    def __init__(self, vocab_size_or_input_dim, embed_dim, num_tokens, input_type="int"):
        """
        MultiEmbedding class that supports either integer token embeddings (nn.Embedding)
        or floating-point token projections (nn.Linear), while ensuring input always has 3D shape.

        Args:
            vocab_size_or_input_dim (int): 
                - If `input_type="int"`, this is the vocabulary size.
                - If `input_type="float"`, this is the per-token input feature dimension.
            embed_dim (int): The total embedding dimension (split across `num_tokens`).
            num_tokens (int): Number of tokens per input.
            input_type (str): "int" for integer tokens (uses nn.Embedding), "float" for float tokens (uses nn.Linear).
        """
        super().__init__()

        print(f'Initializing MultiEmbedding with vocab_size_or_input_dim={vocab_size_or_input_dim}, embed_dim = {embed_dim}, and num_tokens = {num_tokens}')

        self.input_type = input_type.lower()
        self.num_tokens = num_tokens
        self.split_embed_dim = embed_dim // num_tokens  # Each token gets embed_dim / num_tokens

        if self.input_type == "int":
            # Use nn.Embedding for integer tokens
            self.layers = nn.ModuleList([
                nn.Embedding(vocab_size_or_input_dim, self.split_embed_dim) 
                for _ in range(num_tokens)
            ])
        elif self.input_type == "float":
            # Use nn.Linear for continuous tokens
            self.layers = nn.ModuleList([
                nn.Linear(vocab_size_or_input_dim, self.split_embed_dim) 
                for _ in range(num_tokens)
            ])
        else:
            raise ValueError("input_type must be either 'int' or 'float'.")

    def forward(self, x):
        """
        Forward pass ensuring `x` is always 3D.

        Args:
            x (Tensor): 
                - If `input_type="int"`: Shape (batch, seq_len, num_tokens), containing integer indices.
                - If `input_type="float"`: Shape (batch, seq_len, num_tokens * vocab_size_or_input_dim), 
                  containing concatenated floating-point token vectors.

        Returns:
            Tensor: Shape (batch, seq_len, embed_dim), where `embed_dim` is distributed across tokens.
        """
        batch_size, seq_len, stacked_dim = x.shape  # Always 3D

        if self.input_type == "int":
            # Ensure integer input shape is valid
            if stacked_dim != self.num_tokens:
                raise ValueError(f"Expected input shape (batch, seq_len, {self.num_tokens}) for integer tokens, got {x.shape}")

            # Apply embedding to each token independently
            embeddings = torch.stack([self.layers[i](x[:, :, i]) for i in range(self.num_tokens)], dim=2)

        elif self.input_type == "float":
            # Ensure float input shape is valid
            input_dim_per_token = self.layers[0].in_features
            expected_dim = self.num_tokens * input_dim_per_token

            if stacked_dim != expected_dim:
                raise ValueError(f"Expected input shape (batch, seq_len, {expected_dim}) for float tokens, got {x.shape}")

            # Reshape to (batch, seq_len, num_tokens, input_dim_per_token)
            x = x.view(batch_size, seq_len, self.num_tokens, input_dim_per_token)

            # Apply linear projection independently for each token
            embeddings = torch.stack([self.layers[i](x[:, :, i, :]) for i in range(self.num_tokens)], dim=2)

        # Merge num_tokens dimension
        embeddings = embeddings.view(batch_size, seq_len, -1)  # Final shape: (batch, seq_len, embed_dim)
        return embeddings


#--------------------------------------------------------------
#- class FiLM(nn.Module):
#-     def __init__(self, cond_size, embed_size, verbose=0):
#-         super(FiLM, self).__init__()
#-         self.linear_gamma = nn.Linear(cond_size, embed_size)  # Scale factor
#-         self.linear_beta = nn.Linear(cond_size, embed_size)   # Shift factor
#-         self.verbose=verbose
#-     
#-     def forward(self, x, cond):
#-         if self.verbose > 5:
#-             print(f"FiLM cond.shape is : {cond.shape}")
#-             
#-         gamma = self.linear_gamma(cond)
#-         beta = self.linear_beta(cond)
#-         return gamma * x + beta


class FiLM(nn.Module):
    def __init__(self, cond_size, embed_size, verbose=0):
        """
        FiLM modulation with an MLP-based conditioning embedding.
        
        Args:
            cond_size (int): Size of the raw conditioning vector.
            embed_size (int): Size of the transformer's sequence embeddings.
        """
        super(FiLM, self).__init__()
        
        # Add the conditioning embedding
        self.cond_embedding = ConditioningEmbedding(cond_size, 64)
        
        # FiLM scaling and shifting
        self.linear_gamma = nn.Linear(64, embed_size)  # Scale factor
        self.linear_beta = nn.Linear(64, embed_size)   # Shift factor
        self.verbose = verbose

    def forward(self, x, cond):
        # First, compute a learned embedding of the conditioning vector
        cond_emb = self.cond_embedding(cond)

        if self.verbose > 5:
            print(f"FiLM cond.shape is : {cond_emb.shape}")

        # Compute FiLM parameters
        gamma = self.linear_gamma(cond_emb)
        beta = self.linear_beta(cond_emb)

        return gamma * x + beta


#--------------------------------------------------------------
class AdaLN(nn.Module):
    def __init__(self, cond_size, embed_size):
        super(AdaLN, self).__init__()
        self.linear_mu = nn.Linear(cond_size, embed_size)
        self.linear_sigma = nn.Linear(cond_size, embed_size)
        self.linear_gamma = nn.Linear(cond_size, embed_size)
        self.linear_beta = nn.Linear(cond_size, embed_size)

    def forward(self, x, cond):
        # Compute adaptive mean and standard deviation from conditioning
        mu = self.linear_mu(cond)
        sigma = torch.exp(self.linear_sigma(cond))  # Ensure positivity
        norm_x = (x - mu) / (sigma + 1e-6)  # Apply adaptive normalization

        gamma = self.linear_gamma(cond)
        beta = self.linear_beta(cond)
        
        return gamma * norm_x + beta

#--------------------------------------------------------------
# Transformer Block with Toggleable FiLM or AdaLN
class TransformerBlock(nn.Module):
    def __init__(self, cond_size, embed_size, num_heads, dropout, forward_expansion, rotary_positional_embedding, use_adaLN=False, verbose=0):
        super(TransformerBlock, self).__init__()

        self.embed_size = embed_size
        self.verbose = verbose
        self.use_adaLN = use_adaLN  # Toggle between FiLM and AdaLN
        self.attention = MultiheadAttentionWithRoPE(
            embed_dim=embed_size,
            num_heads=num_heads,
            rotary_positional_embedding=rotary_positional_embedding,
            verbose=verbose,
            dropout=dropout,
            bias=True,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(embed_size, elementwise_affine=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout_layer = nn.Dropout(dropout)
        
        # Choose between FiLM and AdaLN
        self.conditioning1 = AdaLN(cond_size, embed_size) if use_adaLN else FiLM(cond_size, embed_size, verbose)
        self.conditioning2 = AdaLN(cond_size, embed_size) if use_adaLN else FiLM(cond_size, embed_size, verbose)

    def forward(self, src, cond, mask=None):
        if not self.use_adaLN:
            src = self.norm1(src)
        if self.verbose > 0:
            print(f"Conditioning-modulated shape before attention: {src.shape}")

        # Apply conditioning (FiLM or AdaLN) before attention
        modulated_src = self.conditioning1(src, cond)
        
        # Pass through attention with RoPE
        attention_output, _ = self.attention(modulated_src, modulated_src, modulated_src, attn_mask=mask)
        x = self.dropout_layer(attention_output) + modulated_src

        if not self.use_adaLN:
            x = self.norm2(x)
        
        # Apply conditioning (FiLM or AdaLN) before the MLP
        modulated_x = self.conditioning2(x, cond)

        # Apply feed-forward network
        forward = self.feed_forward(modulated_x)
        out = self.dropout_layer(forward) + x

        return out

#-------------------------------------------------------------------
class RopeCondDACTransformer(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, forward_expansion, dropout, max_len, num_classes, num_codebooks, vocab_size, cond_size, input_type="float", use_adaLN=False, verbose=0):
        super(RopeCondDACTransformer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.max_len = max_len
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.cond_size = cond_size
        self.verbose = verbose
        self.use_adaLN = use_adaLN  # Toggle for conditioning type

        assert (self.embed_size // num_heads) * num_heads == self.embed_size, f"embed_dim ({self.embed_size}) must be divisible by num_heads ({num_heads})"
        print(f"Setting up MultiEmbedding with vocab_size= {vocab_size}, embed_size= {embed_size}, num_codebooks= {num_codebooks}")
        
        #self.multi_embedding = MultiEmbedding(vocab_size, embed_size // num_codebooks, num_codebooks, input_type=input_type)
        self.multi_embedding = MultiEmbedding(vocab_size, embed_size , num_codebooks, input_type=input_type)

        self.positional_embedding = RotaryPositionalEmbedding(embed_size, max_len)
        self.dropout_layer = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(
                cond_size=cond_size,
                embed_size=embed_size,
                num_heads=num_heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                rotary_positional_embedding=self.positional_embedding,
                use_adaLN=use_adaLN,
                verbose=verbose
            )
            for _ in range(num_layers)
        ])

        self.final_layer = nn.Linear(embed_size, num_codebooks * vocab_size)

    def forward(self, src, cond, mask=None):
        """
        Forward pass for the TransformerDecoder.

        Args:
        - src (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_codebooks).
        - cond (torch.Tensor): Conditioning tensor of shape (batch_size, seq_len, cond_size).
        - mask (torch.Tensor): Attention mask of shape (seq_len, seq_len).

        Returns:
        - logits (torch.Tensor): Output logits reshaped to match target shape.
        """
        if self.verbose > 5:
            print(f"Source shape: {src.shape}")
            print(f"Condition shape: {cond.shape}")
            if mask is not None:
                print(f"Mask shape: {mask.shape}")

        src = self.multi_embedding(src)
        src = self.dropout_layer(src)

        for i, layer in enumerate(self.layers):
            if self.verbose > 6:
                print(f"Passing through layer {i}")
            src = layer(src, cond, mask)

        logits = self.final_layer(src)
        logits = logits.view(logits.size(0), logits.size(1), self.num_codebooks, self.vocab_size)

        if self.verbose > 0:
            print(f"Output shape: {logits.shape}")
            print(f"================================================================")

        return logits

