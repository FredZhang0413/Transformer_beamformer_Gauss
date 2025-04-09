import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#############################################
# Cross-Attention Block (for Antenna-level and User-level)
#############################################
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_pdrop, resid_pdrop):
        super(CrossAttentionBlock, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Linear projections for Query, Key, and Value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(d_model, d_model)
        
        # LayerNorm and MLP (similar to the sublayer in Transformer blocks)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),  # Use GELU activation
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
    
    def forward(self, query, kv):
        """
        Args:
            query: Query token sequence, shape (B, L_q, d_model)
            kv: Key and Value token sequence, shape (B, L_k, d_model)
        Returns:
            Output with shape (B, L_q, d_model)
        """
        B, L_q, _ = query.size()
        B, L_k, _ = kv.size()

        # Apply LayerNorm on inputs before attention (pre-norm residual path)
        query_ln = self.ln1(query)
        kv_ln = self.ln1(kv)
        
        # Compute Q, K, V and reshape to multi-head format
        Q = self.q_proj(query_ln).view(B, L_q, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, L_q, head_dim)
        K = self.k_proj(kv_ln).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)       # (B, n_head, L_k, head_dim)
        V = self.v_proj(kv_ln).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)       # (B, n_head, L_k, head_dim)
        
        # Compute attention scores (scaled dot-product attention)
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, L_q, L_k)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        y = att @ V  # (B, n_head, L_q, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, L_q, self.n_head * self.head_dim)  # (B, L_q, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
        
        # Residual connection
        out = query + y
        # Further process with MLP (with residual connection)
        out = out + self.mlp(self.ln2(out))
        return out

#############################################
# Beamforming Transformer (Modified Version)
#############################################
class BeamformingTransformer(nn.Module):
    def __init__(self, config):
        super(BeamformingTransformer, self).__init__()
        self.config = config
        # Set dimensions
        self.K = config.num_users   # Number of users
        self.N = config.num_tx      # Number of transmit antennas
        
        # Define projection layers for tokens to map to d_model dimensions
        # For antenna-level: project each column token (original dimension = num_users)
        self.antenna_channel_proj = nn.Linear(self.K, config.d_model)
        self.antenna_beam_proj    = nn.Linear(self.K, config.d_model)
        # For user-level: project each row token (original dimension = num_tx)
        self.user_channel_proj = nn.Linear(self.N, config.d_model)
        self.user_beam_proj    = nn.Linear(self.N, config.d_model)
        
        # Define position embeddings for antenna and user tokens separately
        # Each token sequence gets a positional embedding with shape (sequence_length, d_model)
        self.pos_emb_ant = nn.Parameter(th.zeros(self.N, config.d_model))
        self.pos_emb_user = nn.Parameter(th.zeros(self.K, config.d_model))
        
        # Define two cross-attention blocks for antenna-level and user-level attention
        self.cross_attn_ant = CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
                                                    attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        self.cross_attn_user = CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
                                                     attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        
        # Final fusion and output MLP: after flattening the antenna-level and user-level representations,
        # the concatenated vector is mapped to beam_dim (typically 2 * num_tx * num_users) for the predicted beamformer.
        self.out_proj = nn.Sequential(
            nn.Linear((self.N + self.K) * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.beam_dim)
        )
        
        # Weight initialization
        self.apply(self._init_weights)
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, H, W_prev):
        """
        Args:
            H: Constant channel matrix, shape (B, num_users, num_tx) i.e. (B, K, N)
            W_prev: Previous beamformer, shape (B, num_users, num_tx) i.e. (B, K, N)
        Returns:
            W_next: Vectorized predicted beamformer, shape (B, beam_dim)
        """
        B = H.size(0)
        K = self.K  # Number of users
        N = self.N  # Number of transmit antennas
        
        # ---------------------------
        # 1. Antenna-level Cross-Attention
        # ---------------------------
        # For antenna-level, treat tokens as columns: transpose H and W_prev to get shape (B, num_tx, num_users)
        H_ant = H.transpose(1, 2)      # (B, N, K)
        W_ant = W_prev.transpose(1, 2)   # (B, N, K)
        
        # Project tokens to d_model space
        # Each token originally has dimension num_users; after projection, dimension becomes d_model
        H_ant_proj = self.antenna_channel_proj(H_ant)  # (B, N, d_model)
        W_ant_proj = self.antenna_beam_proj(W_ant)       # (B, N, d_model)
        
        # Add antenna-level positional embeddings (each token indexed from 0 to N-1)
        H_ant_proj = H_ant_proj + self.pos_emb_ant.unsqueeze(0)  # (B, N, d_model)
        W_ant_proj = W_ant_proj + self.pos_emb_ant.unsqueeze(0)  # (B, N, d_model)
        
        # Apply cross-attention: use H_ant_proj as Query, W_ant_proj as Key and Value
        x_a = self.cross_attn_ant(H_ant_proj, W_ant_proj)  # (B, N, d_model)
        
        # ---------------------------
        # 2. User-level Cross-Attention
        # ---------------------------
        # For user-level, tokens are along rows: H and W_prev are already (B, num_users, num_tx)
        H_user = H    # (B, K, N)
        W_user = W_prev  # (B, K, N)
        
        # Project tokens to d_model space (each token originally has dimension num_tx)
        H_user_proj = self.user_channel_proj(H_user)  # (B, K, d_model)
        W_user_proj = self.user_beam_proj(W_user)       # (B, K, d_model)
        
        # Add user-level positional embeddings (each token indexed from 0 to K-1)
        H_user_proj = H_user_proj + self.pos_emb_user.unsqueeze(0)  # (B, K, d_model)
        W_user_proj = W_user_proj + self.pos_emb_user.unsqueeze(0)  # (B, K, d_model)
        
        # Apply cross-attention: use H_user_proj as Query, W_user_proj as Key and Value
        x_u = self.cross_attn_user(H_user_proj, W_user_proj)  # (B, K, d_model)
        
        # ---------------------------
        # 3. Fusion and Output Prediction
        # ---------------------------
        # Flatten the antenna-level and user-level outputs and concatenate
        x_a_flat = x_a.view(B, -1)  # (B, N * d_model)
        x_u_flat = x_u.view(B, -1)  # (B, K * d_model)
        x_fused = th.cat([x_a_flat, x_u_flat], dim=-1)  # (B, (N+K)*d_model)
        
        # Use an MLP to produce the final vectorized beamformer (beam_dim = 2 * num_tx * num_users)
        W_next = self.out_proj(x_fused)  # (B, beam_dim)
        # Normalize the beamformer
        norm = th.norm(W_next, dim=1, keepdim=True)
        W_next = W_next / norm
        return W_next

#############################################
# Example: Config class and model call
#############################################
class BeamformerTransformerConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs['d_model']          # Token dimension in Transformer
        self.d_action = kwargs['d_action']          # Reserved parameter, not used here
        self.beam_dim = kwargs['beam_dim']          # Dimension of beamformer vector (2 * num_users * num_tx)
        self.n_head = kwargs['n_head']              # Number of attention heads
        self.attn_pdrop = kwargs['attn_pdrop']
        self.resid_pdrop = kwargs['resid_pdrop']
        self.num_users = kwargs['num_users']        # K
        self.num_tx = kwargs['num_tx']              # N
        # Other training parameters remain unchanged
        # ...

if __name__ == "__main__":
    # Example parameter settings
    num_users = 16
    num_tx = 16
    d_model = 32  # Token dimension for Transformer
    beam_dim = 2 * num_tx * num_users  # Output beamformer vectorized size
    n_head = 8
    attn_pdrop = 0.05
    resid_pdrop = 0.05
    
    # Create configuration object
    config = BeamformerTransformerConfig(
        d_model=d_model,
        d_action=16,            # This parameter is not used in the new design
        beam_dim=beam_dim,
        n_head=n_head,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        num_users=num_users,
        num_tx=num_tx
    )
    
    # Instantiate the model
    model = BeamformingTransformer(config)
    
    # Simulate input: assume batch size of 8, H and W_prev have shape (B, num_users, num_tx)
    B = 8
    H = th.randn(B, num_users, num_tx)
    W_prev = th.randn(B, num_users, num_tx)
    
    # Forward pass
    W_next = model(H, W_prev)
    print("W_next shape:", W_next.shape)  # Expected shape (B, beam_dim)
