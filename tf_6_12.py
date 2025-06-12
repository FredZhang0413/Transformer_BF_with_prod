
import math
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# from baseline_mmse import compute_mmse_beamformer
from pdb import set_trace as bp
import torch.nn.functional as fun
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import os

mse_loss_fn = nn.MSELoss()
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

## wmmse: 74.48693517604116；ZF: 64.0581446998507 (24*24 Gaussian channel)
## wmmse: 52.07755418089099；ZF: 43.94968651013214 (16*16 Gaussian channel)
## wmmse: 27.24612893140909；ZF: 21.29116212713009 (8*8 Gaussian channel)
## wmmse: 14.27623317904294；ZF: 12.13094550542433 (4*4 Gaussian channel)

##### 1. already tried
### consider learning rate scheduler (works)
### consider hard switch learning policy (done)
### consider subspace curriculum learning strategy (uncertain)
### consider increasing the episode length T (10-15)
### consider using the objective CL (works)
### T = 1, try no L2O (it works when objective CL is used)
### consider remove the weight decay in optimizer (slight improve?)
### consider tweaking the MLP modules (slight improve, without layernorm)


##### 2. waiting for the trial
### consider the contrastive learning (improtant!)
### self attention vs cross attention (important)
### consider using the hybrid-supervision training strategy


### select the optimal transformer block number and head number (ongoing)
### Do attentions within H, W and HW
### gradient clipping?
### meta-learning (MAML) / reptile method
### entropy regularization, force a wider solution space
### trust region method (TRM)
### optimzation restart 

### focus on the problem: the gradient is too sparse


###### 3. seems infeasible
### consider noise injection
### consider the learning rate warm-up at the early stage
### consider random initialization of the beamformer



#############################################
# 1. System performance: sum rate for multi-user MISO
#############################################
def sum_rate(H, W, sigma2=1.0):
    prod = th.bmm(H, W)  # (B, num_users, num_users)
    signal_power = th.abs(th.diagonal(prod, dim1=-2, dim2=-1))**2
    interference = th.sum(th.abs(prod)**2, dim=-1) - signal_power
    N = sigma2
    SINR = signal_power / (interference + N)  # (B, num_users)
    reward = th.log2(1 + SINR).sum(dim=-1).mean()
    return reward

def compute_mmse_beamformer(mat_H, K, N, P, noise_power=1, device=th.device("cuda:0" if th.cuda.is_available() else "cpu")):
    P = th.diag_embed(th.ones(mat_H.shape[0], 1, device=device).repeat(1, K)).to(th.cfloat) ## (B, K, K)
    eye_N = th.diag_embed((th.zeros(mat_H.shape[0], N, device=device) + noise_power))
    denominator = th.inverse(eye_N + th.bmm(mat_H.conj().transpose(1,2), th.bmm(P / K, mat_H)))
    wslnr_max = th.bmm(denominator, mat_H.conj().transpose(1,2)).transpose(1,2)
    wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)
    mat_W = th.bmm(wslnr_max, th.sqrt(P/ K))
    mat_HW = th.bmm(mat_H, mat_W.transpose(-1, -2))
    S = th.abs(th.diagonal(mat_HW, dim1=-2, dim2=-1))**2
    I = th.sum(th.abs(mat_HW)**2, dim=-1) - th.abs(th.diagonal(mat_HW, dim1=-2, dim2=-1))**2
    N = noise_power
    SINR = S/(I+N)
    return mat_W, th.log2(1+SINR).sum(dim=-1).unsqueeze(-1)

#############################################
# Modified Cross-Attention Block supporting separate Query, Key, and Value.
#############################################
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_pdrop, resid_pdrop):
        super(CrossAttentionBlock, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Linear projections for Query, Key, and Value.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(d_model, d_model)

        # New LayerNorm for self-attention
        self.ln_self = nn.LayerNorm(d_model)
        
        # LayerNorm for each sublayer.
        self.ln1_q = nn.LayerNorm(d_model)
        self.ln1_k = nn.LayerNorm(d_model)
        self.ln1_v = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # MLP block with an extra layer (scheme 2).
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.LayerNorm(4 * d_model), # TBD
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(4 * d_model, 4 * d_model),  # additional layer
            nn.LayerNorm(4 * d_model), # TBD
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
    
    def forward(self, query, key, value):
        """
        Performs cross-attention using separate Query, Key, and Value inputs.
        
        Args:
            query: Query tokens, shape (B, L_q, d_model).
            key:   Key tokens, shape (B, L_k, d_model).
            value: Value tokens, shape (B, L_v, d_model); L_k == L_v is assumed.
        Returns:
            Output with shape (B, L_q, d_model).
        """
        B, L_q, _ = query.size()
        B, L_k, _ = key.size()
        
        # Apply LayerNorm separately.
        q_ln = self.ln1_q(query) 
        k_ln = self.ln1_k(key)
        v_ln = self.ln1_v(value)
        
        # Compute Q, K, V and reshape for multi-head attention.
        Q = self.q_proj(q_ln).view(B, L_q, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, L_q, head_dim)
        K = self.k_proj(k_ln).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, L_k, head_dim)
        V = self.v_proj(v_ln).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, L_k, head_dim)
        
        # Scaled dot-product attention.
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, L_q, L_k)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Compute weighted sum of values.
        y = att @ V  # (B, n_head, L_q, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, L_q, self.n_head * self.head_dim)  # (B, L_q, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
        
        # Residual connection and MLP.
        out = query + y

        ### self-attention on updated query features
        sa_input = self.ln_self(out)  # Apply LayerNorm to the output of the attention block.

        # Compute Q, K, V for self-attention (all from the same tokens)
        Qs = self.q_proj(sa_input).view(B, L_q, self.n_head, self.head_dim).transpose(1, 2)
        Ks = self.k_proj(sa_input).view(B, L_q, self.n_head, self.head_dim).transpose(1, 2)
        Vs = self.v_proj(sa_input).view(B, L_q, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product self-attention
        att_s = (Qs @ Ks.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att_s = F.softmax(att_s, dim=-1)
        att_s = self.attn_drop(att_s)

        # Compute weighted sum of self-attention values
        y_s = att_s @ Vs
        y_s = y_s.transpose(1, 2).contiguous().view(B, L_q, self.n_head * self.head_dim)
        y_s = self.proj(y_s)
        y_s = self.resid_drop(y_s)

        # Residual connection and MLP for self-attention
        out_mlp = out + y_s  # Residual connection

        out_mlp_ln = self.ln2(out_mlp)  # Apply LayerNorm to the output of the attention block.
        y_mlp = self.mlp(out_mlp_ln)  # Apply MLP to the output of the attention block.

        out_final = out_mlp + y_mlp  # Apply LayerNorm and MLP
        return out_final  # (B, L_q, d_model)

#############################################
# Modified Beamforming Transformer for Complex Inputs with updated V computations.
#############################################
class BeamformingTransformer(nn.Module):
    def __init__(self, config):
        super(BeamformingTransformer, self).__init__()
        self.config = config
        # Set dimensions
        self.K = config.num_users   # Number of users
        self.N = config.num_tx      # Number of transmit antennas
        
        # Projection layers for tokens.
        # For antenna-level tokens: each token originally has dimension K.
        self.antenna_channel_proj = nn.Linear(self.K, config.d_model)  # For Query tokens
        self.antenna_beam_proj    = nn.Linear(self.K, config.d_model)  # For Key tokens
        
        # For user-level tokens: each token originally has dimension N.
        self.user_channel_proj = nn.Linear(self.N, config.d_model)      # For Query tokens
        self.user_beam_proj    = nn.Linear(self.N, config.d_model)      # For Key tokens
        
        # New projection layers for value tokens derived from the product.
        self.antenna_prod_proj = nn.Linear(self.K, config.d_model)       # For antenna-level V tokens
        self.user_prod_proj    = nn.Linear(self.K, config.d_model)       # For user-level V tokens
        
        # # Positional embeddings.
        # # For antenna-level, tokens from columns (real and imaginary parts).
        # self.pos_emb_ant = nn.Parameter(th.zeros(2 * self.N, config.d_model))
        # # For user-level, tokens from rows (real and imaginary parts).
        # self.pos_emb_user = nn.Parameter(th.zeros(2 * self.K, config.d_model))

        # Positional embeddings. (divided: H, W, P)
        self.pos_emb_ant_H = nn.Parameter(th.zeros(2 * self.N, config.d_model))  # for H_real/H_imag
        self.pos_emb_ant_W = nn.Parameter(th.zeros(2 * self.N, config.d_model))  # for W_real/W_imag
        self.pos_emb_ant_P = nn.Parameter(th.zeros(2 * self.N, config.d_model))  # for P_real/P_imag
        # For user-level tokens: there are also 6N tokens.
        self.pos_emb_user_H = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        self.pos_emb_user_W = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        self.pos_emb_user_P = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        
        # Cross-Attention blocks with multiple layers.
        self.ant_mhca_layers = nn.ModuleList([
            CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head,
                                 attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
            for _ in range(config.n_layers)
        ])

        self.user_mhca_layers = nn.ModuleList([
            CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head,
                                 attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
            for _ in range(config.n_layers)
        ])

        #### generate delta_w and delta_p for antenna and user-level
        self.inter_w_ant = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.inter_p_ant = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.inter_w_user = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.inter_p_user = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])

        # self.cross_attn_ant = CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
        #                                             attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        # self.cross_attn_user = CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
        #                                              attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        
        # Final fusion and output projection MLP.
        total_tokens = 2*self.N + 2*self.K
        self.out_proj = nn.Sequential(
            nn.Linear(total_tokens * config.d_model, 4 * config.d_model),
            nn.LayerNorm(4 * config.d_model),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(4 * config.d_model, 2 * config.d_model),
            nn.LayerNorm(2 * config.d_model),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(2 * config.d_model, config.beam_dim),
        )
        
        # Weight initialization.
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
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, H, W_prev):
        """
        Args:
            H: Complex channel matrix, shape (B, num_users, num_tx) (i.e. (B, K, N)).
            W_prev: Previous beamformer, shape (B, num_users, num_tx) (i.e. (B, K, N)), complex.
        Returns:
            W_next: Vectorized predicted beamformer, shape (B, beam_dim)
        """
        B = H.size(0)
        K = self.K  # Number of users
        N = self.N  # Number of transmit antennas
        
        # ---------------------------
        # Compute the product: Prod = H * (W_prev^T).
        # H and W_prev are complex with shape (B, K, N); W_prev^T has shape (B, N, K), so Prod is (B, K, K).
        # ---------------------------
        Prod = th.bmm(H, W_prev.transpose(-1, -2))  # (B, K, K)
        
        # ---------------------------
        # 1. Antenna-level Cross-Attention
        #    - Query tokens from H (column-wise tokens).
        #    - Key tokens from W_prev (column-wise tokens).
        #    - Value tokens from Prod are obtained column-wise.
        # ---------------------------
        # Create antenna-level tokens.
        H_ant = th.cat([H.real.transpose(1,2), H.imag.transpose(1,2)], dim=1)    # (B, 2*N, K)
        W_ant = th.cat([W_prev.real.transpose(1,2), W_prev.imag.transpose(1,2)], dim=1)  # (B, 2*N, K)
        
        # Project Query and Key tokens.
        H_ant_proj = self.antenna_channel_proj(H_ant)  # (B, 2*N, d_model)
        W_ant_proj = self.antenna_beam_proj(W_ant)       # (B, 2*N, d_model)
        
        # Create Value tokens:
        # Each column of Prod is used as an antenna-level token.
        P_ant = th.cat([Prod.real.transpose(1,2), Prod.imag.transpose(1,2)], dim=1)  # (B, 2*K, K)
        # Project V_ant to the desired dimension.
        P_ant_proj = self.antenna_prod_proj(P_ant)  # (B, 2*K, d_model)
        # Add positional embedding to Value tokens (antenna-level).
        P_ant_proj = P_ant_proj + self.pos_emb_ant_P.unsqueeze(0)
        
        # Add positional embeddings to Query and Key tokens.
        H_ant_proj = H_ant_proj + self.pos_emb_ant_H.unsqueeze(0)
        W_ant_proj = W_ant_proj + self.pos_emb_ant_W.unsqueeze(0)
        
        # Multi-layer Cross-attention blocks for antenna-level
        # x_a = self.cross_attn_ant(H_ant_proj, W_ant_proj, P_ant_proj)  # (B, 2*N, d_model)
        # x_a = H_ant_proj
        # for layer in self.ant_mhca_layers:
        #     x_a = layer(x_a, W_ant_proj, P_ant_proj)  
        # x_a_final = x_a

        for i in range(self.config.n_layers):
            H_ant_proj = self.ant_mhca_layers[i](H_ant_proj, W_ant_proj, P_ant_proj)  # (B, 2*N, d_model)
            delta_w_ant = self.inter_w_ant[i](H_ant_proj)  # (B, 2*N, d_model)
            delta_p_ant = self.inter_p_ant[i](P_ant_proj)  # (B, 2*K, d_model)
            W_ant_proj = W_ant_proj + delta_w_ant  # Update Key tokens
            P_ant_proj = P_ant_proj + delta_p_ant  # Update Value tokens
        x_a_final = H_ant_proj  # Final output for antenna-level

        # ---------------------------
        # 2. User-level Cross-Attention
        #    - Query tokens from H (row-wise tokens).
        #    - Key tokens from W_prev (row-wise tokens).
        #    - Value tokens from Prod are obtained row-wise.
        # ---------------------------
        H_user = th.cat([H.real, H.imag], dim=1)  # (B, 2*K, N)
        W_user = th.cat([W_prev.real, W_prev.imag], dim=1)  # (B, 2*K, N)
        
        # Project Query and Key tokens.
        H_user_proj = self.user_channel_proj(H_user)  # (B, 2*K, d_model)
        W_user_proj = self.user_beam_proj(W_user)       # (B, 2*K, d_model)
        
        # Create Value tokens:
        # Each row of Prod is used as a user-level token.
        P_user = th.cat([Prod.real, Prod.imag], dim=1)  # (B, 2*K, K)
        # Project V_user to the desired dimension.
        P_user_proj = self.user_prod_proj(P_user)  # (B, 2*K, d_model)
        # Add positional embedding to Value tokens (user-level).
        P_user_proj = P_user_proj + self.pos_emb_user_P.unsqueeze(0)
        
        # Add positional embeddings to Query and Key tokens.
        H_user_proj = H_user_proj + self.pos_emb_user_H.unsqueeze(0)
        W_user_proj = W_user_proj + self.pos_emb_user_W.unsqueeze(0)
        
        # Cross-attention for user-level.
        # x_u = self.cross_attn_user(H_user_proj, W_user_proj, P_user_proj)  # (B, 2*K, d_model)
        # x_u = H_user_proj # (B, 2*K, d_model)
        # for layer in self.user_mhca_layers:
        #     x_u = layer(x_u, W_user_proj, P_user_proj)
        # x_u_final = x_u

        for i in range(self.config.n_layers):
            H_user_proj = self.user_mhca_layers[i](H_user_proj, W_user_proj, P_user_proj)
            delta_w_user = self.inter_w_user[i](H_user_proj)  # (B, 2*K, d_model)
            delta_p_user = self.inter_p_user[i](P_user_proj)  # (B, 2*K, d_model)
            W_user_proj = W_user_proj + delta_w_user  # Update Key tokens
            P_user_proj = P_user_proj + delta_p_user  # Update Value tokens
        x_u_final = H_user_proj  # Final output for user-level

        # ---------------------------
        # 3. Fusion and Output Prediction
        # ---------------------------
        # Flatten the outputs and concatenate.
        x_a_flat = x_a_final.view(B, -1)      # (B, (2*N)*d_model)
        x_u_flat = x_u_final.view(B, -1)      # (B, (2*K)*d_model)
        x_fused = th.cat([x_a_flat, x_u_flat], dim=-1)  # (B, (2*N+2*K)*d_model)
        
        # Project fused features to final beamformer vector and normalize.
        W_next = self.out_proj(x_fused)  # (B, beam_dim)
        norm = th.norm(W_next, dim=1, keepdim=True)
        W_next = W_next / norm
        return W_next

#############################################
# 5. Channel dataset: random Gaussian H
#############################################
class ChannelDataset(Dataset):
    def __init__(self, num_samples, num_users, num_tx, P):
        self.num_samples = num_samples
        self.num_users = num_users
        self.num_tx = num_tx
        self.P = P
        ### pre-generate the channel matrix
        self.data = th.randn(num_samples, 2, num_users, num_tx, dtype=th.float32, device=device) * (math.sqrt(2) / 2)  # Shape: (num_samples, 2, num_users, num_tx)
        self.data = self.data * (P ** 0.5)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]  # Shape: (2, num_users, num_tx)


#############################################
# 5. Optimizer configuration
#############################################

def configure_optimizer(model, learning_rate, weight_decay):
    """
    Configure optimizer with selective weight decay for linear and conv2d layers.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(layer_type in name for layer_type in ['linear', 'conv', 'fc', 'weight']):
            if not any(exclude in name for exclude in ['ln', 'norm', 'layernorm', 'emb', 'embedding', 'bias']):
                decay_params.append(param)
                continue
        no_decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = th.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


# def lr_lambda(epoch, config):
#     ## warm start
#     total_steps = config.max_epoch
#     warmup_steps = int(total_steps * 0.00)  
#     ## linearly increase the lr in the warm-up phase
#     if epoch < warmup_steps:
#         return math.sqrt(float(epoch) / float(max(1, warmup_steps)))   
#     ## cosine decay after warm-up
#     progress = float(epoch - warmup_steps) / float(max(1, total_steps - warmup_steps))
#     cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress)) ## 1 -> 0
#     # minimum learning rate: 0.05 * initial_lr
#     return max(0.2, cosine_decay)


def get_mmse_obj(H, W, sigma2=1.0):
    prod = th.bmm(H, W)  # (B, num_users, num_users)
    fro_norm = th.norm(prod, p='fro', dim=(1,2)) ** 2
    trace_real = th.stack([th.trace(prod).real for prod in prod])
    MSE_obj = fro_norm - trace_real
    return MSE_obj


def custom_lr_scheduler(optimizer, config):
    initial_lr = config.learning_rate
    mid_lr = initial_lr * 0.3 ### very intricate here
    final_lr = initial_lr * 0.15

    def lr_lambda(epoch):
        if epoch < config.mmse_epoch:
            # Phase 1: Linear decay from initial_lr to mid_lr during supervised learning
            decay_ratio = epoch / config.mmse_epoch
            target_lr = initial_lr * (1 - decay_ratio) + mid_lr * decay_ratio
            return target_lr / optimizer.param_groups[0]['initial_lr']
        else:
            # Phase 2: Cosine decay from mid_lr to final_lr during unsupervised learning
            progress = float(epoch - config.mmse_epoch) / float(max(1, config.max_epoch - config.mmse_epoch))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1→0
            target_lr = mid_lr * cosine_factor + final_lr * (1 - cosine_factor)
            return target_lr / optimizer.param_groups[0]['initial_lr']
    
    # Store the initial learning rate in the optimizer
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = initial_lr
        
    return th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#############################################
# 6. Training Routine
#############################################
def train_beamforming_transformer(config, pretrained_path: str = None, history_path: str = None):
    """
    Train the beamforming transformer based on the given configuration.
    """
    # dataset = ChannelDataset(num_samples=config.pbar_size * config.batch_size,
    #                          num_users=config.num_users,
    #                          num_tx=config.num_tx,
    #                          P=config.SNR_power)
    # dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # 1) Instantiate the model
    model = BeamformingTransformer(config).to(device)

    ###### optimizer and lr scheduler: scheme 1 (linearly decays lr)
    optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)

    scheduler = custom_lr_scheduler(optimizer, config)

    # scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=config.max_epoch,  
    #     eta_min=config.learning_rate * 0.5 
    # )
    # scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - 0.5 * (epoch / config.max_epoch)) ### linearly decays lr

    # ####### optimizer and lr scheduler: scheme 2 (warm-start + cosine decay)
    # optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)
    # scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_lambda(epoch, config))

    # 2) Optionally load pretrained weights
    if pretrained_path is not None:
        checkpoint = th.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained weights from {pretrained_path}")
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 3) Optionally load pre-stored history
    if history_path is not None and pretrained_path is not None:
        saved_history = th.load(history_path)
        rate_history = saved_history.get('rate_history', [])
        ave_rate_history = saved_history.get('ave_rate_history', [])
        test_rate_history = saved_history.get('test_rate_history', [])
        start_epoch_prev = saved_history.get('start_epoch', 0) 
        print(f"Resuming from epoch {start_epoch_prev}")
    else:
        rate_history = []
        ave_rate_history = []
        test_rate_history = []
        start_epoch_prev = 0
  
    global mmse_rate_printed
    mmse_rate_printed = False
    model.train()
    
    # rate_history = []
    # test_rate_history = []
    # ave_rate_history = []

    # save_dir = f"{config.num_users}_{config.num_tx}_6_5_store"
    # os.makedirs(save_dir, exist_ok=True)
    # best_model_path = os.path.join(save_dir, "best_model.pt")

    best_model_path = "best_model.pt"
    
    ### inherit the previous rate history, including the start_epoch
    for epoch in range(start_epoch_prev, config.max_epoch):

        dataset = ChannelDataset(num_samples=config.pbar_size * config.batch_size,
                             num_users=config.num_users,
                             num_tx=config.num_tx,
                             P=config.SNR_power)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        # # Soft switch learning policy.
        teacher_weight = max(0.0, 1 - (epoch / config.mmse_epoch))  # Linearly decrease from 1 to 0
        # teacher_weight = 0.0  

        epoch_loss = 0
        epoch_rate = 0
        epoch_rate_update = 0
        pbar_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        max_rate = 0
        epoch_grad_norm = 0
        for batch in pbar:
            # Channel input: shape (B, 2, num_users, num_tx)
            H_tensor = batch.to(device)
            # Convert to complex channel matrix: (B, num_users, num_tx)
            H_mat = H_tensor[:, 0, :, :] + 1j * H_tensor[:, 1, :, :]
            
            # Compute initial beamformer using MMSE.
            W0, _ = compute_mmse_beamformer(H_mat, config.num_users, config.num_tx, 
                                            config.SNR_power, config.sigma2, device)
            W0 = W0.transpose(-1, -2).to(device)  # (B, num_tx, num_users)
            vec_w0 = th.cat((th.real(W0).reshape(-1, config.num_tx * config.num_users), 
                             th.imag(W0).reshape(-1, config.num_tx * config.num_users)), dim=-1)
            vec_w0 = vec_w0.reshape(-1, 2 * config.num_tx * config.num_users)
            norm_W0 = th.norm(vec_w0, dim=1, keepdim=True)
            normlized_W0 = vec_w0 / norm_W0
            # Reconstruct complex beamformer: (B, num_tx, num_users)
            W_mat_0 = (normlized_W0[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) +
                       1j * normlized_W0[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
            
            # rate_0 = sum_rate(H_mat, W_mat_0, sigma2=config.sigma2)
            # if not mmse_rate_printed:
            #     mmse_rate_printed = True
            #     print(f"MMSE Rate: {rate_0.item():.4f}")
            
            # Initialize W_prev (transpose to get shape (B, num_users, num_tx)).
            W_prev = W_mat_0.transpose(-1, -2).to(device)
            total_rate = 0
            total_mse_loss = 0
            total_rate_update = 0

            ### save more time by not computing the MMSE beamformer at the later stage
            if teacher_weight > 0.0:
                for t in range(1, config.T + 1):
                    # The model predicts the next beamformer.
                    W_next = model(H_mat, W_prev)  # (B, beam_dim)
                    # Convert W_next to complex beamformer matrix: (B, num_tx, num_users)
                    W_mat = (W_next[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) + 1j * W_next[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
                    W_prev = W_mat.transpose(-1, -2).to(device)  # (B, num_users, num_tx)
                    rate = sum_rate(H_mat, W_mat, sigma2=config.sigma2)
                    total_rate += rate

                    #### consider PGD refinement
                    W_pre_update = W_mat.clone().requires_grad_(True) 
                    for _ in range(config.pgd_steps):
                        loss_inner = sum_rate(H_mat, W_pre_update, sigma2=config.sigma2)  # (B, num_tx, num_users)
                        grad_w = th.autograd.grad(loss_inner, W_pre_update, create_graph=True)[0]  ## maintain second-order gradient
                        W_pre_update = (W_pre_update + config.pgd_step_size * grad_w)  # (B, num_tx, num_users)
                    W_update = W_pre_update
                    rate_update = sum_rate(H_mat, W_update, sigma2=config.sigma2)
                    total_rate_update += rate_update

                    ### supervised MSE loss
                    mse_loss = fun.mse_loss(W_next, normlized_W0)
                    total_mse_loss += mse_loss

                    # ### unsupervised MSE loss
                    # mse_loss = get_mmse_obj(H_mat, W_mat, sigma2=config.sigma2)   
                    # total_mse_loss += mse_loss.mean()            
             
                loss_unsupervised = - total_rate / config.T
                loss_supervised = total_mse_loss / config.T
                loss_update = - total_rate_update / config.T
                loss = (1 - teacher_weight) * (loss_unsupervised+loss_update) + (teacher_weight * 30000) * loss_supervised ## supervised MSE
                # print((1 - teacher_weight) * loss_unsupervised.item(), (teacher_weight * 20000) * loss_supervised.item())
                # bp()
                # loss = (1 - teacher_weight) * loss_unsupervised + (teacher_weight / 50) * loss_supervised ## supervised MSE

            else:
                for t in range(1, config.T + 1):
                    # The model predicts the next beamformer.
                    W_next = model(H_mat, W_prev)  # (B, beam_dim)
                    # Convert W_next to complex beamformer matrix: (B, num_tx, num_users)
                    W_mat = (W_next[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) + 1j * W_next[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
                    W_prev = W_mat.transpose(-1, -2).to(device)  # (B, num_users, num_tx)
                    rate = sum_rate(H_mat, W_mat, sigma2=config.sigma2)
                    total_rate += rate  

                    #### consider PGD refinement
                    W_pre_update = W_mat.clone().requires_grad_(True) 
                    for _ in range(config.pgd_steps):
                        loss_inner = sum_rate(H_mat, W_pre_update, sigma2=config.sigma2)  # (B, num_tx, num_users)
                        grad_w = th.autograd.grad(loss_inner, W_pre_update, create_graph=True)[0]  ## maintain second-order gradient
                        W_pre_update = (W_pre_update + config.pgd_step_size * grad_w)  # (B, num_tx, num_users)
                    W_update = W_pre_update
                    rate_update = sum_rate(H_mat, W_update, sigma2=config.sigma2)
                    total_rate_update += rate_update

                loss = - total_rate - total_rate_update # Unsupervised loss

            optimizer.zero_grad()
            loss.backward()

            with th.no_grad():
                total_norm = th.tensor(0.0, device=device)
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.norm(2)
                        total_norm += param_norm ** 2
                current_grad_norm = th.sqrt(total_norm).item()
                epoch_grad_norm += current_grad_norm

            # ### add gradient clipping
            # th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            ### test gradient values
            th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=200.0)
            # print(f"Gradient norm: {grad_norm:.4f}\n")

            optimizer.step()

            ave_rate = total_rate.item() / config.T
            ave_rate_update = total_rate_update.item() / config.T
            epoch_loss += loss.item()
            epoch_rate += ave_rate
            epoch_rate_update += ave_rate_update
            rate_history.append(th.tensor(ave_rate))
            max_rate = max(max_rate, ave_rate)
            pbar_batches += 1
            # pbar.set_description(f"Epoch {epoch+1}, Avg Rate: {ave_rate:.4f}, Max Rate: {max_rate:.4f}")
            pbar.set_description(f"Epoch {epoch+1}, Avg Rate: {ave_rate:.4f}, Update Rate: {ave_rate_update:.4f}")
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        ave_pbar_rate = epoch_rate / pbar_batches
        ave_pbar_rate_update = epoch_rate_update / pbar_batches
        ave_grad_norm = epoch_grad_norm / pbar_batches
        test_pbar_rate = max_rate
        print(f"Epoch {epoch+1} Average Sum Rate: {ave_pbar_rate:.4f}, Updated Sum Rate: {ave_pbar_rate_update:.4f}, Learning Rate: {current_lr:.2e}, teacher_weight: {teacher_weight:.2f}, Gradient Norm: {ave_grad_norm:.4f}")
        ave_rate_history.append(th.tensor(ave_pbar_rate))
        test_rate_history.append(th.tensor(test_pbar_rate))

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'avg_rate': ave_pbar_rate,
                'max_rate': test_pbar_rate,
                'config': config.__dict__ 
            }

            # checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            # th.save(checkpoint, checkpoint_path)
            # # print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
                
            th.save(checkpoint, best_model_path)
            print(f"Best model saved with rate {ave_pbar_rate:.4f}")

            saved_history = {
                'rate_history': rate_history,
                'ave_rate_history': ave_rate_history,
                'test_rate_history': test_rate_history,
                'start_epoch': epoch + 1,
                'learning_rate': current_lr,
                'config': config.__dict__
            }

            # th.save(saved_history, os.path.join(save_dir, "saved_history.pt"))
            th.save(saved_history, "saved_history.pt")
            print(f"History saved at epoch {epoch + 1}")


    rate_history = th.stack(rate_history)
    ave_rate_history = th.stack(ave_rate_history)
    test_rate_history = th.stack(test_rate_history)
    th.save(rate_history, f"rate_train_history_{config.num_users}_{config.num_tx}_hybrid_attn_multi_layer.pth")
    rate_history = th.load(f"rate_train_history_{config.num_users}_{config.num_tx}_hybrid_attn_multi_layer.pth")
    th.save(ave_rate_history, f"rate_ave_history_{config.num_users}_{config.num_tx}_hybrid_attn_multi_layer.pth")
    ave_rate_history = th.load(f"rate_ave_history_{config.num_users}_{config.num_tx}_hybrid_attn_multi_layer.pth")
    th.save(test_rate_history, f"rate_test_history_{config.num_users}_{config.num_tx}_hybrid_attn_multi_layer.pth")
    test_rate_history = th.load(f"rate_test_history_{config.num_users}_{config.num_tx}_hybrid_attn_multi_layer.pth")

    # Create x-axis values for both plots
    x_rate_history = th.arange(len(rate_history))
    x_test_rate_history = th.arange(len(test_rate_history)) * config.pbar_size  # Scale by pbar_size

    plt.figure(figsize=(10, 6))
    plt.plot(x_rate_history, rate_history, marker='.', linestyle='-', color='b', label='Training Rate')
    plt.plot(x_test_rate_history, ave_rate_history, marker='*', linestyle='-', color='g', label='Average Rate')
    plt.plot(x_test_rate_history, test_rate_history, marker='o', linestyle='-', color='r', label='Testing Rate')
    plt.title(f"Training and Testing Rate when N={config.num_tx} and K={config.num_users}")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Sum Rate")
    plt.grid(True, which="both", ls="--")
    plt.show()

#############################################
# 8. Config class
#############################################
class BeamformerTransformerConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs['d_model']      # Transformer model dimension
        self.beam_dim = kwargs['beam_dim']      # Beamformer vector dimension
        self.n_head = kwargs['n_head']          # Number of attention heads
        self.n_layers = kwargs['n_layers']      # Number of transformer layers
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self.weight_decay = kwargs['weight_decay']
        self.max_epoch = kwargs['max_epoch']
        self.num_users = kwargs['num_users']
        self.num_tx = kwargs['num_tx']
        self.sigma2 = kwargs['sigma2']
        self.T = kwargs['T']
        self.SNR_power = kwargs['SNR_power']
        self.attn_pdrop = kwargs['attn_pdrop']
        self.resid_pdrop = kwargs['resid_pdrop']
        self.mlp_ratio = kwargs['mlp_ratio']
        self.subspace_dim = kwargs['subspace_dim']
        self.pbar_size = kwargs['pbar_size']
        self.mmse_epoch = kwargs['mmse_epoch']
        self.pgd_step_size = kwargs['pgd_step_size']
        self.pgd_steps = kwargs['pgd_steps']

if __name__ == "__main__":

    ## Example configuration where num_users = num_tx = 32, already the best setting.
    num_users = 32
    num_tx = 32  
    beam_dim = 2*num_tx*num_users # Beamformer vector dimension
    d_model = 768 # Transformer single-token dimension
    n_head = 12 # Number of attention heads
    n_layers = 5 # Number of transformer layers
    T = 1 # Number of time steps
    batch_size = 64
    learning_rate = 6e-5
    weight_decay = 0.0
    max_epoch = 2000
    sigma2 = 1.0  
    SNR = 15
    SNR_power = 10 ** (SNR/10) # SNR power in dB
    attn_pdrop = 0.0
    resid_pdrop = 0.0
    mlp_ratio = 4
    subspace_dim = 4
    pbar_size = 500
    mmse_epoch = 200  # Epoch incorporating mmse supervision
    pgd_step_size = 1.75e-2  # Step size for PGD refinement
    pgd_steps = 5  # Number of PGD steps for refinement

    # # Example configuration where num_users = num_tx = 20.
    # num_users = 28
    # num_tx = 28
    # d_model = 640 # Transformer single-token dimension
    # beam_dim = 2*num_tx*num_users # Beamformer vector dimension
    # n_head = 10 # Number of attention heads
    # n_layers = 5 # Number of transformer layers
    # T = 1 # Number of time steps
    # batch_size = 64
    # learning_rate = 2e-4
    # weight_decay = 0.0
    # max_epoch = 2000
    # sigma2 = 1.0  
    # SNR = 15
    # SNR_power = 10 ** (SNR/10) # SNR power in dB
    # attn_pdrop = 0.0
    # resid_pdrop = 0.0
    # mlp_ratio = 4
    # subspace_dim = 4
    # pbar_size = 500
    # mmse_epoch = 200  # Epoch incorporating mmse supervision
    
#    # Example configuration where num_users = num_tx = 20.
#     num_users = 20
#     num_tx = 20
#     d_model = 640 # Transformer single-token dimension
#     beam_dim = 2*num_tx*num_users # Beamformer vector dimension
#     n_head = 10 # Number of attention heads
#     n_layers = 5 # Number of transformer layers
#     T = 1 # Number of time steps
#     batch_size = 64
#     learning_rate = 2e-4
#     weight_decay = 0.0
#     max_epoch = 2000
#     sigma2 = 1.0  
#     SNR = 15
#     SNR_power = 10 ** (SNR/10) # SNR power in dB
#     attn_pdrop = 0.0
#     resid_pdrop = 0.0
#     mlp_ratio = 4
#     subspace_dim = 4
#     pbar_size = 500
#     mmse_epoch = 200  # Epoch incorporating mmse supervision

    # Create configuration object.
    config = BeamformerTransformerConfig(
        d_model=d_model,
        beam_dim=beam_dim,
        n_head=n_head,
        n_layers=n_layers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epoch=max_epoch,
        num_users=num_users,
        num_tx=num_tx,
        sigma2=sigma2,
        T=T,
        SNR_power=SNR_power,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        mlp_ratio=mlp_ratio,
        subspace_dim=subspace_dim,
        pbar_size=pbar_size,
        mmse_epoch=mmse_epoch,
        pgd_step_size=pgd_step_size,
        pgd_steps=pgd_steps
    )

    model_path_stage_1 = "best_model.pt"
    history_path_stage_1 = "saved_history.pt"

    ### Train the beamforming transformer.
    train_beamforming_transformer(config) ## first time training
    # train_beamforming_transformer(config, pretrained_path=model_path_stage_1, history_path=history_path_stage_1) ## training with pretrained model/historical data