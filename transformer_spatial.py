import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from baseline_mmse import compute_mmse_beamformer
from pdb import set_trace as bp


#############################################
# 1. System performance: sum rate for multi-user MISO
#############################################
def sum_rate(H, W, sigma2=1.0):
    prod = th.bmm(H, W)  # (B, num_users, num_users)
    signal_power = th.abs(th.diagonal(prod, dim1=-2, dim2=-1))**2
    interference = th.sum(th.abs(prod)**2, dim=-1) - signal_power
    N = sigma2
    SINR = signal_power / (interference + N) # (B, num_users)
    reward = th.log2(1 + SINR).sum(dim=-1).mean()
    return reward


#############################################
# Cross-Attention Block (for both Antenna-level and User-level)
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
        
        # LayerNorm and MLP sublayer (similar to standard Transformer blocks)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
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

        # Pre-norm on the inputs for residual connection
        query_ln = self.ln1(query)
        kv_ln = self.ln1(kv)
        
        # Compute Q, K, V and reshape into multi-head format
        Q = self.q_proj(query_ln).view(B, L_q, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, L_q, head_dim)
        K = self.k_proj(kv_ln).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)       # (B, n_head, L_k, head_dim)
        V = self.v_proj(kv_ln).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)       # (B, n_head, L_k, head_dim)
        
        # Compute attention scores (scaled dot-product attention)
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, L_q, L_k)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Compute weighted sum of V
        y = att @ V  # (B, n_head, L_q, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, L_q, self.n_head * self.head_dim)  # (B, L_q, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
        
        # Residual connection: add the attention output to the original query
        out = query + y
        # Further process with an MLP (with its own residual connection)
        out = out + self.mlp(self.ln2(out))
        return out

#############################################
# Beamforming Transformer (Modified for Complex Inputs)
#############################################
class BeamformingTransformer(nn.Module):
    def __init__(self, config):
        super(BeamformingTransformer, self).__init__()
        self.config = config
        # Set dimensions
        self.K = config.num_users   # Number of users
        self.N = config.num_tx      # Number of transmit antennas
        
        # Define projection layers for tokens to map to d_model dimensions.
        # For antenna-level tokens: each token has dimension K (from one column) before splitting,
        # but after splitting real/imaginary, each token is still of length K.
        self.antenna_channel_proj = nn.Linear(self.K, config.d_model)
        self.antenna_beam_proj    = nn.Linear(self.K, config.d_model)
        # For user-level tokens: each token has dimension N (from one row)
        self.user_channel_proj = nn.Linear(self.N, config.d_model)
        self.user_beam_proj    = nn.Linear(self.N, config.d_model)
        
        # Define positional embeddings for antenna-level and user-level tokens.
        # For antenna-level, there are now 2*N tokens (real and imaginary parts for each column).
        self.pos_emb_ant = nn.Parameter(th.zeros(2 * self.N, config.d_model))
        # For user-level, there are 2*K tokens.
        self.pos_emb_user = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        
        # Define cross-attention blocks for antenna-level and user-level
        self.cross_attn_ant = CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
                                                    attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        self.cross_attn_user = CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
                                                     attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        
        # Define final fusion and output projection MLP.
        # We flatten the antenna-level (2*N tokens) and user-level (2*K tokens) features and then concatenate.
        # The final output dimension is beam_dim (typically 2 * num_tx * num_users).
        self.out_proj = nn.Sequential(
            nn.Linear((2 * self.N + 2 * self.K) * config.d_model, config.d_model),
            nn.ReLU(),  # Alternatively, you may try LeakyReLU here if needed.
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
            H: Constant channel matrix, shape (B, num_users, num_tx) i.e. (B, K, N), complex-valued.
            W_prev: Previous beamformer, shape (B, num_users, num_tx) i.e. (B, K, N), complex-valued.
        Returns:
            W_next: Vectorized predicted beamformer, shape (B, beam_dim)
        """
        B = H.size(0)
        K = self.K  # Number of users
        N = self.N  # Number of transmit antennas
        
        # ---------------------------
        # 1. Antenna-level Cross-Attention
        # ---------------------------
        # For antenna-level, tokens are constructed from the columns of H.
        # Each column (of shape (B, K)) is split into its real and imaginary parts.
        # This creates 2*N tokens of shape (B, K) each.
        H_ant = th.cat([H.real.transpose(1,2), H.imag.transpose(1,2)], dim=1)   # (B, 2*N, K)
        W_ant = th.cat([W_prev.real.transpose(1,2), W_prev.imag.transpose(1,2)], dim=1)  # (B, 2*N, K)
        
        # Project tokens to d_model space
        H_ant_proj = self.antenna_channel_proj(H_ant)  # (B, 2*N, d_model)
        W_ant_proj = self.antenna_beam_proj(W_ant)       # (B, 2*N, d_model)
        
        # Add antenna-level positional embeddings (each token is indexed 0 ... 2*N-1)
        H_ant_proj = H_ant_proj + self.pos_emb_ant.unsqueeze(0)  # (B, 2*N, d_model)
        W_ant_proj = W_ant_proj + self.pos_emb_ant.unsqueeze(0)  # (B, 2*N, d_model)
        
        # Apply cross-attention: use H_ant_proj as Query, W_ant_proj as Key and Value
        x_a = self.cross_attn_ant(H_ant_proj, W_ant_proj)  # (B, 2*N, d_model)
        
        # ---------------------------
        # 2. User-level Cross-Attention
        # ---------------------------
        # For user-level, tokens are constructed from the rows of H.
        # Each row (of shape (B, N)) is split into its real and imaginary parts,
        # resulting in 2*K tokens of shape (B, N) each.
        H_user = th.cat([H.real, H.imag], dim=1)  # (B, 2*K, N)
        W_user = th.cat([W_prev.real, W_prev.imag], dim=1)  # (B, 2*K, N)
        
        # Project tokens to d_model space
        H_user_proj = self.user_channel_proj(H_user)  # (B, 2*K, d_model)
        W_user_proj = self.user_beam_proj(W_user)       # (B, 2*K, d_model)
        
        # Add user-level positional embeddings (each token is indexed 0 ... 2*K-1)
        H_user_proj = H_user_proj + self.pos_emb_user.unsqueeze(0)  # (B, 2*K, d_model)
        W_user_proj = W_user_proj + self.pos_emb_user.unsqueeze(0)  # (B, 2*K, d_model)
        
        # Apply cross-attention: use H_user_proj as Query, W_user_proj as Key and Value
        x_u = self.cross_attn_user(H_user_proj, W_user_proj)  # (B, 2*K, d_model)
        
        # ---------------------------
        # 3. Fusion and Output Prediction
        # ---------------------------
        # Flatten the antenna-level and user-level outputs and concatenate them
        x_a_flat = x_a.view(B, -1)  # (B, (2*N) * d_model)
        x_u_flat = x_u.view(B, -1)  # (B, (2*K) * d_model)
        x_fused = th.cat([x_a_flat, x_u_flat], dim=-1)  # (B, (2*N+2*K) * d_model)
        
        # Project the fused features to the final beamformer vector (beam_dim = 2 * num_tx * num_users)
        W_next = self.out_proj(x_fused)  # (B, beam_dim)
        # Normalize the predicted beamformer
        norm = th.norm(W_next, dim=1, keepdim=True)
        W_next = W_next / norm
        return W_next

    

#############################################
# 5. Channel dataset: random Gaussian H
#############################################
# basis_vectors, _ = th.linalg.qr(th.rand(2 * 16 * 16, 2 * 16 * 16, dtype=th.float))
def generate_basis_vectors(num_users, num_tx):
    vector_dim = 2 * num_users * num_tx  # Real + imaginary components
    basis_vectors, _ = th.linalg.qr(th.rand(vector_dim, vector_dim, dtype=th.float))
    return basis_vectors

class ChannelDataset(Dataset):
    def __init__(self, num_samples, num_users, num_tx, P, subspace_dim):
        self.num_samples = num_samples
        self.num_users = num_users
        self.num_tx = num_tx
        self.P = P
        self.subspace_dim = subspace_dim
        self.basis_vectors = generate_basis_vectors(num_users, num_tx)
        
    def __len__(self):
        return self.num_samples
    
    #### generate a batch of channels based on a certain subspace dimension
    
    def __getitem__(self, idx):
        # coordinates = th.randn(self.subspace_dim, 1)
        # basis_vectors_subset = self.basis_vectors[:self.subspace_dim].T
        # vec_channel = th.matmul(basis_vectors_subset, coordinates).reshape(2 * self.num_users * self.num_tx)
        # H_real = vec_channel[:self.num_tx * self.num_users].reshape(self.num_users, self.num_tx) ### (num_users, num_tx)
        # H_imag = vec_channel[self.num_tx * self.num_users:].reshape(self.num_users, self.num_tx) ### (num_users, num_tx)
        # H_complex = H_real + 1j * H_imag
        # norm_H_complex = th.sum(th.abs(H_complex)**2)
        # SNR_power = self.num_users*self.num_tx*self.P
        # H_real = H_real * th.sqrt(SNR_power / norm_H_complex) ### (num_users, num_tx)
        # H_imag = H_imag * th.sqrt(SNR_power / norm_H_complex) ### (num_users, num_tx)
        # H_combined = th.stack([H_real, H_imag], dim=0) ### Shape: (2, num_users, num_tx)
        # # H_combined = (vec_channel[:,:self.num_tx * self.num_users].reshape(-1, self.num_users, self.num_tx) + 1j * vec_channel[:,self.num_tx * self.num_users:].reshape(-1, self.num_users, self.num_tx)) ### (num_samples, num_users, num_tx)

        # Generate complex channel matrix (real + imaginary parts)
        H_real = np.random.randn(self.num_users, self.num_tx).astype(np.float32)
        H_imag = np.random.randn(self.num_users, self.num_tx).astype(np.float32)
        # Stack real and imaginary parts
        H_combined = np.stack([H_real, H_imag], axis=0)*((self.P/2)**0.5)  ### Shape: (2, num_users, num_tx)       
 
        return th.tensor(H_combined) ### transform to tensor


#############################################
# 5. Optimizer configuration
#############################################
def configure_optimizer(model, learning_rate, weight_decay):
    """
    Configure optimizer with selective weight decay - only for linear and Conv2d layers.
        
    Returns:
        Configured AdamW optimizer
    """
    # Separate parameters into those that should have weight decay and those that shouldn't
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        # Check if parameter requires grad first to avoid unnecessary computation
        if not param.requires_grad:
            continue
            
        # Apply weight decay to linear and conv2d layers only
        if any(layer_type in name for layer_type in ['linear', 'conv', 'fc', 'weight']):
            # Ensure we don't include layer norm or embedding weights that happen to have 'weight' in name
            if not any(exclude_type in name for exclude_type in ['ln', 'norm', 'layernorm', 'emb', 'embedding', 'bias']):
                decay_params.append(param)
                continue
                
        # All other parameters get no weight decay
        no_decay_params.append(param)
    
    # # Log parameter counts for debugging
    # print(f"Parameters with weight decay: {len(decay_params)}")
    # print(f"Parameters without weight decay: {len(no_decay_params)}")
    
    # Create parameter groups
    optimizer_grouped_parameters = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = th.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


#############################################
# 6. Training Routine
#############################################
def train_beamforming_transformer(config):
    """
    Train the beamforming transformer based on the given configuration.
    Args:
        config: An instance of BeamformerTransformerConfig with model and training parameters.
    """
    dataset = ChannelDataset(num_samples=1000 * config.batch_size,
                             num_users=config.num_users,
                             num_tx=config.num_tx,
                             P=config.SNR_power,
                             subspace_dim=config.subspace_dim)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = BeamformingTransformer(config).to(device)
    optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)

    # Create the learning rate scheduler that linearly decays LR from 1e-4 to 1e-5 over max_epoch epochs
    scheduler = th.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - 0.9 * (epoch / config.max_epoch)
    )
    
    global mmse_rate_printed
    mmse_rate_printed = False
    model.train()
    
    for epoch in range(config.max_epoch):
        epoch_loss = 0
        epoch_rate = 0
        pbar_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            # Channel input: shape (B, 2, num_users, num_tx)
            H_tensor = batch.to(device)
            batch_size = H_tensor.size(0)
            # Convert H to complex format: (B, num_users, num_tx)
            H_mat = H_tensor[:, 0, :, :] + 1j * H_tensor[:, 1, :, :]
            
            # Compute initial beamformer using MMSE (for initialization)
            W0, _ = compute_mmse_beamformer(H_mat, config.num_users, config.num_tx,
                                            config.SNR_power, config.sigma2, device)
            W0 = W0.transpose(-1, -2).to(device) # (B, num_tx, num_users)
            # Vectorize beamformer: separate real and imaginary parts
            vec_w0 = th.cat((th.real(W0).reshape(-1, config.num_tx * config.num_users),
                             th.imag(W0).reshape(-1, config.num_tx * config.num_users)), dim=-1)
            vec_w0 = vec_w0.reshape(-1, 2 * config.num_tx * config.num_users)
            norm_W0 = th.norm(vec_w0, dim=1, keepdim=True)
            normlized_W0 = vec_w0 / norm_W0
            # Reconstruct complex beamformer from normalized vector
            W_mat_0 = (normlized_W0[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) +
                       1j * normlized_W0[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
            rate_0 = sum_rate(H_mat, W_mat_0, sigma2=config.sigma2)
            if not mmse_rate_printed:
                mmse_rate_printed = True
                print(f"MMSE Rate: {rate_0.item():.4f}")
            
            # Use iterative process for T steps (learning-to-optimize loop)
            # W_prev = normlized_W0  # Initial beamformer in vectorized form
            W_prev = W_mat_0.transpose(-1, -2).to(device)  # (B, num_users, num_tx)
            total_rate = 0
            max_rate = 0
            for t in range(1, config.T + 1):
                # Our model takes the current channel H and the previous beamformer W_prev to predict the next beamformer.
                W_next = model(H_mat, W_prev) # (B, beam_dim)
                
                # Convert W_next to complex beamformer matrix shape (B, num_tx, num_users)
                W_mat = (W_next[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) +
                         1j * W_next[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
                W_prev = W_mat.transpose(-1, -2).to(device)  # (B, num_users, num_tx)
                rate = sum_rate(H_mat, W_mat, sigma2=config.sigma2)
                total_rate += rate
                max_rate = max(max_rate, rate.item())
            
            loss = - total_rate / config.T
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_rate += total_rate.item() / config.T
            pbar_batches += 1
            pbar.set_description(f"Epoch {epoch+1}, Avg Rate: {(total_rate.item() / config.T):.4f}, Max Rate: {max_rate:.4f}")
        
        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()
        
        # Optionally, print out the current learning rate for debugging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} completed. Learning Rate: {current_lr:.2e}")
        print(f"Epoch {epoch+1} Average Sum Rate: {epoch_rate/pbar_batches:.4f}")
        # print(f"Epoch {epoch+1} Average Loss: {epoch_loss/pbar_batches:.4f}, Average Sum Rate: {epoch_rate/pbar_batches:.4f}")


#############################################
# 8. Config class
#############################################
class BeamformerTransformerConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs['d_model']  # Transformer model dimension
        self.beam_dim = kwargs['beam_dim']  # Beamformer dimension
        self.n_head = kwargs['n_head'] # Number of attention heads
        self.n_layers = kwargs['n_layers'] # Number of transformer layers
        self.max_seq_length = kwargs['max_seq_length'] # Maximum sequence length
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


if __name__ == "__main__":
    # Set all parameters in the main function
    num_users = 16
    num_tx = 16
    d_model = 384 # Transformer single-token dimension
    beam_dim = 2*num_tx*num_users # Beamformer vector dimension
    n_head = 8 # Number of attention heads
    n_layers = 6 # Number of transformer layers
    T = 15 # Number of time steps
    max_seq_length = T+1 # Maximum sequence length (episodes of L2O)
    batch_size = 256 
    learning_rate = 5e-5
    weight_decay = 0.1
    max_epoch = 50
    sigma2 = 1.0  
    SNR = 15
    SNR_power = 10 ** (SNR/10) # SNR power in dB
    attn_pdrop = 0.05
    resid_pdrop = 0.05
    mlp_ratio = 4
    subspace_dim = 4
    
    # Create config object with parameters
    config = BeamformerTransformerConfig(
        d_model=d_model,
        beam_dim=beam_dim,
        n_head=n_head,
        n_layers=n_layers,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epoch=max_epoch,
        num_users=num_users,
        num_tx=num_tx,
        sigma2=sigma2,
        T=T,
        SNR_power = SNR_power,
        attn_pdrop=attn_pdrop,
        resid_pdrop = resid_pdrop,
        mlp_ratio = mlp_ratio,
        subspace_dim = subspace_dim
    )
    
    # Train the model with the given config
    train_beamforming_transformer(config)

### consider learning rate scheduler
### consider using the hybrid-supervision training strategy
### consider curriculum learning strategy
### consider different transformer block number and head number
### consider increasing the episode length T (10-15)
### consider noise injection
### consider the learning rate warm-up at the early stage
### consider random initialization of the beamformer