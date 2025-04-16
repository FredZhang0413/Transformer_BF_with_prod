import math
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from baseline_mmse import compute_mmse_beamformer
from pdb import set_trace as bp
import torch.nn.functional as fun
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Define MSE loss function.
mse_loss_fn = nn.MSELoss()

#############################################
# 1. System performance: sum rate for multi-user MISO.
#############################################
def sum_rate(H, W, sigma2=1.0):
    """
    Calculate the sum rate given channel H and beamformer W.
    Args:
        H: Complex channel matrix (B, num_users, num_tx).
        W: Beamforming matrix (B, num_tx, num_users) in complex form.
        sigma2: Noise variance.
    Returns:
        Average sum rate (scalar).
    """
    prod = th.bmm(H, W)  # (B, num_users, num_users)
    signal_power = th.abs(th.diagonal(prod, dim1=-2, dim2=-1))**2
    interference = th.sum(th.abs(prod)**2, dim=-1) - signal_power
    N = sigma2
    SINR = signal_power / (interference + N)
    reward = th.log2(1 + SINR).sum(dim=-1).mean()
    return reward

#############################################
# 2. Cross-Attention Block (unchanged from before).
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
        
        # LayerNorm for each sublayer.
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # MLP block with an extra layer.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(4 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
    
    def forward(self, query, key, value):
        """
        Performs cross-attention with separate Query, Key, and Value tokens.
        Args:
            query: Tensor of shape (B, L_q, d_model).
            key:   Tensor of shape (B, L_k, d_model).
            value: Tensor of shape (B, L_v, d_model) (Assumes L_k == L_v).
        Returns:
            Output tensor of shape (B, L_q, d_model).
        """
        B, L_q, _ = query.size()
        B, L_k, _ = key.size()
        
        # Apply LayerNorm.
        q_ln = self.ln1(query)
        k_ln = self.ln1(key)
        v_ln = self.ln1(value)
        
        # Linear projections and reshape for multi-head attention.
        Q = self.q_proj(q_ln).view(B, L_q, self.n_head, self.head_dim).transpose(1, 2)
        K = self.k_proj(k_ln).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)
        V = self.v_proj(v_ln).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention.
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Compute weighted sum.
        y = att @ V
        y = y.transpose(1, 2).contiguous().view(B, L_q, self.n_head * self.head_dim)
        y = self.proj(y)
        y = self.resid_drop(y)
        
        # Residual connection and MLP.
        out = query + y
        out = out + self.mlp(self.ln2(out))
        return out

#############################################
# 3. TransformerStep: One untied L2O step.
#############################################
class TransformerStep(nn.Module):
    def __init__(self, config, step_idx):
        """
        Represents one untied transformer optimizer step.
        Args:
            config: Configuration object.
            step_idx: Step index (starting from 1). Determines the history length.
        """
        super(TransformerStep, self).__init__()
        self.step_idx = step_idx
        self.t = step_idx   # Number of historical beamformers used in the step.
        self.N = config.num_tx
        self.K = config.num_users
        self.d_model = config.d_model
        
        # Antenna-level branch (each history yields 2*N tokens after real-imag separation).
        self.antenna_channel_proj = nn.Linear(self.K, config.d_model)
        self.antenna_beam_proj = nn.Linear(self.K, config.d_model)
        self.antenna_prod_proj = nn.Linear(self.K, config.d_model)
        # Positional embedding for antenna branch: token count = t * 2*N.
        self.pos_emb_ant = nn.Parameter(th.zeros(self.t * 2 * self.N, config.d_model))
        # Cross-attention block.
        self.cross_attn_ant = CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
                                                    attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        
        # User-level branch (each history yields 2*K tokens).
        self.user_channel_proj = nn.Linear(self.N, config.d_model)
        self.user_beam_proj = nn.Linear(self.N, config.d_model)
        self.user_prod_proj = nn.Linear(self.K, config.d_model)
        # Positional embedding for user branch: token count = t * 2*K.
        self.pos_emb_user = nn.Parameter(th.zeros(self.t * 2 * self.K, config.d_model))
        # Cross-attention block.
        self.cross_attn_user = CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
                                                     attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        
        # Fusion MLP: fuse flattened antenna and user branch features.
        fused_input_dim = self.t * (2 * self.N + 2 * self.K) * config.d_model
        self.out_proj = nn.Sequential(
            nn.Linear(fused_input_dim, config.d_model),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(config.d_model, config.beam_dim)
        )
        
        # Initialize weights.
        self.apply(self._init_weights)
        
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
    
    def forward(self, H, W_history, prod_history):
        """
        Perform one L2O step using all historical beamformers and products.
        Args:
            H: Complex channel matrix (B, K, N).
            W_history: List of historical beamformers (each of shape (B, num_users, num_tx)).
            prod_history: List of product matrices (each of shape (B, K, K)).
        Returns:
            W_next: Next predicted beamformer (B, beam_dim) as a vector.
        """
        B = H.size(0)
        t = self.t  # Current history length; assert len(W_history)==t.
        assert len(W_history) == t and len(prod_history) == t, "History length mismatch."
        
        # ------------------------------
        # Antenna-level branch tokens.
        # For each history, generate:
        #   H tokens: (B, 2*N, K),
        #   Beam tokens: (B, 2*N, K),
        #   Product tokens: (B, 2*K, K).
        # ------------------------------
        H_ant_list = []
        W_ant_list = []
        V_ant_list = []
        for i in range(t):
            # Channel tokens from H: (B, 2*N, K)
            H_ant = th.cat([H.real.transpose(1,2), H.imag.transpose(1,2)], dim=1)
            # Beamformer tokens from history: (B, 2*N, K)
            W_i = W_history[i]
            W_ant = th.cat([W_i.real.transpose(1,2), W_i.imag.transpose(1,2)], dim=1)
            # Product tokens from history: (B, 2*K, K)
            prod_i = prod_history[i]
            V_ant = th.cat([prod_i.real.transpose(1,2), prod_i.imag.transpose(1,2)], dim=1)
            
            H_ant_list.append(H_ant)
            W_ant_list.append(W_ant)
            V_ant_list.append(V_ant)
        
        # Concatenate along token dimension.
        H_ant_all = th.cat(H_ant_list, dim=1)   # Shape: (B, t*2*N, K)
        W_ant_all = th.cat(W_ant_list, dim=1)   # Shape: (B, t*2*N, K)
        V_ant_all = th.cat(V_ant_list, dim=1)     # Shape: (B, t*2*K, K)
        
        # Project tokens.
        H_ant_proj = self.antenna_channel_proj(H_ant_all)  # (B, t*2*N, d_model)
        W_ant_proj = self.antenna_beam_proj(W_ant_all)       # (B, t*2*N, d_model)
        V_ant_proj = self.antenna_prod_proj(V_ant_all)       # (B, t*2*K, d_model)
        
        # Add positional embeddings (shape: (t*2*N, d_model)).
        H_ant_proj = H_ant_proj + self.pos_emb_ant.unsqueeze(0)
        W_ant_proj = W_ant_proj + self.pos_emb_ant.unsqueeze(0)
        V_ant_proj = V_ant_proj + self.pos_emb_ant.unsqueeze(0)
        
        # Antenna-level cross-attention.
        x_a = self.cross_attn_ant(H_ant_proj, W_ant_proj, V_ant_proj)  # (B, t*2*N, d_model)
        
        # ------------------------------
        # User-level branch tokens.
        # For each history, generate:
        #   H tokens: (B, 2*K, N),
        #   Beam tokens: (B, 2*K, N),
        #   Product tokens: (B, 2*K, K).
        # ------------------------------
        H_user_list = []
        W_user_list = []
        V_user_list = []
        for i in range(t):
            H_user = th.cat([H.real, H.imag], dim=1)  # (B, 2*K, N)
            W_i = W_history[i]
            W_user = th.cat([W_i.real, W_i.imag], dim=1)  # (B, 2*K, N)
            prod_i = prod_history[i]
            V_user = th.cat([prod_i.real, prod_i.imag], dim=1)  # (B, 2*K, K)
            
            H_user_list.append(H_user)
            W_user_list.append(W_user)
            V_user_list.append(V_user)
            
        H_user_all = th.cat(H_user_list, dim=1)  # (B, t*2*K, N)
        W_user_all = th.cat(W_user_list, dim=1)    # (B, t*2*K, N)
        V_user_all = th.cat(V_user_list, dim=1)    # (B, t*2*K, K)
        
        # Project tokens.
        H_user_proj = self.user_channel_proj(H_user_all)  # (B, t*2*K, d_model)
        W_user_proj = self.user_beam_proj(W_user_all)       # (B, t*2*K, d_model)
        V_user_proj = self.user_prod_proj(V_user_all)       # (B, t*2*K, d_model)
        
        # Add positional embeddings.
        H_user_proj = H_user_proj + self.pos_emb_user.unsqueeze(0)
        W_user_proj = W_user_proj + self.pos_emb_user.unsqueeze(0)
        V_user_proj = V_user_proj + self.pos_emb_user.unsqueeze(0)
        
        # User-level cross-attention.
        x_u = self.cross_attn_user(H_user_proj, W_user_proj, V_user_proj)  # (B, t*2*K, d_model)
        
        # ------------------------------
        # Fusion: Flatten and concatenate both branches.
        # ------------------------------
        x_a_flat = x_a.view(B, -1)  # (B, t*2*N*d_model)
        x_u_flat = x_u.view(B, -1)  # (B, t*2*K*d_model)
        x_fused = th.cat([x_a_flat, x_u_flat], dim=-1)  # (B, t*(2*N+2*K)*d_model)
        
        # Final output projection and normalization.
        W_next = self.out_proj(x_fused)  # (B, beam_dim)
        norm = th.norm(W_next, dim=1, keepdim=True)
        W_next = W_next / norm
        return W_next

#############################################
# 4. Beamforming Transformer for L2O with untied steps.
#############################################
class BeamformingTransformerL2O(nn.Module):
    def __init__(self, config):
        """
        L2O transformer with untied parameters over T steps.
        Args:
            config: Configuration object (must include T).
        """
        super(BeamformingTransformerL2O, self).__init__()
        self.config = config
        self.T = config.T  # Number of optimization steps
        # Create a list of untied transformer steps.
        self.steps = nn.ModuleList([TransformerStep(config, step_idx=i+1) for i in range(self.T)])
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters in L2O model: {total_params:,}")
    
    def forward(self, H, W0):
        """
        Unroll the L2O optimization for T steps.
        Args:
            H: Complex channel matrix (B, num_users, num_tx), where num_users = K.
            W0: Initial beamformer (from MMSE) in shape (B, num_users, num_tx).
        Returns:
            A list of predicted beamformers (each of shape (B, beam_dim)) for each step.
        """
        B = H.size(0)
        # The initial beamformer should have shape (B, num_users, num_tx).
        W0_correct = W0  # Already in shape (B, num_users, num_tx).
        # Initialize history lists.
        W_history = []
        prod_history = []
        # Append the initial beamformer.
        W_history.append(W0_correct)
        # Compute the initial product: prod0 = H * (W0_correct)^T.
        prod0 = th.bmm(H, W0_correct.transpose(-1, -2))
        prod_history.append(prod0)
        
        # To store outputs from each L2O step.
        W_list = []
        # Unroll the optimization for T steps.
        for step in self.steps:
            # Each step uses all historical beamformers and product matrices.
            W_next = step(H, W_history, prod_history)
            # Append current prediction to history.
            W_history.append(W_next)
            prod_next = th.bmm(H, W_next.transpose(-1, -2))
            prod_history.append(prod_next)
            W_list.append(W_next)
        # Return the list of beamformer predictions from every step.
        return W_list

#############################################
# 5. Channel dataset: random Gaussian H.
#############################################
def generate_basis_vectors(num_users, num_tx):
    vector_dim = 2 * num_users * num_tx  # For real + imaginary components.
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
    
    def __getitem__(self, idx):
        # Generate a random Gaussian complex channel matrix.
        scale = math.sqrt(2) / 2
        H_real = th.randn(self.num_users, self.num_tx) * scale
        H_imag = th.randn(self.num_users, self.num_tx) * scale
        H_combined = th.stack([H_real, H_imag], dim=0)  # (2, num_users, num_tx)
        H_combined = H_combined * (self.P ** 0.5)
        return th.tensor(H_combined)

#############################################
# 6. Optimizer configuration.
#############################################
def configure_optimizer(model, learning_rate, weight_decay):
    """
    Configure optimizer with selective weight decay for layers.
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

#############################################
# 7. Training Routine.
#############################################
def train_beamforming_transformer(config):
    """
    Train the beamforming transformer using the L2O framework.
    In this version, the L2O model unrolls T steps and returns a list of predicted beamformers.
    For each step, the sum rate is computed and accumulated.
    """
    dataset = ChannelDataset(num_samples=config.pbar_size * config.batch_size,
                             num_users=config.num_users,
                             num_tx=config.num_tx,
                             P=config.SNR_power,
                             subspace_dim=config.subspace_dim)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # Instantiate the L2O model.
    model = BeamformingTransformerL2O(config).to(device)
    optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)

    # Learning rate scheduler that linearly decays.
    scheduler = th.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - 0.0 * (epoch / config.max_epoch)
    )
    
    global mmse_rate_printed
    mmse_rate_printed = False
    model.train()
    teacher_weight = 0  # Switch based on epoch.
    rate_history = []
    test_rate_history = []
    ave_rate_history = []
    
    for epoch in range(config.max_epoch):
        # Hard switch learning policy.
        if epoch < 3:
            teacher_weight = 1
        else:
            teacher_weight = 0.0

        epoch_loss = 0
        epoch_rate = 0
        pbar_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        max_rate = 0
        for batch in pbar:
            # Each batch: Channel input shape (B, 2, num_users, num_tx)
            H_tensor = batch.to(device)
            batch_size = H_tensor.size(0)
            # Convert tensor to complex channel matrix: (B, num_users, num_tx)
            H_mat = H_tensor[:, 0, :, :] + 1j * H_tensor[:, 1, :, :]
            
            # Compute initial MMSE beamformer.
            W0, _ = compute_mmse_beamformer(H_mat, config.num_users, config.num_tx, 
                                            config.SNR_power, config.sigma2, device)
            # Transpose to get shape (B, num_tx, num_users) and then convert to (B, num_users, num_tx).
            W0 = W0.transpose(-1, -2).to(device)
            vec_w0 = th.cat((th.real(W0).reshape(-1, config.num_tx * config.num_users), 
                             th.imag(W0).reshape(-1, config.num_tx * config.num_users)), dim=-1)
            vec_w0 = vec_w0.reshape(-1, 2 * config.num_tx * config.num_users)
            norm_W0 = th.norm(vec_w0, dim=1, keepdim=True)
            normlized_W0 = vec_w0 / norm_W0
            # Reconstruct complex beamformer: (B, num_tx, num_users)
            W_mat_0 = (normlized_W0[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) +
                       1j * normlized_W0[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
            rate_0 = sum_rate(H_mat, W_mat_0, sigma2=config.sigma2)
            if not mmse_rate_printed:
                mmse_rate_printed = True
                print(f"MMSE Rate: {rate_0.item():.4f}")
            
            # The initial beamformer for L2O is in shape (B, num_users, num_tx).
            W0_correct = W_mat_0.transpose(-1, -2).to(device)
            # Call the L2O model to get predictions from all T steps.
            W_list = model(H_mat, W0_correct)  # W_list: list of length config.T, each element is (B, beam_dim)
            
            total_rate = 0
            total_mse_loss = 0
            # Loop over each step's prediction.
            for W_next in W_list:
                # Convert predicted beamformer from vectorized form to complex matrix form: (B, num_tx, num_users)
                W_mat = (W_next[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) +
                         1j * W_next[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
                # Compute sum rate for the current step.
                rate = sum_rate(H_mat, W_mat, sigma2=config.sigma2)
                # Compute MSE loss compared to the normalized initial beamformer.
                mse_loss = fun.mse_loss(W_next, normlized_W0)
                total_rate += rate
                total_mse_loss += mse_loss
            
            # Combine unsupervised (sum rate maximization) and supervised (MSE) losses.
            loss_unsupervised = - total_rate / config.T
            loss_supervised = total_mse_loss / config.T
            loss = (1 - teacher_weight) * loss_unsupervised + (teacher_weight * 2000) * loss_supervised
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ave_rate = total_rate.item() / config.T
            epoch_loss += loss.item()
            epoch_rate += ave_rate
            rate_history.append(th.tensor(ave_rate))
            max_rate = max(max_rate, ave_rate)
            pbar_batches += 1
            pbar.set_description(f"Epoch {epoch+1}, Avg Rate: {ave_rate:.4f}, Max Rate: {max_rate:.4f}")
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        ave_pbar_rate = epoch_rate / pbar_batches
        test_pbar_rate = max_rate
        print(f"Epoch {epoch+1} Average Sum Rate: {ave_pbar_rate:.4f}")
        ave_rate_history.append(th.tensor(ave_pbar_rate))
        test_rate_history.append(th.tensor(test_pbar_rate))

    rate_history = th.stack(rate_history)
    ave_rate_history = th.stack(ave_rate_history)
    test_rate_history = th.stack(test_rate_history)
    th.save(rate_history, f"rate_train_history_{config.num_users}_{config.num_tx}.pth")
    rate_history = th.load(f"rate_train_history_{config.num_users}_{config.num_tx}.pth")
    th.save(ave_rate_history, f"rate_ave_history_{config.num_users}_{config.num_tx}.pth")
    ave_rate_history = th.load(f"rate_ave_history_{config.num_users}_{config.num_tx}.pth")
    th.save(test_rate_history, f"rate_test_history_{config.num_users}_{config.num_tx}.pth")
    test_rate_history = th.load(f"rate_test_history_{config.num_users}_{config.num_tx}.pth")

    # Create x-axis values for plots.
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
# 8. Config class.
#############################################
class BeamformerTransformerConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs['d_model']      # Transformer model dimension.
        self.beam_dim = kwargs['beam_dim']      # Beamformer vector dimension.
        self.n_head = kwargs['n_head']          # Number of attention heads.
        self.n_layers = kwargs['n_layers']      # Number of transformer layers.
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self.weight_decay = kwargs['weight_decay']
        self.max_epoch = kwargs['max_epoch']
        self.num_users = kwargs['num_users']
        self.num_tx = kwargs['num_tx']
        self.sigma2 = kwargs['sigma2']
        self.T = kwargs['T']                    # Number of L2O steps.
        self.SNR_power = kwargs['SNR_power']
        self.attn_pdrop = kwargs['attn_pdrop']
        self.resid_pdrop = kwargs['resid_pdrop']
        self.mlp_ratio = kwargs['mlp_ratio']
        self.subspace_dim = kwargs['subspace_dim']
        self.pbar_size = kwargs['pbar_size']

if __name__ == "__main__":
    # Example configuration where num_users = num_tx = 8.
    num_users = 8
    num_tx = 8
    d_model = 128          # Transformer token dimension.
    beam_dim = 2 * num_tx * num_users  # Beamformer vector dimension.
    n_head = 8
    n_layers = 5
    T = 5                  # Number of L2O steps set to 5.
    batch_size = 256 
    learning_rate = 5e-4
    weight_decay = 0.05
    max_epoch = 100
    sigma2 = 1.0  
    SNR = 15
    SNR_power = 10 ** (SNR / 10)
    attn_pdrop = 0.0
    resid_pdrop = 0.0
    mlp_ratio = 4
    subspace_dim = 4
    pbar_size = 3000

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
        pbar_size=pbar_size
    )
    
    # Train the L2O beamforming transformer.
    train_beamforming_transformer(config)
