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

mse_loss_fn = nn.MSELoss()

#############################################
# 1. System performance: Sum Rate for multi-user MISO
#############################################
def sum_rate(H, W, sigma2=1.0):
    prod = th.bmm(H, W)  # (B, N, N), where N = num_users = num_tx
    signal_power = th.abs(th.diagonal(prod, dim1=-2, dim2=-1))**2
    interference = th.sum(th.abs(prod)**2, dim=-1) - signal_power
    N_val = sigma2
    SINR = signal_power / (interference + N_val)
    reward = th.log2(1 + SINR).sum(dim=-1).mean()
    return reward

#############################################
# Modified Cross-Attention Block supporting self-attention.
#############################################
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_pdrop, resid_pdrop):
        super(SelfAttentionBlock, self).__init__()
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
            nn.LayerNorm(4 * d_model),
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(4 * d_model, 4 * d_model),  # additional layer
            nn.LayerNorm(4 * d_model),
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
    
    def forward(self, query, key, value):
        """
        Performs cross-attention using separate Query, Key, and Value inputs.
        
        Args:
            query: Query tokens, shape (B, L, d_model).
            key:   Key tokens, shape (B, L, d_model).
            value: Value tokens, shape (B, L, d_model).
        Returns:
            Output with shape (B, L, d_model).
        """
        B, L, _ = query.size()
        # Apply LayerNorm
        q_ln = self.ln1(query)
        k_ln = self.ln1(key)
        v_ln = self.ln1(value)
        # Compute Q, K, V and reshape for multi-head attention.
        Q = self.q_proj(q_ln).view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, L, head_dim)
        K = self.k_proj(k_ln).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        V = self.v_proj(v_ln).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention.
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, L, L)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # Weighted sum of values.
        y = att @ V  # (B, n_head, L, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, L, self.n_head * self.head_dim)
        y = self.proj(y)
        y = self.resid_drop(y)
        # Residual connection and MLP.
        out = query + y
        out = out + self.mlp(self.ln2(out))
        return out

#############################################
# Modified Beamforming Transformer with Updated Token Construction.
#############################################
class BeamformingTransformer(nn.Module):
    def __init__(self, config):
        super(BeamformingTransformer, self).__init__()
        self.config = config
        # Set dimensions; note that num_users = num_tx = N.
        self.N = config.num_tx  # N (number of transmit antennas)
        self.K = config.num_users  # equals N
        
        # New projection layers for tokens constructed from concatenated matrices.
        # Each token (a column vector) has dimension N; project to d_model.
        self.ant_proj = nn.Linear(self.N, config.d_model)
        self.user_proj = nn.Linear(self.N, config.d_model)
        
        # Updated positional embeddings:
        # For antenna-level tokens: there are 6N tokens, divide into 3 segments of 2N each.
        self.pos_emb_ant_H = nn.Parameter(th.zeros(2 * self.N, config.d_model))  # for H_real/H_imag
        self.pos_emb_ant_W = nn.Parameter(th.zeros(2 * self.N, config.d_model))  # for W_real/W_imag
        self.pos_emb_ant_P = nn.Parameter(th.zeros(2 * self.N, config.d_model))  # for P_real/P_imag
        # For user-level tokens: there are also 6N tokens.
        self.pos_emb_user_H = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        self.pos_emb_user_W = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        self.pos_emb_user_P = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        
        # Self-attention blocks (using the cross-attention module in self-attention mode).
        self.self_attn_ant = SelfAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
                                                    attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        self.self_attn_user = SelfAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
                                                     attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        
        # Final fusion and output projection.
        # The fused feature dimension is 12N*d_model (6N from antenna + 6N from user).
        total_tokens = 6*self.N + 6*self.K
        # self.out_mlp = nn.Sequential(
        #     nn.Linear(total_tokens * d_model, d_model),
        #     # nn.ReLU(),  
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Linear(d_model, config.beam_dim)
        # )
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
        Forward pass for the Beamforming Transformer.
        
        Args:
            H: Complex channel matrix, shape (B, N, N).
            W_prev: Previous beamformer matrix, shape (B, N, N), complex.
        Returns:
            W_next: Vectorized predicted beamformer, shape (B, beam_dim).
        """
        B = H.size(0)
        N = self.N  # N
        
        # ---------------------------
        # Compute the product: Prod = H * (W_prev^T).
        # Since the matrices are square, Prod has shape (B, N, N).
        Prod = th.bmm(H, W_prev.transpose(-1, -2))
        
        # ---------------------------
        # 1. Antenna-level Self-Attention
        # Construct matrix C_antenna by concatenating the real and imaginary parts of H, W_prev, and Prod along columns.
        # C_ant has shape (B, N, 6N); we then transpose to obtain tokens of shape (B, 6N, N)
        C_ant = th.cat([H.real, H.imag, W_prev.real, W_prev.imag, Prod.real, Prod.imag], dim=2)
        tokens_ant = C_ant.transpose(1, 2)  # (B, 6N, N)
        
        # Project tokens into d_model and add positional embeddings.
        tokens_ant_proj = self.ant_proj(tokens_ant)  # (B, 6N, d_model)

        H_part_ant = tokens_ant_proj[:, :2*self.N, :]
        W_part_ant = tokens_ant_proj[:, 2*self.N:4*self.N, :]
        P_part_ant = tokens_ant_proj[:, 4*self.N:6*self.N, :]

        H_part_ant = H_part_ant + self.pos_emb_ant_H.unsqueeze(0)
        W_part_ant = W_part_ant + self.pos_emb_ant_W.unsqueeze(0)
        P_part_ant = P_part_ant + self.pos_emb_ant_P.unsqueeze(0)

        tokens_ant_proj = th.cat([H_part_ant, W_part_ant, P_part_ant], dim=1) # (B, 6N, d_model)
        
        # Perform self-attention on antenna-level tokens.
        x_a = self.self_attn_ant(tokens_ant_proj, tokens_ant_proj, tokens_ant_proj)  # (B, 6N, d_model)
        
        # ---------------------------
        # 2. User-level Self-Attention
        # Transpose H, W_prev, and Prod; then concatenate their real and imaginary parts along columns.
        # This forms C_user of shape (B, N, 6N), which is transposed to shape (B, 6N, N) so that each column is a token.
        H_T = H.transpose(1, 2)
        W_T = W_prev.transpose(1, 2)
        Prod_T = Prod.transpose(1, 2)
        C_user = th.cat([H_T.real, H_T.imag, W_T.real, W_T.imag, Prod_T.real, Prod_T.imag], dim=2)
        tokens_user = C_user.transpose(1, 2)  # (B, 6N, N)
        
        # Project tokens into d_model and add positional embeddings.
        tokens_user_proj = self.user_proj(tokens_user)  # (B, 6N, d_model)

        H_part_user = tokens_user_proj[:, :2*self.K, :]
        W_part_user = tokens_user_proj[:, 2*self.K:4*self.K, :]
        P_part_user = tokens_user_proj[:, 4*self.K:6*self.K, :]

        H_part_user = H_part_user + self.pos_emb_user_H.unsqueeze(0)
        W_part_user = W_part_user + self.pos_emb_user_W.unsqueeze(0)
        P_part_user = P_part_user + self.pos_emb_user_P.unsqueeze(0)

        tokens_user_proj = th.cat([H_part_user, W_part_user, P_part_user], dim=1) # (B, 6N, d_model)
        
        # Perform self-attention on user-level tokens.
        x_u = self.self_attn_user(tokens_user_proj, tokens_user_proj, tokens_user_proj)  # (B, 6N, d_model)
        
        # ---------------------------
        # 3. Fusion and Output Prediction
        # Flatten both attention outputs and concatenate.
        x_a_flat = x_a.view(B, -1)   # (B, 6N * d_model)
        x_u_flat = x_u.view(B, -1)   # (B, 6N * d_model)
        x_fused = th.cat([x_a_flat, x_u_flat], dim=-1)  # (B, 12N * d_model)
        
        # Final projection to beamformer vector followed by normalization.
        W_next = self.out_proj(x_fused)  # (B, beam_dim)
        norm = th.norm(W_next, dim=1, keepdim=True)
        W_next = W_next / norm
        return W_next

#############################################
# Channel Dataset (unchanged)
#############################################
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
    
    def __getitem__(self, idx):
        scale = math.sqrt(2) / 2
        H_real = th.randn(self.num_users, self.num_tx) * scale
        H_imag = th.randn(self.num_users, self.num_tx) * scale
        H_combined = th.stack([H_real, H_imag], dim=0)  # Shape: (2, num_users, num_tx)
        H_combined = H_combined * (self.P ** 0.5)
        return th.tensor(H_combined)

#############################################
# Optimizer configuration (unchanged)
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

#############################################
# Training Routine (unchanged)
#############################################
def train_beamforming_transformer(config, pretrained_path: str = None):
    """
    Train the beamforming transformer based on the given configuration.
    If `pretrained_path` is provided, load those weights before training.
    Returns the path where the final model was saved.
    """
    dataset = ChannelDataset(num_samples=config.pbar_size * config.batch_size,
                             num_users=config.num_users,
                             num_tx=config.num_tx,
                             P=config.SNR_power,
                             subspace_dim=config.subspace_dim)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # 1) Instantiate the model
    model = BeamformingTransformer(config).to(device)

    # 2) Optionally load pretrained weights
    if pretrained_path is not None:
        checkpoint = th.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)

    # Learning rate scheduler that linearly decays lr.
    scheduler = th.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - 0.1 * (epoch / config.max_epoch)
    )
    
    global mmse_rate_printed
    mmse_rate_printed = False
    model.train()
    teacher_weight = 0  # 0 or 1 depending on epoch switching
    rate_history = []
    test_rate_history = []
    ave_rate_history = []
    
    for epoch in range(config.max_epoch):
        # # Hard switch learning policy.
        # if epoch < 3:
        #     teacher_weight = 1
        # else:
        #     teacher_weight = 0.0

        # Soft switch learning policy.
        teacher_weight = max(0.0, 1.0 - (epoch / 10))

        epoch_loss = 0
        epoch_rate = 0
        pbar_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        max_rate = 0
        for batch in pbar:
            H_tensor = batch.to(device)
            batch_size = H_tensor.size(0)
            # Convert to complex channel matrix: (B, N, N)
            H_mat = H_tensor[:, 0, :, :] + 1j * H_tensor[:, 1, :, :]
            
            # Compute initial beamformer using MMSE.
            W0, _ = compute_mmse_beamformer(H_mat, config.num_users, config.num_tx, 
                                            config.SNR_power, config.sigma2, device)
            W0 = W0.transpose(-1, -2).to(device)  # (B, N, N)
            vec_w0 = th.cat((th.real(W0).reshape(-1, config.num_tx * config.num_users), 
                             th.imag(W0).reshape(-1, config.num_tx * config.num_users)), dim=-1)
            vec_w0 = vec_w0.reshape(-1, 2 * config.num_tx * config.num_users)
            norm_W0 = th.norm(vec_w0, dim=1, keepdim=True)
            normlized_W0 = vec_w0 / norm_W0
            # Reconstruct complex beamformer: (B, N, N)
            W_mat_0 = (normlized_W0[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) +
                       1j * normlized_W0[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
            rate_0 = sum_rate(H_mat, W_mat_0, sigma2=config.sigma2)
            if not mmse_rate_printed:
                mmse_rate_printed = True
                print(f"MMSE Rate: {rate_0.item():.4f}")
            
            # Initialize W_prev (transpose to get shape (B, N, N)).
            W_prev = W_mat_0.transpose(-1, -2).to(device)
            total_rate = 0
            total_mse_loss = 0
            for t in range(1, config.T + 1):
                # The model predicts the next beamformer.
                W_next = model(H_mat, W_prev)  # (B, beam_dim)
                # Convert W_next to complex beamformer matrix: (B, N, N)
                W_mat = (W_next[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) +
                         1j * W_next[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
                W_prev = W_mat.transpose(-1, -2).to(device)  # (B, N, N)
                rate = sum_rate(H_mat, W_mat, sigma2=config.sigma2)
                mse_loss = fun.mse_loss(W_next, normlized_W0)
                total_rate += rate
                total_mse_loss += mse_loss
            
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
    th.save(rate_history, f"rate_train_history_{config.num_users}_{config.num_tx}_self_attn.pth")
    th.save(ave_rate_history, f"rate_ave_history_{config.num_users}_{config.num_tx}_self_attn.pth")
    th.save(test_rate_history, f"rate_test_history_{config.num_users}_{config.num_tx}_self_attn.pth")

    # Save the trained model weights
    import os
    save_dir = r"D:/phd_2025_spring/Transformer_BF_with_prod-main/pre_trained_model"
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model with configuration info in the filename
    model_filename = f"transformer_model_{config.num_users}_{config.num_tx}_d_{config.d_model}_T_{config.T}_self_attn_stage_{config.checknum}.pt"
    model_path = os.path.join(save_dir, model_filename)
    
    # Save both model state dict and full model
    th.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'epoch': config.max_epoch,
        'final_rate': test_rate_history[-1].item()
    }, model_path)
    
    print(f"Model saved to {model_path}")

    return model_path


    # # Create x-axis values for both plots.
    # x_rate_history = th.arange(len(rate_history))
    # x_test_rate_history = th.arange(len(test_rate_history)) * config.pbar_size

    # plt.figure(figsize=(10, 6))
    # plt.plot(x_rate_history, rate_history, marker='.', linestyle='-', color='b', label='Training Rate')
    # plt.plot(x_test_rate_history, ave_rate_history, marker='*', linestyle='-', color='g', label='Average Rate')
    # plt.plot(x_test_rate_history, test_rate_history, marker='o', linestyle='-', color='r', label='Testing Rate')
    # plt.title(f"Training and Testing Rate when N={config.num_tx} and N={config.num_users}")
    # plt.legend()
    # plt.xlabel("Epochs")
    # plt.ylabel("Sum Rate")
    # plt.grid(True, which="both", ls="--")
    # plt.show()

#############################################
# Config Class (unchanged)
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
        self.checknum = kwargs['checknum'] 

#############################################
# Main entry point.
#############################################
if __name__ == "__main__":

    config = BeamformerTransformerConfig(
        d_model=256,          # Transformer token dimension
        beam_dim=2 * 16 * 16,  # Beamformer vector dimension (2 * num_tx * num_users)
        n_head=8,
        n_layers=6,
        batch_size=256, 
        learning_rate=5e-4,
        weight_decay=0.05,
        max_epoch=100,
        num_users=16,
        num_tx=16,
        sigma2=1.0,
        T=1,
        SNR_power=10 ** (15 / 10),  # 15dB SNR
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        mlp_ratio=4,
        subspace_dim=4,
        pbar_size=500,
        checknum=1  
    )

    config_2 = BeamformerTransformerConfig(
        d_model=256,          # Transformer token dimension
        beam_dim=2 * 16 * 16,  # Beamformer vector dimension (2 * num_tx * num_users)
        n_head=8,
        n_layers=6,
        batch_size=256, 
        learning_rate=5e-5,
        weight_decay=0.05,
        max_epoch=300,
        num_users=16,
        num_tx=16,
        sigma2=1.0,
        T=5,
        SNR_power=10 ** (15 / 10),  # 15dB SNR
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        mlp_ratio=4,
        subspace_dim=4,
        pbar_size=500,
        checknum=2
    )
        
    # Train the beamforming transformer.
    model_path_stage_1 = train_beamforming_transformer(config)

    model_path_stage2 = train_beamforming_transformer(config_2, pretrained_path=model_path_stage_1)


