import math
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from baseline_mmse import compute_mmse_beamformer
from pdb import set_trace as bp
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

mse_loss_fn = nn.MSELoss()

#############################################
# 1. System performance: Sum Rate for multi-user MISO
#############################################
def sum_rate(H, W, sigma2=1.0):
    # Compute batch-wise sum rate
    prod = th.bmm(H, W)  # (B, N, N)
    signal_power = th.abs(th.diagonal(prod, dim1=-2, dim2=-1))**2
    interference = th.sum(th.abs(prod)**2, dim=-1) - signal_power
    SINR = signal_power / (interference + sigma2)
    reward = th.log2(1 + SINR).sum(dim=-1).mean()
    return reward

#############################################
# 2. Self-Attention Block
#############################################
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_pdrop, resid_pdrop):
        super(SelfAttentionBlock, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Q/K/V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(d_model, d_model)
        # LayerNorms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # Feed-forward MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(4 * d_model, 4 * d_model),  # extra layer
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
        # Pre-LN
        q_ln = self.ln1(query)
        k_ln = self.ln1(key)
        v_ln = self.ln1(value)
        B, L, _ = query.size()
        # Compute Q, K, V and reshape
        Q = self.q_proj(q_ln).view(B, self.n_head, L, self.head_dim)
        K = self.k_proj(k_ln).view(B, self.n_head, L, self.head_dim)
        V = self.v_proj(v_ln).view(B, self.n_head, L, self.head_dim)
        # Scaled dot-product
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # Weighted sum
        y = att @ V
        y = y.transpose(1, 2).contiguous().view(B, L, self.n_head * self.head_dim)
        y = self.proj(y)
        y = self.resid_drop(y)
        # Residual + MLP
        out = query + y
        out = out + self.mlp(self.ln2(out))
        return out

#############################################
# 3. Beamforming Transformer with Historical Self-Attention
#############################################
class BeamformingTransformer(nn.Module):
    def __init__(self, config):
        super(BeamformingTransformer, self).__init__()
        self.config = config
        self.N = config.num_tx
        self.K = config.num_users
        self.T = config.T
        
        # Projections for primary tokens
        self.ant_proj = nn.Linear(self.K, config.d_model)
        self.user_proj = nn.Linear(self.N, config.d_model)

        # Projections for historical W tokens
        self.hist_ant_proj = nn.Linear(self.K, config.d_model)
        self.hist_user_proj = nn.Linear(self.N, config.d_model)
        # Positional embeddings for primary tokens (unchanged)
        self.pos_emb_ant_H = nn.Parameter(th.zeros(2 * self.N, config.d_model))
        self.pos_emb_ant_W = nn.Parameter(th.zeros(2 * self.N, config.d_model))
        self.pos_emb_ant_P = nn.Parameter(th.zeros(2 * self.N, config.d_model))
        self.pos_emb_user_H = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        self.pos_emb_user_W = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        self.pos_emb_user_P = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        # Self-attention blocks
        self.self_attn_ant = SelfAttentionBlock(config.d_model, config.n_head,
                                                 config.attn_pdrop, config.resid_pdrop)
        self.self_attn_user = SelfAttentionBlock(config.d_model, config.n_head,
                                                  config.attn_pdrop, config.resid_pdrop)
        # Historical self-attention blocks
        self.hist_attn_ant = SelfAttentionBlock(config.d_model, config.n_head,
                                                 config.attn_pdrop, config.resid_pdrop)
        self.hist_attn_user = SelfAttentionBlock(config.d_model, config.n_head,
                                                  config.attn_pdrop, config.resid_pdrop)
        # Output projection: account for primary + historical tokens
        total_primary = 6 * self.N + 6 * self.K  # primary token count
        total_hist = 2 * self.T * self.N + 2 * self.T * self.K          # hist: 2*N*T for ant + 2*K*T for user, here N=K
        total_tokens = total_primary + total_hist
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
        # Initialize weights
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

    def forward(self, H, W_prev, W_hist_cat):

        ## 1. Spatial-domain self-attention
        # H: (B, K, N); W_prev: (B, K, N); W_hist_cat: (B, T, K, N)
        B = H.size(0)

        # Primary self-attention as before
        Prod = th.bmm(H, W_prev.transpose(-1, -2))

        # Antenna-level primary tokens
        C_ant = th.cat([H.real, H.imag,
                        W_prev.real, W_prev.imag,
                        Prod.real, Prod.imag], dim=2)  # (B, K, 6N)
        tokens_ant = C_ant.transpose(1, 2)  # (B, 6N, K)
        tokens_ant = self.ant_proj(tokens_ant) # (B, 6N, d_model)

        # Add pos embeddings per segment
        Hp = tokens_ant[:, :2*self.N, :] + self.pos_emb_ant_H.unsqueeze(0)
        Wp = tokens_ant[:, 2*self.N:4*self.N, :] + self.pos_emb_ant_W.unsqueeze(0)
        Pp = tokens_ant[:, 4*self.N:6*self.N, :] + self.pos_emb_ant_P.unsqueeze(0)

        tokens_ant = th.cat([Hp, Wp, Pp], dim=1) # (B, 6N, d_model)

        x_a = self.self_attn_ant(tokens_ant, tokens_ant, tokens_ant) # (B, 6N, d_model)
        x_a_flat = x_a.view(B, -1) # Flatten to (B, 6N*d_model)

        # User-level primary tokens
        H_T = H.transpose(1, 2)
        W_T = W_prev.transpose(1, 2)
        Prod_T = Prod.transpose(1, 2)
        C_user = th.cat([H_T.real, H_T.imag,
                          W_T.real, W_T.imag,
                          Prod_T.real, Prod_T.imag], dim=2) # (B, N, 6K)
        
        tokens_user = C_user.transpose(1, 2)
        tokens_user = self.user_proj(tokens_user) # (B, 6K, d_model)

        Hu = tokens_user[:, :2*self.K, :] + self.pos_emb_user_H.unsqueeze(0)
        Wu = tokens_user[:, 2*self.K:4*self.K, :] + self.pos_emb_user_W.unsqueeze(0)
        Pu = tokens_user[:, 4*self.K:6*self.K, :] + self.pos_emb_user_P.unsqueeze(0)

        tokens_user = th.cat([Hu, Wu, Pu], dim=1)

        x_u = self.self_attn_user(tokens_user, tokens_user, tokens_user)
        x_u_flat = x_u.view(B, -1) # Flatten to (B, 6K*d_model)


        ## 2. Temporal-domain self-attention

        # Historical self-attention: antenna-level
        # W_hist_cat: (B, T, N, N)
        W_real = W_hist_cat.real.permute(0,2,1,3).reshape(B, self.N, self.T*self.N)
        W_imag = W_hist_cat.imag.permute(0,2,1,3).reshape(B, self.N, self.T*self.N)
        C_hist_ant = th.cat([W_real, W_imag], dim=2)  # (B, N, 2*T*N)
        tokens_hist_ant = C_hist_ant.transpose(1,2)   # (B, 2*T*N, N)
        tokens_hist_ant = self.hist_ant_proj(tokens_hist_ant) # (B, 2*T*N, d_model)
        y_a = self.hist_attn_ant(tokens_hist_ant, tokens_hist_ant, tokens_hist_ant) # (B, 2*T*N, d_model)
        y_a_flat = y_a.view(B, -1) # Flatten to (B, 2*T*N*d_model)

        # Historical self-attention: user-level
        W_T_hist = W_hist_cat.transpose(-1,-2)
        W_real_u = W_T_hist.real.permute(0,2,1,3).reshape(B, self.K, self.T*self.K)
        W_imag_u = W_T_hist.imag.permute(0,2,1,3).reshape(B, self.K, self.T*self.K)
        C_hist_user = th.cat([W_real_u, W_imag_u], dim=2)
        tokens_hist_user = C_hist_user.transpose(1,2)
        tokens_hist_user = self.hist_user_proj(tokens_hist_user) # (B, 2*T*K, d_model)
        y_u = self.hist_attn_user(tokens_hist_user, tokens_hist_user, tokens_hist_user) # (B, 2*T*K, d_model)
        y_u_flat = y_u.view(B, -1) # Flatten to (B, 2*T*K*d_model)

        # Fuse all features and project to next beamformer
        fused = th.cat([x_a_flat, x_u_flat, y_a_flat, y_u_flat], dim=1) # (B, 6N*d_model + 6K*d_model + 2*T*N*d_model + 2*T*K*d_model)
        W_next = self.out_proj(fused)
        W_next = W_next / th.norm(W_next, dim=1, keepdim=True)
        return W_next

#############################################
# 4. Channel Dataset (unchanged)
#############################################
class ChannelDataset(Dataset):
    def __init__(self, num_samples, num_users, num_tx, P, subspace_dim):
        super().__init__()
        self.num_samples = num_samples
        self.num_users = num_users
        self.num_tx = num_tx
        self.P = P
        self.subspace_dim = subspace_dim
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        scale = math.sqrt(2) / 2
        H_real = th.randn(self.num_users, self.num_tx) * scale
        H_imag = th.randn(self.num_users, self.num_tx) * scale
        H_combined = th.stack([H_real, H_imag], dim=0) * (self.P ** 0.5)
        return H_combined # (2, K, N)

#############################################
# 5. Optimizer configuration (unchanged)
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
# 6. Training Routine with history buffer
#############################################
def train_beamforming_transformer(config):

    dataset = ChannelDataset(config.pbar_size * config.batch_size,
                             config.num_users, config.num_tx,
                             config.SNR_power, config.subspace_dim)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = BeamformingTransformer(config).to(device)

    optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                    lambda epoch: 1 - 0.7 * (epoch / config.max_epoch))
    
    global mmse_rate_printed
    mmse_rate_printed = False

    model.train()
    rate_history = []
    test_rate_history = []
    ave_rate_history = []

    for epoch in range(config.max_epoch):
        teacher = max(0.0, 1.0 - (epoch / 10))

        epoch_loss = 0
        epoch_rate = 0
        pbar_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        max_rate = 0

        for batch in pbar:
            H_tensor = batch.to(device) # (B, 2, K, N)
            H_mat = H_tensor[:,0,:,:] + 1j * H_tensor[:,1,:,:] # (B, K, N)
            # Initial MMSE beamformer
            W0, _ = compute_mmse_beamformer(H_mat, config.num_users,
                                            config.num_tx, config.SNR_power,
                                            config.sigma2, device)
            
            W0 = W0.transpose(-1, -2).to(device)  # (B, N, K)
            vec_w0 = th.cat((th.real(W0).reshape(-1, config.num_tx * config.num_users), 
                             th.imag(W0).reshape(-1, config.num_tx * config.num_users)), dim=-1)
            vec_w0 = vec_w0.reshape(-1, 2 * config.num_tx * config.num_users)
            norm_W0 = th.norm(vec_w0, dim=1, keepdim=True)
            normlized_W0 = vec_w0 / norm_W0
            W_mat_0 = (normlized_W0[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) +
                       1j * normlized_W0[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
            rate_0 = sum_rate(H_mat, W_mat_0, sigma2=config.sigma2)
            if not mmse_rate_printed:
                mmse_rate_printed = True
                print(f"MMSE Rate: {rate_0.item():.4f}")

            W_prev = W_mat_0.transpose(-1,-2) # (B, K, N)
            # Initialize history buffer with W_prev
            hist_ws = [W_prev for _ in range(config.T)]  # (B, T, K, N)
            total_rate, total_mse = 0, 0

            for t in range(1, config.T+1):
                W_cat = th.stack(hist_ws, dim=1) # (B, T, K, N)
                # Predict next beam vector
                W_vec = model(H_mat, W_prev, W_cat)
                # Reshape to complex matrix
                real = W_vec[:,:config.num_tx*config.num_users].reshape(-1,config.num_tx,config.num_users)
                imag = W_vec[:,config.num_tx*config.num_users:].reshape(-1,config.num_tx,config.num_users)
                W_mat = real + 1j*imag # (B, N, K)

                W_prev = W_mat.transpose(-1,-2) # (B, K, N)
                # Update history and prev
                hist_ws.pop(0); hist_ws.append(W_prev)
                
                # Losses
                rate = sum_rate(H_mat, W_mat, config.sigma2)
                mse = F.mse_loss(W_vec, normlized_W0)
                total_rate += rate
                total_mse += mse

            # Combine losses
            loss_unsup = - total_rate / config.T
            loss_sup = total_mse / config.T
            loss = (1-teacher)*loss_unsup + (teacher*2000)*loss_sup
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
        ave_pbar_rate = epoch_rate / pbar_batches
        test_pbar_rate = max_rate
        print(f"Epoch {epoch+1} Average Sum Rate: {ave_pbar_rate:.4f}")
        ave_rate_history.append(th.tensor(ave_pbar_rate))
        test_rate_history.append(th.tensor(test_pbar_rate))

    rate_history = th.stack(rate_history)
    ave_rate_history = th.stack(ave_rate_history)
    test_rate_history = th.stack(test_rate_history)
    th.save(rate_history, f"rate_train_history_{config.num_users}_{config.num_tx}_self_attn_tied_L2O.pth")
    th.save(ave_rate_history, f"rate_ave_history_{config.num_users}_{config.num_tx}_self_attn_tied_L2O.pth")
    th.save(test_rate_history, f"rate_test_history_{config.num_users}_{config.num_tx}_self_attn_tied_L2O.pth")

    # Save the trained model weights
    import os
    save_dir = r"D:/phd_2025_spring/Transformer_BF_with_prod-main/pre_trained_model"
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model with configuration info in the filename
    model_filename = f"transformer_model_{config.num_users}_{config.num_tx}_d_{config.d_model}_T_{config.T}_self_attn_tied_L2O.pt"
    model_path = os.path.join(save_dir, model_filename)
    
    # Save both model state dict and full model
    th.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'epoch': config.max_epoch,
        'final_rate': test_rate_history[-1].item()
    }, model_path)
    
    print(f"Model saved to {model_path}")

    # Create x-axis values for both plots.
    x_rate_history = th.arange(len(rate_history))
    x_test_rate_history = th.arange(len(test_rate_history)) * config.pbar_size
    x_baseline_rate = th.arange(0, len(rate_history), 5000)
    wmmse_history = th.tensor([51.67755418089099 for _ in range(len(x_baseline_rate))])
    mmse_history = th.tensor([43.14968651013214 for _ in range(len(x_baseline_rate))])

    plt.figure(figsize=(10, 6))
    plt.plot(x_rate_history, rate_history, marker='.', linestyle='-', color='b', label='Training Rate')
    # plt.plot(x_test_rate_history, ave_rate_history, marker='*', linestyle='-', color='g', label='Average Rate')
    plt.plot(x_test_rate_history, test_rate_history, marker='o', linestyle='-', color='r', label='Testing Rate')
    plt.plot(x_baseline_rate, wmmse_history, marker='.', linestyle='--', color='k', label='WMMSE Rate')
    plt.plot(x_baseline_rate, mmse_history, marker='.', linestyle='--', color='g', label='MMSE Rate')
    plt.title(f"Training and Testing Rate when N={config.num_tx} and N={config.num_users} with Self-Attention Tied L2O")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Sum Rate")
    plt.grid(True, which="both", ls="--")
    plt.show()

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

#############################################
# Main entry point.
#############################################
if __name__ == "__main__":
    # Example configuration where num_users = num_tx = 8.
    num_users = 16
    num_tx = 16
    d_model = 256          # Transformer token dimension
    beam_dim = 2 * num_tx * num_users  # Beamformer vector dimension
    n_head = 8
    n_layers = 6
    T = 3
    batch_size = 256 
    learning_rate = 2e-4
    weight_decay = 0.05
    max_epoch = 400
    sigma2 = 1.0  
    SNR = 15
    SNR_power = 10 ** (SNR / 10)
    attn_pdrop = 0.0
    resid_pdrop = 0.0
    mlp_ratio = 4
    subspace_dim = 4
    pbar_size = 1000
    
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
    
    # Train the beamforming transformer.
    train_beamforming_transformer(config)
