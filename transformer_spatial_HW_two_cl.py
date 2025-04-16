
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
        
        # LayerNorm for each sublayer.
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # MLP block with an extra layer (scheme 2).
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(4 * d_model, 4 * d_model),  # additional layer
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
        q_ln = self.ln1(query)
        k_ln = self.ln1(key)
        v_ln = self.ln1(value)
        
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
        out = out + self.mlp(self.ln2(out))
        return out

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
        
        # Positional embeddings.
        # For antenna-level, tokens from columns (real and imaginary parts).
        self.pos_emb_ant = nn.Parameter(th.zeros(2 * self.N, config.d_model))
        # For user-level, tokens from rows (real and imaginary parts).
        self.pos_emb_user = nn.Parameter(th.zeros(2 * self.K, config.d_model))
        
        # Cross-Attention blocks.
        self.cross_attn_ant = CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
                                                    attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        self.cross_attn_user = CrossAttentionBlock(d_model=config.d_model, n_head=config.n_head, 
                                                     attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop)
        
        # Final fusion and output projection MLP.
        self.out_proj = nn.Sequential(
            nn.Linear((2 * self.N + 2 * self.K) * config.d_model, config.d_model),
            nn.ReLU(),  
            nn.Linear(config.d_model, config.beam_dim)
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
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
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
        V_ant = th.cat([Prod.real.transpose(1,2), Prod.imag.transpose(1,2)], dim=1)  # (B, 2*K, K)
        # Project V_ant to the desired dimension.
        V_ant_proj = self.antenna_prod_proj(V_ant)  # (B, 2*K, d_model)
        # Add positional embedding to Value tokens (antenna-level).
        V_ant_proj = V_ant_proj + self.pos_emb_ant.unsqueeze(0)
        
        # Add positional embeddings to Query and Key tokens.
        H_ant_proj = H_ant_proj + self.pos_emb_ant.unsqueeze(0)
        W_ant_proj = W_ant_proj + self.pos_emb_ant.unsqueeze(0)
        
        # Cross-attention for antenna-level.
        x_a = self.cross_attn_ant(H_ant_proj, W_ant_proj, V_ant_proj)  # (B, 2*N, d_model)
        
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
        V_user = th.cat([Prod.real, Prod.imag], dim=1)  # (B, 2*K, K)
        # Project V_user to the desired dimension.
        V_user_proj = self.user_prod_proj(V_user)  # (B, 2*K, d_model)
        # Add positional embedding to Value tokens (user-level).
        V_user_proj = V_user_proj + self.pos_emb_user.unsqueeze(0)
        
        # Add positional embeddings to Query and Key tokens.
        H_user_proj = H_user_proj + self.pos_emb_user.unsqueeze(0)
        W_user_proj = W_user_proj + self.pos_emb_user.unsqueeze(0)
        
        # Cross-attention for user-level.
        x_u = self.cross_attn_user(H_user_proj, W_user_proj, V_user_proj)  # (B, 2*K, d_model)
        
        # ---------------------------
        # 3. Fusion and Output Prediction
        # ---------------------------
        # Flatten the outputs and concatenate.
        x_a_flat = x_a.view(B, -1)      # (B, (2*N)*d_model)
        x_u_flat = x_u.view(B, -1)      # (B, (2*K)*d_model)
        x_fused = th.cat([x_a_flat, x_u_flat], dim=-1)  # (B, (2*N+2*K)*d_model)
        
        # Project fused features to final beamformer vector and normalize.
        W_next = self.out_proj(x_fused)  # (B, beam_dim)
        norm = th.norm(W_next, dim=1, keepdim=True)
        W_next = W_next / norm
        return W_next

#############################################
# 5. Channel dataset: random Gaussian H
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
        coordinates = th.randn(self.subspace_dim, 1)
        basis_vectors_subset = self.basis_vectors[:self.subspace_dim].T
        vec_channel = th.matmul(basis_vectors_subset, coordinates).reshape(2 * self.num_users * self.num_tx)
        H_real = vec_channel[:self.num_tx * self.num_users].reshape(self.num_users, self.num_tx) ### (num_users, num_tx)
        H_imag = vec_channel[self.num_tx * self.num_users:].reshape(self.num_users, self.num_tx) ### (num_users, num_tx)
        H_complex = H_real + 1j * H_imag
        norm_H_complex = th.sum(th.abs(H_complex)**2)
        SNR_power = self.num_users*self.num_tx*self.P
        H_real = H_real * th.sqrt((SNR_power*0.5) / norm_H_complex) ### (num_users, num_tx)
        H_imag = H_imag * th.sqrt((SNR_power*0.5) / norm_H_complex) ### (num_users, num_tx)
        H_combined = th.stack([H_real, H_imag], dim=0) ### Shape: (2, num_users, num_tx)

        # scale = math.sqrt(2) / 2
        # H_real = th.randn(self.num_users, self.num_tx) * scale
        # H_imag = th.randn(self.num_users, self.num_tx) * scale
        # H_combined = th.stack([H_real, H_imag], dim=0)  # Shape: (2, num_users, num_tx)
        # H_combined = H_combined * (self.P ** 0.5)

        return th.tensor(H_combined)

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

def get_mmse_obj(H, W, sigma2=1.0):
    prod = th.bmm(H, W)  # (B, num_users, num_users)
    fro_norm = th.norm(prod, p='fro', dim=(1,2)) ** 2
    trace_real = th.stack([th.trace(prod).real for prod in prod])
    MSE_obj = fro_norm - trace_real
    return MSE_obj

#############################################
# 6. Training Routine
#############################################
def train_beamforming_transformer(config):
    """
    Train the beamforming transformer based on the given configuration.
    """
    # dataset = ChannelDataset(num_samples=config.pbar_size * config.batch_size,
    #                          num_users=config.num_users,
    #                          num_tx=config.num_tx,
    #                          P=config.SNR_power,
    #                          subspace_dim=config.subspace_dim)
    # dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = BeamformingTransformer(config).to(device)
    optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)

    # Learning rate scheduler that linearly decays lr.
    scheduler = th.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - 0.3 * (epoch / config.max_epoch)
    )
    
    global mmse_rate_printed
    mmse_rate_printed = False
    model.train()
    
    rate_history = []
    test_rate_history = []
    ave_rate_history = []

    teacher_weight = 0  # 0 or 1 depending on epoch switching
    initial_subspace_dim = config.ini_sub_dim
    cl_increment = config.ini_sub_dim
    # current_subspace_dim = initial_subspace_dim
    
    for epoch in range(config.max_epoch):

        current_subspace_dim = min(initial_subspace_dim + epoch * cl_increment, 2*config.num_users * config.num_tx)
        teacher_weight = min(1, teacher_weight + 0.01)  # Gradually increase teacher weight
        print(f"Current subspace dimension and teacher weight: {current_subspace_dim}, {teacher_weight:.2f}")

        dataset = ChannelDataset(num_samples=config.pbar_size*config.batch_size, 
                                 num_users=config.num_users, 
                                 num_tx=config.num_tx, 
                                 P=config.SNR_power, 
                                 subspace_dim=current_subspace_dim)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        # # Hard switch learning policy.
        # if epoch % config.sub_dim_epoch == 0:
        #     teacher_weight = 1
        # else:
        #     teacher_weight = 0.0

        epoch_loss = 0
        epoch_rate = 0
        pbar_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        max_rate = 0
        for batch in pbar:
            # Channel input: shape (B, 2, num_users, num_tx)
            H_tensor = batch.to(device)
            batch_size = H_tensor.size(0)
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
            rate_0 = sum_rate(H_mat, W_mat_0, sigma2=config.sigma2)
            if not mmse_rate_printed:
                mmse_rate_printed = True
                print(f"MMSE Rate: {rate_0.item():.4f}")
            
            # Initialize W_prev (transpose to get shape (B, num_users, num_tx)).
            W_prev = W_mat_0.transpose(-1, -2).to(device)
            total_rate = 0
            total_mse_loss = 0
            for t in range(1, config.T + 1):
                # The model predicts the next beamformer.
                W_next = model(H_mat, W_prev)  # (B, beam_dim)
                # Convert W_next to complex beamformer matrix: (B, num_tx, num_users)
                W_mat = (W_next[:, :config.num_tx * config.num_users].reshape(-1, config.num_tx, config.num_users) +
                         1j * W_next[:, config.num_tx * config.num_users:].reshape(-1, config.num_tx, config.num_users))
                W_prev = W_mat.transpose(-1, -2).to(device)  # (B, num_users, num_tx)
                rate = sum_rate(H_mat, W_mat, sigma2=config.sigma2)
                total_rate += rate

                ### supervised MSE loss
                mse_loss = fun.mse_loss(W_next, normlized_W0)
                total_mse_loss += mse_loss

                # ### unsupervised MSE loss
                # mse_loss = get_mmse_obj(H_mat, W_mat, sigma2=config.sigma2)   
                # total_mse_loss += mse_loss.mean()            
                
            
            loss_unsupervised = - total_rate / config.T
            loss_supervised = total_mse_loss / config.T
            loss = (1 - teacher_weight) * loss_unsupervised + (teacher_weight * 2000) * loss_supervised ## supervised MSE
            # loss = (1 - teacher_weight) * loss_unsupervised + (teacher_weight / 50) * loss_supervised ## supervised MSE
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
        # print(f"Epoch {epoch+1} completed. Learning Rate: {current_lr:.2e}")
        print(f"Epoch {epoch+1} Average Sum Rate: {ave_pbar_rate:.4f}")
        ave_rate_history.append(th.tensor(ave_pbar_rate))
        test_rate_history.append(th.tensor(test_pbar_rate))


    rate_history = th.stack(rate_history)
    ave_rate_history = th.stack(ave_rate_history)
    test_rate_history = th.stack(test_rate_history)
    th.save(rate_history, f"rate_train_history_{config.num_users}_{config.num_tx}_bind_cl.pth")
    rate_history = th.load(f"rate_train_history_{config.num_users}_{config.num_tx}_bind_cl.pth")
    th.save(ave_rate_history, f"rate_ave_history_{config.num_users}_{config.num_tx}_bind_cl.pth")
    ave_rate_history = th.load(f"rate_ave_history_{config.num_users}_{config.num_tx}_bind_cl.pth")
    th.save(test_rate_history, f"rate_test_history_{config.num_users}_{config.num_tx}_bind_cl.pth")
    test_rate_history = th.load(f"rate_test_history_{config.num_users}_{config.num_tx}_bind_cl.pth")

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
        self.ini_sub_dim = kwargs['ini_sub_dim']

if __name__ == "__main__":

    # # Example configuration where num_users = num_tx = 16.
    num_users = 16
    num_tx = 16
    d_model = 256 # Transformer single-token dimension
    beam_dim = 2*num_tx*num_users # Beamformer vector dimension
    n_head = 8 # Number of attention heads
    n_layers = 6 # Number of transformer layers
    T = 1 # Number of time steps
    batch_size = 256 
    learning_rate = 5e-5
    weight_decay = 0.05
    # max_epoch = 100
    sigma2 = 1.0  
    SNR = 15
    SNR_power = 10 ** (SNR/10) # SNR power in dB
    attn_pdrop = 0.0
    # resid_pdrop = 0.05
    # attn_pdrop = 0.0
    resid_pdrop = 0.0
    mlp_ratio = 4
    subspace_dim = 4
    pbar_size = 2000
    ini_sub_dim = 4
    max_epoch = (2*num_users*num_tx) // ini_sub_dim

    # # Example configuration where num_users = num_tx = 8.
    # num_users = 8
    # num_tx = 8
    # d_model = 128          # Transformer token dimension
    # beam_dim = 2 * num_tx * num_users  # Beamformer vector dimension
    # n_head = 8
    # n_layers = 5
    # T = 1
    # batch_size = 256 
    # learning_rate = 5e-4
    # weight_decay = 0.05
    # max_epoch = 200
    # sigma2 = 1.0  
    # SNR = 15
    # SNR_power = 10 ** (SNR / 10)
    # attn_pdrop = 0.0
    # resid_pdrop = 0.0
    # mlp_ratio = 4
    # subspace_dim = 4
    # pbar_size = 3000

    # # Example configuration where num_users = num_tx = 4.
    # num_users = 4
    # num_tx = 4
    # d_model = 64  # Transformer token dimension
    # beam_dim = 2 * num_tx * num_users  # Beamformer vector dimension
    # n_head = 4
    # n_layers = 4
    # T = 1
    # batch_size = 256 
    # learning_rate = 5e-4
    # weight_decay = 0.05
    # max_epoch = 200
    # sigma2 = 1.0  
    # SNR = 15
    # SNR_power = 10 ** (SNR / 10)
    # attn_pdrop = 0.0
    # resid_pdrop = 0.0
    # mlp_ratio = 4
    # subspace_dim = 4
    # pbar_size = 2000
    
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
        ini_sub_dim=ini_sub_dim
    )
    
    # Train the beamforming transformer.
    train_beamforming_transformer(config)