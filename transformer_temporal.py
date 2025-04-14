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
# def step(self, action, rate_reward=True):
#         ### compute the reward (MISO)
#         HW = th.bmm(self.mat_H, action.transpose(-1, -2))
#         S = th.abs(th.diagonal(HW, dim1=-2, dim2=-1))**2
#         I = th.sum(th.abs(HW)**2, dim=-1) - th.abs(th.diagonal(HW, dim1=-2, dim2=-1))**2
#         N = 1
#         SINR = S/(I+N)
#         if(rate_reward==True):
#             self.reward = th.log2(1+SINR).sum(dim=-1).unsqueeze(-1)
#         elif(rate_reward==False):
#             self.reward = self.get_mmse_reward(HW)
def sum_rate(H, W, sigma2=1.0):
    prod = th.bmm(H, W)  # (B, num_users, num_users)
    signal_power = th.abs(th.diagonal(prod, dim1=-2, dim2=-1))**2
    interference = th.sum(th.abs(prod)**2, dim=-1) - signal_power
    N = sigma2
    SINR = signal_power / (interference + N) # (B, num_users)
    reward = th.log2(1 + SINR).sum(dim=-1).mean()
    return reward

#############################################
# 2. Encoders
#############################################

#### 'CNN + pooling + FC' for state encoder (vec(H) --> h)
class StateEncoder(nn.Module): ### Layer normalization can be applied to stabilize training
    def __init__(self, d_state):
        super(StateEncoder, self).__init__()
        self.conv = nn.Sequential(
            # Input has 2 channels for real and imaginary parts
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU()
            # nn.AdaptiveAvgPool2d((1,1)) ### double-check whether it's beneficial
        )
        self.fc = nn.Linear(128* num_users* num_tx, d_state)
        
    def forward(self, H):
        # H shape: (B, 2, num_users, num_tx)
        # 2 channels: real and imaginary parts
        x = self.conv(H)
        x = x.view(x.size(0), -1)
        return self.fc(x) ### (B, d_state)

#### 'FC' for action encoder (vec(W) --> w)
class ActionEncoder(nn.Module):
    def __init__(self, beam_dim, d_action):
        super(ActionEncoder, self).__init__()
        self.fc = nn.Linear(beam_dim, d_action)
        
    def forward(self, W):
        return self.fc(W) ### (B, d_action)

#### 'FC' for token projection ([h ; w] --> v)
class TokenProjector(nn.Module):
    def __init__(self, d_state, d_action, d_model):
        super(TokenProjector, self).__init__()
        self.fc = nn.Linear(d_state + d_action, d_model) ### (B, d_model)
        
    def forward(self, h, a):
        x = th.cat([h, a], dim=-1)
        return self.fc(x) 

#############################################
# 3. Transformer model
#############################################
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, attn_pdrop, resid_pdrop, block_size):
        super(CausalSelfAttention, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", th.tril(th.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))
    
    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio, attn_pdrop, resid_pdrop, block_size):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, block_size=block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(), ## fit for the transformer
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
    
    ### residual connection + layer normalization
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

#############################################
# 4. Beamforming Transformer model
#############################################
class BeamformingTransformer(nn.Module):
    def __init__(self, config):
        super(BeamformingTransformer, self).__init__()
        self.config = config
        self.state_encoder = StateEncoder(config.d_state)
        self.action_encoder = ActionEncoder(config.beam_dim, config.d_action)
        self.token_projector = TokenProjector(config.d_state, config.d_action, config.d_model)
        self.pos_emb = nn.Parameter(th.zeros(config.max_seq_length, config.d_model))
        self.blocks = nn.Sequential(*[
            TransformerBlock(config.d_model, config.n_head, mlp_ratio=config.mlp_ratio, attn_pdrop=config.attn_pdrop, resid_pdrop=config.resid_pdrop, block_size=config.max_seq_length)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.action_decoder = nn.Linear(config.d_model, config.beam_dim)
        self.max_seq_length = config.max_seq_length

        # Apply weight initialization
        self.apply(self._init_weights)
        
        # Optional: Log the number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    #### adopt different initialization for different types of layers
    def _init_weights(self, module):
        """Initialize the weights - important for transformer training stability"""
        if isinstance(module, nn.Linear):
            # Standard initialization for linear layers
            th.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                th.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # Layer normalization: bias=0, weight=1
            th.nn.init.zeros_(module.bias)
            th.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            # Initialization for Conv2d layers
            th.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                th.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            th.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    #### prediction based on historical solutions and the current state
    def forward(self, H, beam_history):
        B, t, _ = beam_history.shape  ### B: batch size, t: time steps
        h_embed = self.state_encoder(H)
        beam_history_flat = beam_history.reshape(B * t, -1)  ##(B*t, beam_dim)
        beam_embed = self.action_encoder(beam_history_flat)
        beam_embed = beam_embed.reshape(B, t, -1) ##(B, t, d_action)
        h_repeat = h_embed.unsqueeze(1).repeat(1, t, 1) ##(B, t, d_state)
        tokens = self.token_projector(h_repeat, beam_embed) ##(B, t, d_model)
        pos_emb = self.pos_emb[:t, :].unsqueeze(0) ##(1, t, d_model)
        tokens = tokens + pos_emb ##(B, t, d_model)
        x = self.blocks(tokens) ##(B, t, d_model)
        x = self.ln_f(x) 
        x_last = x[:, -1, :] ##(B, d_model)
        W_next = self.action_decoder(x_last) ##(B, beam_dim), still flattened
        norm_W_next = th.norm(W_next, dim=1, keepdim=True)
        # W_next = W_next * (math.sqrt(self.config.SNR_power) / norm_W_next)
        W_next = W_next / (norm_W_next)
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
        coordinates = th.randn(self.subspace_dim, 1)
        basis_vectors_subset = self.basis_vectors[:self.subspace_dim].T
        vec_channel = th.matmul(basis_vectors_subset, coordinates).reshape(2 * self.num_users * self.num_tx)
        H_real = vec_channel[:self.num_tx * self.num_users].reshape(self.num_users, self.num_tx) ### (num_users, num_tx)
        H_imag = vec_channel[self.num_tx * self.num_users:].reshape(self.num_users, self.num_tx) ### (num_users, num_tx)
        H_complex = H_real + 1j * H_imag
        norm_H_complex = th.sum(th.abs(H_complex)**2)
        SNR_power = self.num_users*self.num_tx*self.P
        H_real = H_real * th.sqrt(SNR_power / norm_H_complex) ### (num_users, num_tx)
        H_imag = H_imag * th.sqrt(SNR_power / norm_H_complex) ### (num_users, num_tx)
        H_combined = th.stack([H_real, H_imag], dim=0) ### Shape: (2, num_users, num_tx)
        # H_combined = (vec_channel[:,:self.num_tx * self.num_users].reshape(-1, self.num_users, self.num_tx) + 1j * vec_channel[:,self.num_tx * self.num_users:].reshape(-1, self.num_users, self.num_tx)) ### (num_samples, num_users, num_tx)

        # # Generate complex channel matrix (real + imaginary parts)
        # H_real = np.random.randn(self.num_users, self.num_tx).astype(np.float32)
        # H_imag = np.random.randn(self.num_users, self.num_tx).astype(np.float32)
        # # Stack real and imaginary parts
        # H_combined = np.stack([H_real, H_imag], axis=0)*((self.P/2)**0.5)  ### Shape: (2, num_users, num_tx)       
 
        return th.tensor(H_combined) ### transform to tensor

#############################################
# 6. Optimizer configuration
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
# 7. Training routine
#############################################
def train_beamforming_transformer(config):
    """
    Train the beamforming transformer with the given configuration.
    Curriculum learning: at the beginning of each epoch, subspace_dim is increased by: (2*num_users*num_tx) / max_epoch
    """

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = BeamformingTransformer(config).to(device)
    optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)
    # optimizer = th.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    global mmse_rate_printed
    mmse_rate_printed = False

    model.train() ### set model to training mode
    # ---------------------------
    # Curriculum learning setup:
    # ---------------------------
    initial_subspace_dim = config.ini_sub_dim
    cl_increment = config.ini_sub_dim

    for epoch in range(config.max_epoch):

        # current_subspace_dim = initial_subspace_dim + epoch * cl_increment
        current_subspace_dim = min(initial_subspace_dim + epoch * cl_increment, 2*config.num_users * config.num_tx)
        print(f"Current subspace dimension: {current_subspace_dim}")
        dataset = ChannelDataset(num_samples=1000*config.batch_size, num_users=config.num_users, num_tx=config.num_tx, P=config.SNR_power, subspace_dim=current_subspace_dim)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        epoch_loss = 0
        epoch_rate = 0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            H = batch.to(device) ### (B, 2, num_users, num_tx)
            batch_size = H.size(0)
            H_mat = H[:,0,:,:] + 1j * H[:,1,:,:] ### (B, num_users, num_tx)
            W_history = [] ### historical solutions, gradually increased

            # W0 = th.randn(batch_size, config.beam_dim, device=device) # 1. Random initialization

            W0, _ = compute_mmse_beamformer(H_mat, config.num_users, config.num_tx, config.SNR_power, config.sigma2, device) # 2. MMSE initialization
            W0 = W0.transpose(-1, -2).to(device) 
            vec_w0 = th.cat((th.real(W0).reshape(-1,config.num_tx*config.num_users), th.imag(W0).reshape(-1,config.num_tx*config.num_users)), dim=-1).reshape(-1, 2*config.num_tx*config.num_users) # (train_batches, 2*Ntx*Nrx)
            
            norm_W0 = th.norm(vec_w0, dim=1, keepdim=True)
            # normlized_W0 = vec_w0 * (math.sqrt(config.SNR_power) / norm_W0) ### normalize the initial solution
            normlized_W0 = vec_w0 / (norm_W0) ### normalize the initial solution
            W_mat_0 = normlized_W0[:,:config.num_tx * config.num_users].reshape(-1,config.num_tx,config.num_users) + 1j * normlized_W0[:,config.num_tx * config.num_users:].reshape(-1,config.num_tx,config.num_users) ### (B, num_tx, num_users)
            rate_0 = sum_rate(H_mat, W_mat_0, sigma2=config.sigma2)
            # w = th.randn((config.batch_size, 8,8), dtype=th.cfloat).to(device)
            # w = w/w.norm(dim=(1,2), keepdim=True)
            # rate_0 = sum_rate(H_mat, w, sigma2=config.sigma2)
            if not mmse_rate_printed:
                mmse_rate_printed = True
                print(f"MMSE Rate: {rate_0.item():.4f}")
            # bp()
            W_history.append(normlized_W0)
            total_rate = 0
            max_rate = 0
            for t in range(1, config.T + 1):
                beam_history = th.stack(W_history, dim=1) ### (B, t, beam_dim)
                W_next = model(H, beam_history) ### (B, beam_dim)
                W_history.append(W_next)
                #### H and W are both complex
                W_mat = W_next[:,:config.num_tx * config.num_users].reshape(-1,config.num_tx,config.num_users) + 1j * W_next[:,config.num_tx * config.num_users:].reshape(-1,config.num_tx,config.num_users) ### (B, num_tx, num_users)
                rate = sum_rate(H_mat, W_mat, sigma2=config.sigma2)
                total_rate += rate
                max_rate = max(max_rate, rate.item())
            loss = - total_rate / (config.T)
            optimizer.zero_grad()
            loss.backward()
            # th.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_rate += total_rate.item() / (config.T * batch_size)
            num_batches += 1
            pbar.set_description(f"Epoch {epoch+1}, Avg Rate: {(total_rate.item()/(config.T)):.4f}, Max Rate: {max_rate:.4f}")
        
        # print(f"Epoch {epoch+1} Average Loss: {epoch_loss/num_batches:.4f}, Average Sum Rate: {epoch_rate/num_batches:.4f}")

#############################################
# 8. Config class
#############################################
class BeamformerTransformerConfig:
    def __init__(self, **kwargs):
        self.d_state = kwargs['d_state']  # State embedding dimension
        self.d_action = kwargs['d_action']  # Action embedding dimension
        self.d_model = kwargs['d_model']  # Transformer model dimension
        self.beam_dim = kwargs['beam_dim']  # Beamformer dimension
        self.n_head = kwargs['n_head'] # Number of attention heads
        self.n_layers = kwargs['n_layers'] # Number of transformer layers
        self.max_seq_length = kwargs['max_seq_length'] # Maximum sequence length
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self.grad_norm_clip = kwargs['grad_norm_clip']
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
        self.ini_sub_dim = kwargs['ini_sub_dim'] 


if __name__ == "__main__":
    # Set all parameters in the main function
    num_users = 16
    num_tx = 16
    d_state = 16 # State embedding dimension
    d_action = 16 # Action embedding dimension
    d_model = 32 # Transformer single-token dimension
    beam_dim = 2*num_tx*num_users # Beamformer vector dimension
    n_head = 8 # Number of attention heads
    n_layers = 4 # Number of transformer layers
    T = 3 # Number of time steps
    max_seq_length = T+1 # Maximum sequence length (episodes of L2O)
    batch_size = 256 
    learning_rate = 5e-5
    grad_norm_clip = 1.0
    weight_decay = 0.1
    # max_epoch = 50
    sigma2 = 1.0  
    SNR = 15
    SNR_power = 10 ** (SNR/10) # SNR power in dB
    attn_pdrop = 0.05
    resid_pdrop = 0.05
    mlp_ratio = 4
    ini_sub_dim = 2
    max_epoch = (2*num_users*num_tx) // ini_sub_dim

    
    # Create config object with parameters
    config = BeamformerTransformerConfig(
        d_state=d_state,
        d_action=d_action,
        d_model=d_model,
        beam_dim=beam_dim,
        n_head=n_head,
        n_layers=n_layers,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        grad_norm_clip=grad_norm_clip,
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
        ini_sub_dim = ini_sub_dim
    )
    
    # Train the model with the given config
    train_beamforming_transformer(config)


### to do: 
### 1. normalize the H(Gaussian sampled, no need) and W (SNR power)
### 2. add the curriculum learning for the subspace dimension
### 3. check the parameters for the transformer model
### 4. discuss whether the pooling in state encoder is correct
### 5. check the learning rate, whether is should be dynamically adjusted