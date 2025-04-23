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

#############################################
# 1. System performance: sum rate for multi-user MISO
#############################################
def sum_rate(H, W, sigma2=1.0):
    """
    Compute average sum rate over batch.
    H: (B, K, N) complex channel
    W: (B, N, K) complex beamformer
    """
    prod = th.bmm(H, W)  # (B, K, K)
    signal = th.abs(th.diagonal(prod, dim1=-2, dim2=-1))**2
    interference = th.sum(th.abs(prod)**2, dim=-1) - signal
    SINR = signal / (interference + sigma2)
    return th.log2(1 + SINR).sum(dim=-1).mean()

#############################################
# 2. Cross-Attention block with Q/K/V
#############################################
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.out_proj = nn.Linear(d_model, d_model)

        # LayerNorms
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)

        # MLP block
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
        Cross-attention with separate Q, K, V.
        query: (B, L_q, d_model)
        key:   (B, L_k, d_model)
        value: (B, L_v, d_model), L_k == L_v
        """
        B, L_q, _ = query.size()
        _, L_k, _ = key.size()

        # LayerNorm
        q = self.ln_q(query)
        k = self.ln_q(key)
        v = self.ln_q(value)

        # Compute Q,K,V and reshape
        Q = self.q_proj(q).view(B, L_q, self.n_head, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(B, L_k, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Aggregate
        y = att @ V  # (B, n_head, L_q, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, L_q, self.n_head * self.head_dim)
        y = self.out_proj(y)
        y = self.resid_drop(y)

        # Residual + MLP
        out = query + y
        out = out + self.mlp(self.ln_out(out))
        return out

#############################################
# 3. Beamforming Transformer with contrastive feature
#############################################
class BeamformingTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.K = config.num_users
        self.N = config.num_tx
        d_model = config.d_model

        # Projections for Query/Key tokens
        self.ant_Q = nn.Linear(self.K, d_model)
        self.ant_K = nn.Linear(self.K, d_model)
        self.user_Q = nn.Linear(self.N, d_model)
        self.user_K = nn.Linear(self.N, d_model)

        # Projections for Value tokens (from H·W^T)
        self.ant_V = nn.Linear(self.K, d_model)
        self.user_V = nn.Linear(self.K, d_model)

        # Positional embeddings
        self.pos_ant = nn.Parameter(th.zeros(2*self.N, d_model))
        self.pos_user = nn.Parameter(th.zeros(2*self.K, d_model))

        # Cross-attention blocks
        self.attn_ant = CrossAttentionBlock(d_model, config.n_head, config.attn_pdrop, config.resid_pdrop)
        self.attn_user = CrossAttentionBlock(d_model, config.n_head, config.attn_pdrop, config.resid_pdrop)

        # Fusion & output
        total_tokens = 2*self.N + 2*self.K
        # self.out_mlp = nn.Sequential(
        #     nn.Linear(total_tokens * d_model, d_model),
        #     # nn.ReLU(),  
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Linear(d_model, config.beam_dim)
        # )
        self.out_mlp = nn.Sequential(
            nn.Linear(total_tokens * d_model, 4 * d_model),
            # nn.LayerNorm(4 * d_model),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(4 * d_model, 2 * d_model),
            # nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(2 * d_model, config.beam_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, H, W_prev, return_feature=False):
        """
        H: (B, K, N) complex
        W_prev: (B, K, N) complex
        return_feature: if True, also return x_fused
        """
        B = H.size(0)

        # Compute Prod = H · W_prev^T -> (B, K, K)
        Prod = th.bmm(H, W_prev.transpose(-1,-2))

        # Antenna-level tokens: columns of H, W_prev, Prod
        H_ant = th.cat([H.real.transpose(1,2), H.imag.transpose(1,2)], dim=1)  # (B,2N,K)
        W_ant = th.cat([W_prev.real.transpose(1,2), W_prev.imag.transpose(1,2)], dim=1)  # (B,2N,K)
        V_ant = th.cat([Prod.real.transpose(1,2), Prod.imag.transpose(1,2)], dim=1)  # (B,2K,K)

        # Project + add pos emb
        Q_ant = self.ant_Q(H_ant) + self.pos_ant.unsqueeze(0)
        K_ant = self.ant_K(W_ant) + self.pos_ant.unsqueeze(0)
        V_ant = self.ant_V(V_ant) + self.pos_ant.unsqueeze(0)

        x_ant = self.attn_ant(Q_ant, K_ant, V_ant)  # (B,2N,d_model)

        # User-level tokens: rows of H, W_prev, Prod
        H_usr = th.cat([H.real, H.imag], dim=1)   # (B,2K,N)
        W_usr = th.cat([W_prev.real, W_prev.imag], dim=1)  # (B,2K,N)
        V_usr = th.cat([Prod.real, Prod.imag], dim=1)      # (B,2K,K)

        Q_usr = self.user_Q(H_usr) + self.pos_user.unsqueeze(0)
        K_usr = self.user_K(W_usr) + self.pos_user.unsqueeze(0)
        V_usr = self.user_V(V_usr) + self.pos_user.unsqueeze(0)

        x_usr = self.attn_user(Q_usr, K_usr, V_usr)  # (B,2K,d_model)

        # Flatten & fuse
        x = th.cat([x_ant.view(B,-1), x_usr.view(B,-1)], dim=1)  # (B, total_tokens*d_model)
        W_next = self.out_mlp(x)
        W_next = W_next / W_next.norm(dim=1, keepdim=True)

        if return_feature:
            return W_next, x
        return W_next

#############################################
# 4. Channel dataset: random Gaussian
#############################################
class ChannelDataset(Dataset):
    def __init__(self, num_samples, num_users, num_tx, P, subspace_dim):
        self.num_samples = num_samples
        self.K = num_users
        self.N = num_tx
        self.P = P

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        # Real & imag Gaussian
        # scale = math.sqrt(2)/2
        scale = 1
        H_r = th.randn(self.K, self.N) * scale
        H_i = th.randn(self.K, self.N) * scale
        H = th.stack([H_r, H_i], dim=0) * (self.P**0.5)  # (2,K,N)
        return H

#############################################
# 5. Optimizer config
#############################################
def configure_optimizer(model, lr, wd):
    decay, nodecay = [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if any(x in n for x in ['weight','linear','conv']) and not any(x in n for x in ['ln','norm','bias']):
            decay.append(p)
        else:
            nodecay.append(p)
    return optim.AdamW([{'params':decay,'weight_decay':wd},{'params':nodecay,'weight_decay':0}], lr=lr)

#############################################
# 6. Training with contrastive loss
#############################################
def train_beamforming_transformer(config):
    # Data loader
    ds = ChannelDataset(config.pbar_size*config.batch_size,
                        config.num_users, config.num_tx,
                        config.SNR_power, config.subspace_dim)
    loader = DataLoader(ds, config.batch_size, shuffle=True)

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = BeamformingTransformer(config).to(device)
    opt = configure_optimizer(model, config.learning_rate, config.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(opt, lambda e: 1 - 0.3*(e/config.max_epoch))

    # Contrastive params
    lambda_init = 100
    temperature = 0.1
    # teacher = 0.0

    for epoch in range(config.max_epoch):
        model.train()
        lambda_cl = lambda_init * (1 - epoch/config.max_epoch)
        teacher = 1.0 if epoch < 2 else 0.0

        epoch_rate = 0
        pbar_batches = 0
        max_rate = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for H_xy in pbar:
            H_xy = H_xy.to(device)                     # (B,2,K,N)
            B = H_xy.size(0)
            H = H_xy[:,0] + 1j*H_xy[:,1]               # (B,K,N)

            # MMSE initialization
            W0,_ = compute_mmse_beamformer(H, config.num_users,
                                           config.num_tx, config.SNR_power,
                                           config.sigma2, device)
            W0 = W0.transpose(-1,-2).to(device)
            vec0 = th.cat([W0.real.reshape(B,-1), W0.imag.reshape(B,-1)],dim=1)
            vec0 = vec0 / vec0.norm(dim=1,keepdim=True)
            W_prev = (vec0[:,:config.num_tx*config.num_users].reshape(B,config.num_tx,config.num_users)
                      + 1j*vec0[:,config.num_tx*config.num_users:].reshape(B,config.num_tx,config.num_users))
            W_prev = W_prev.transpose(-1,-2)

            # Contrastive features
            W_pred, z_anchor = model(H, W_prev, return_feature=True)
            noise = 0.1 * th.randn_like(H_xy)
            Hn = H_xy + noise
            Hn = Hn[:,0] + 1j*Hn[:,1]
            _, z_pos = model(Hn, W_prev, return_feature=True)

            # Compute contrastive loss
            za = F.normalize(z_anchor, dim=1)
            zp = F.normalize(z_pos, dim=1)
            sim = th.mm(za, zp.T) / temperature  # (B,B)
            labels = th.arange(B, device=device)
            cl_loss = F.cross_entropy(sim, labels)

            # Iterative beamformer and rate/MSE
            total_rate = 0.0
            total_mse = 0.0
            for t in range(1, config.T+1):
                W_next = model(H, W_prev)
                W_mat = (W_next[:,:config.num_tx*config.num_users].reshape(B,config.num_tx,config.num_users)
                         + 1j*W_next[:,config.num_tx*config.num_users:].reshape(B,config.num_tx,config.num_users))
                rate = sum_rate(H, W_mat, sigma2=config.sigma2)
                total_rate += rate
                total_mse += F.mse_loss(W_next, vec0)

                W_prev = W_mat.transpose(-1,-2)

            loss_unsup = - total_rate / config.T
            loss_sup   = total_mse / config.T
            # print(f"unsup_loss: {loss_unsup.item():.4f}, con_loss: {cl_loss.item():.4f}")
            # bp()
            loss = (1-teacher)*(loss_unsup + lambda_cl*cl_loss) + (teacher*2000)*loss_sup

            opt.zero_grad()
            loss.backward()
            opt.step()

            ave_rate = total_rate.item() / config.T
            epoch_rate += ave_rate
            # rate_history.append(th.tensor(ave_rate))
            max_rate = max(max_rate, ave_rate)
            pbar_batches += 1
            pbar.set_description(f"Epoch {epoch+1}, Avg Rate: {ave_rate:.4f}, Max Rate: {max_rate:.4f}, unsup_loss: {loss_unsup.item():.4f}, con_loss: {lambda_cl * (cl_loss.item()):.4f}")

        scheduler.step()
        ave_pbar_rate = epoch_rate / pbar_batches
        # print(f"Epoch {epoch+1} completed. Learning Rate: {current_lr:.2e}")
        print(f"Epoch {epoch+1} Average Sum Rate: {ave_pbar_rate:.4f}")


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
    learning_rate = 1e-4
    weight_decay = 0.03
    max_epoch = 500
    sigma2 = 1.0  
    SNR = 15
    SNR_power = 10 ** (SNR/10) # SNR power in dB
    attn_pdrop = 0.0
    # resid_pdrop = 0.05
    # attn_pdrop = 0.0
    resid_pdrop = 0.0
    mlp_ratio = 4
    subspace_dim = 4
    pbar_size = 1000

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
        pbar_size=pbar_size
    )
    
    # Train the beamforming transformer.
    train_beamforming_transformer(config)