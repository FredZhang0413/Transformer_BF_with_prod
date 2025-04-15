import torch as th
import numpy as np
import pickle as pkl
from baseline_mmse import compute_mmse_beamformer
class MIMOEnv():
    def __init__(self, K, N, P, episode_length, num_env, noise_power=1, device=th.device("cuda:0" if th.cuda.is_available() else "cpu")):
        self.N = N  # #antennas
        self.K = K  # #users
        self.P = P  # Power
        self.noise_power = noise_power
        self.device = device
        ### basis vectors for subspace curriculum learning
        self.basis_vectors, _ = th.linalg.qr(th.rand(2 * self.K * self.N, 2 * self.K * self.N, dtype=th.float, device=self.device))
        self.subspace_dim = 4
        # self.subspace_dim = 2 * self.K * self.N
        self.num_env = num_env
        self.episode_length = episode_length
        # with open("./K8N8Samples=100.pkl", 'rb') as f:
        self.test_H = th.randn(500, self.K, self.N, dtype=th.cfloat, device=self.device)

    def reset(self, test=False, test_P = None): ### default training state: hybrid P, subspace dim
        # self.test = False

        if self.subspace_dim < 2 * self.K * self.N:
            self.vec_H = self.generate_channel_batch(self.N, self.K, self.num_env, self.subspace_dim, self.basis_vectors)
        else:
            self.vec_H = th.randn(self.num_env, 2 * self.K * self.N, dtype=th.cfloat, device=self.device)

        if test:
            self.test = True
            self.test_P = test_P
            self.mat_H = self.test_H * np.sqrt(test_P)

        else:
            self.mat_H = (self.vec_H[:, :self.K * self.N] + self.vec_H[:, self.K * self.N:] * 1.j).reshape(-1, self.K, self.N)
            # indices = np.random.choice(self.mat_H.shape[0], self.num_env, replace=False)
            # self.mat_H = self.mat_H[indices,:,:]
            ### training with hybrid SNR: 10/15/20 dB
            self.mat_H[:self.mat_H.shape[0] // 3] *= np.sqrt(10 ** 1)
            self.mat_H[self.mat_H.shape[0] // 3:2 * self.mat_H.shape[0] // 3] *= np.sqrt(10 ** 1.5)
            self.mat_H[2 * self.mat_H.shape[0] // 3:] *= np.sqrt(10 ** 2)
            self.mat_H = self.mat_H / np.sqrt(2)
        self.mat_W, mmse_sum_rate= compute_mmse_beamformer(self.mat_H, K=self.K, N=self.N, P=self.P, noise_power=1, device=self.device)
        HW = th.bmm(self.mat_H, self.mat_W.transpose(-1, -2))
        self.num_steps = 0
        self.done = False
        #### (H,W,HW) as a input pair later
        return self.mat_H, self.mat_W, self.P, HW, mmse_sum_rate
    
    def step(self, action, rate_reward=True):
        ### compute the reward (MISO)
        HW = th.bmm(self.mat_H, action.transpose(-1, -2))
        S = th.abs(th.diagonal(HW, dim1=-2, dim2=-1))**2
        I = th.sum(th.abs(HW)**2, dim=-1) - th.abs(th.diagonal(HW, dim1=-2, dim2=-1))**2
        N = 1
        SINR = S/(I+N)
        if(rate_reward==True):
            self.reward = th.log2(1+SINR).sum(dim=-1).unsqueeze(-1)
        elif(rate_reward==False):
            self.reward = self.get_mmse_reward(HW)
        
        self.mat_W = action.detach()
        ### L2O episode progresses
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.mat_H, self.mat_W, self.P, HW.detach()), self.reward, self.done

    #### generate a batch of channels based on a certain subspace dimension
    def generate_channel_batch(self, N, K, batch_size, subspace_dim, basis_vectors):
        coordinates = th.randn(batch_size, subspace_dim, 1, device=self.device)
        basis_vectors_batch = basis_vectors[:subspace_dim].T.repeat(batch_size, 1).reshape(-1, 2 * K * N, subspace_dim)
        vec_channel = th.bmm(basis_vectors_batch, coordinates).reshape(-1, 2 * K * N)
        return vec_channel
       
    def get_reward(self, H, W):
        HW = th.matmul(H, W.T)
        S = th.abs(HW.diag()) ** 2
        I = th.sum(th.abs(HW)**2, dim=-1) - th.abs(HW.diag()) ** 2
        N = 1
        SINR = S / (I + N)
        reward = th.log2(1 + SINR).sum(dim=-1)
        return reward, HW
    
    def get_mmse_reward(self,HW):
        fro_norm = th.norm(HW, p='fro',dim=(1,2)) ** 2
        trace_real = th.stack([th.trace(HW).real for HW in HW])
        reward = - (fro_norm - trace_real)
        return reward.unsqueeze(-1)