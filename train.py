import os
import torch as th
from net_mimo import Policy_Net_MIMO
from env_mimo import MIMOEnv
from tqdm import tqdm
from baseline_mmse import compute_mmse_beamformer
import sys
import matplotlib.pyplot as plt
# def train_curriculum_learning(policy_net_mimo, optimizer, save_path, device, K=4, N=4, P=10, noise_power=1, num_epochs=40000,
#                     num_epochs_per_subspace=1000, num_epochs_to_save_model=10000, episode_length=6, reward_change=0):
def train_curriculum_learning(policy_net_mimo, optimizer, device, K, N, P, num_epochs, num_epochs_per_subspace, episode_length, batch_size, noise_power=1, reward_change=0):
    env_mimo = MIMOEnv(K=K, N=N, P=P, noise_power=noise_power, device=device, num_env=batch_size, episode_length=episode_length)
    pbar = tqdm(range(num_epochs))
    sum_rate = th.zeros(500, env_mimo.episode_length, 3)
    test_P = [10 ** 1, 10 ** 1.5, 10 ** 2]
      
    # env_mimo.reset()
    # _, mmse_sum_rate= compute_mmse_beamformer(env_mimo.mat_H,  K=env_mimo.K, N=env_mimo.N, P=env_mimo.P, noise_power=1, device=env_mimo.device)

    rate_history_10db = []
    rate_history_15db = []
    rate_history_20db = []

    rate_mmse_10db = []
    rate_mmse_15db = []
    rate_mmse_20db = []
    
    
    T = env_mimo.episode_length
    for epoch in pbar:
        #state = env_mimo.reset()
        state, _ = [*env_mimo.reset()[:-1]], env_mimo.reset()[-1]
        obj_H = 0
        reward_list = []
        
        #### designed for object curriculum learning
        if(epoch < reward_change):
            rate_reward = False
        else:
            rate_reward = True

        while(1):
            action = policy_net_mimo(state)
            ## (self.mat_H, self.mat_W, self.P, HW.detach()), self.reward, self.done
            next_state, reward, done = env_mimo.step(action, rate_reward=rate_reward)
            reward_list.append(reward.mean())
            state = next_state
            if done:
                break
        
        for i in range(T):
            obj_H += reward_list[i]
        
        obj_H *= -1.0
        optimizer.zero_grad()
        obj_H.backward()
        optimizer.step()
        
        
        # if (epoch+1) % num_epochs_to_save_model == 0:
        #     th.save(policy_net_mimo.state_dict(), save_path + f"{epoch}.pth")
        if (epoch + 1) % num_epochs_per_subspace == 0 and env_mimo.subspace_dim <= 2 * K * N:
            env_mimo.subspace_dim += 4
        if (epoch) % 50 == 0:
            sum_rate_mmse_group = th.zeros(3)
            with th.no_grad():
                for i_p in range(3):
                    ### (self.mat_H, self.mat_W, self.P, HW), mmse_sum_rate
                    state, sum_rate_mmse = [*env_mimo.reset(test=True, test_P = test_P[i_p])[:-1]], env_mimo.reset(test=True, test_P = test_P[i_p])[-1] ## batch size = 100
                    sum_rate_mmse_group[i_p-1] = sum_rate_mmse.mean().item()
                    while(1):
                        action = policy_net_mimo(state)
                        next_state, reward, done = env_mimo.step(action)
                        sum_rate[:, env_mimo.episode_length-1, i_p] = reward.squeeze()
                        state = next_state
                        if done:
                            break
                
                rate_history_10db.append(sum_rate[:, :, 0].max(dim=1)[0].mean().item())
                rate_history_15db.append(sum_rate[:, :, 1].max(dim=1)[0].mean().item())
                rate_history_20db.append(sum_rate[:, :, 2].max(dim=1)[0].mean().item())
                rate_mmse_10db.append(sum_rate_mmse_group[0])
                rate_mmse_15db.append(sum_rate_mmse_group[1])
                rate_mmse_20db.append(sum_rate_mmse_group[2])

                #pbar.set_description(f"id: {epoch}|SNR=5:{sum_rate[:, :, 0].max(dim=1)[0].mean():.6f}|SNR=10:{sum_rate[:, :, 1].max(dim=1)[0].mean():.6f}|SNR15:{sum_rate[:, :, 2].max(dim=1)[0].mean():.6f}|SNR=20: {sum_rate[:, :, 3].max(dim=1)[0].mean():.6f}|SNR=25: {sum_rate[:, :, 4].max(dim=1)[0].mean():.6f}|training_loss: {obj_H.mean().item() / env_mimo.episode_length:.6f}|memory: {th.cuda.memory_allocated():3d}")
                pbar.set_description(f"id: {epoch}|SNR=10:{sum_rate[:, :, 0].max(dim=1)[0].mean():.2f},{sum_rate_mmse_group[0]:.2f}|SNR15:{sum_rate[:, :, 1].max(dim=1)[0].mean():.2f},{sum_rate_mmse_group[1]:.2f}|SNR=20: {sum_rate[:, :, 2].max(dim=1)[0].mean():.2f},{sum_rate_mmse_group[2]:.2f}|training_loss: {obj_H.mean().item() / env_mimo.episode_length:.3f}")

    return rate_history_10db, rate_history_15db, rate_history_20db, rate_mmse_10db, rate_mmse_15db, rate_mmse_20db   

def get_cwd(env_name):
    file_list = os.listdir()
    if env_name not in file_list:
        os.mkdir(env_name)
    file_list = os.listdir('./{}/'.format(env_name))
    max_exp_id = 0
    for exp_id in file_list:
        if int(exp_id) + 1 > max_exp_id:
            max_exp_id = int(exp_id) + 1
    os.mkdir('./{}/{}/'.format(env_name, max_exp_id))
    return f"./{env_name}/{max_exp_id}/"

if __name__  == "__main__":
    N = 16  # number of antennas
    K = 16   # number of users
    SNR = 15
    P = 10 ** (SNR / 10)
    num_epochs = 100000
    num_epochs_per_subspace = 500
    episode_length = 5
    batch_size = 1024
    noise_power = 1
    mid_dim = 512
    learning_rate = 5e-5
    cwd = f"RANDOM_H_CL_REINFORCE_N{N}K{K}SNR{SNR}"
    config = {
        'method': 'REINFORCE',
    }
    env_name = f"RANDOM_N{N}K{K}SNR{SNR}_mimo_beamforming"
    save_path = get_cwd(env_name)
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO(mid_dim=mid_dim, K=K, N=N, P=P).to(device)
    optimizer = th.optim.Adam(policy_net_mimo.parameters(), lr=learning_rate)
    # optimizer = th.optim.RMSprop(policy_net_mimo.parameters(), lr=learning_rate)
    try:
        ### policy_net_mimo, optimizer, device, K, N, P, num_epochs, num_epochs_per_subspace, episode_length, batch_size, noise_power=1, reward_change=0
        rate_history_10db, rate_history_15db, rate_history_20db, rate_mmse_10db, rate_mmse_15db, rate_mmse_20db= train_curriculum_learning(policy_net_mimo, optimizer, device=device, K=K, N=N, P=P, num_epochs=num_epochs, num_epochs_per_subspace=num_epochs_per_subspace, episode_length=episode_length, batch_size=batch_size, noise_power=1, reward_change=0)
        #train_curriculum_learning(policy_net_mimo, optimizer, K=K, N=N, save_path=save_path, device=device, P=P, noise_power=noise_power, episode_length=6)
        history_idx = range(0, len(rate_history_10db),1)
        th.save(policy_net_mimo.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
        print(f"saved at " + save_path)
    except KeyboardInterrupt:
        th.save(policy_net_mimo.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
        print(f"saved at " + save_path)
        exit()

    plt.figure(figsize=(10, 6))
    plt.plot(history_idx, rate_history_10db, marker='o', linestyle='-', color='r', label='10db')
    plt.plot(history_idx, rate_history_15db, marker='o', linestyle='-', color='g', label='15db')
    plt.plot(history_idx, rate_history_20db, marker='o', linestyle='-', color='c', label='20db')
    plt.plot(history_idx, rate_mmse_10db, marker='*', linestyle='-', color='r', label='10db,mmse')
    plt.plot(history_idx, rate_mmse_15db, marker='*', linestyle='-', color='g', label='15db,mmse')
    plt.plot(history_idx, rate_mmse_20db, marker='*', linestyle='-', color='c', label='20db,mmse')
    plt.xlabel('test epoch')
    plt.ylabel('sum rate')
    plt.title(f'Sum Rate vs Test Epoch')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

