import os
import torch as th
from net_mimo import Policy_Net_MIMO
from env_mimo import MIMOEnv
import sys
def test_curriculum_learning(policy_net_mimo, device, K=4, N=4, episode_length=6):
    env_mimo = MIMOEnv(K=K, N=N, device=device, num_env=4096, episode_length=episode_length)
    sum_rate = th.zeros(100, env_mimo.episode_length, 5)
    test_P = [10 ** 0.5, 10 ** 1, 10 ** 1.5, 10 ** 2, 10 ** 2.5]
    with th.no_grad():
        for i_p in range(5):
            state = env_mimo.reset(test=True, test_P = test_P[i_p])
            while(1):
                action = policy_net_mimo(state)
                next_state, reward, done = env_mimo.step(action)
                sum_rate[:, env_mimo.num_steps-1, i_p] = reward.squeeze()
                state = next_state
                if done:
                    break
            print(f"SNR={(i_p+1)*5} dB: {sum_rate[:, :, i_p].max(dim=1)[0].mean().item():.3f}")
            
if __name__  == "__main__":
    N = 8   # number of antennas
    K = 8   # number of users
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO(mid_dim=1024, K=K, N=N).to(device)
    policy_net_mimo.load_state_dict(th.load(f'./{sys.argv[1]}_step.pth', map_location=th.device("cuda:0" if th.cuda.is_available() else "cpu")))
    test_curriculum_learning(policy_net_mimo, K=K, N=N, device=device, episode_length=int(sys.argv[1]))
