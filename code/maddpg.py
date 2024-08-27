import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_speaker_listener_v4


class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class QValueNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device, sigma, tau):
        self.action_dim = action_dim
        #################### TODO ####################
        # 初始化actor和critic的网络和优化器
        # 初始化target_actor和target_critic的网络，
        # 并将参数设置为和actor和critic一致
        ##############################################
        self.device = device
        self.sigma = sigma
        self.tau = tau

    def take_action(self, state, explore=False):
        #################### TODO ####################
        # 根据state选择action
        # 如果explore为True，action需要加上高斯噪声
        # 高斯噪声的均值为0，标准差为self.sigma
        # action的值需要在[0, 1]之间
        ##############################################
        pass

    def soft_update(self):
        #################### TODO ####################
        # 更新target_actor和target_critic的参数
        # target = tau * actor + (1 - tau) * target_actor
        ##############################################
        pass


class MADDPG:
    def __init__(self, state_dims, action_dims, critic_input_dim, hidden_dim, actor_lr, critic_lr, device, gamma, sigma, tau):
        self.agents = []
        for i in range(len(state_dims)):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device, sigma, tau))
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.gamma = gamma

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore=False):
        return [
            agent.take_action(state, explore).cpu().detach().numpy().squeeze()
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        #################### TODO ####################
        # 计算critic loss
        ##############################################
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        #################### TODO ####################
        # 计算actor loss
        ##############################################
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update()


def wrap(l, agents):
    d = {}
    for i, agent in enumerate(agents):
        d[agent] = l[i]
    return d


def unwrap(d, agents):
    l = []
    for agent in agents:
        l.append(d[agent])
    return l 


def evaluate(maddpg, render=False, n_episode=10, episode_length=25):
    if render:
        env_test = simple_speaker_listener_v4.parallel_env(continuous_actions=True, render_mode="human")
    else:
        env_test = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
    env_test.reset()
    agents = env_test.agents
    returns = np.zeros(len(env_test.agents))
    for _ in range(n_episode):
        state, _ = env_test.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(unwrap(state, agents), explore=False)
            actions = wrap(actions, agents)
            next_state, reward, term, trunc, _ = env_test.step(actions)
            state = next_state
            for i, agent in enumerate(agents):
                returns[i] += reward[agent]
    for i in range(len(agents)):
        returns[i] /= n_episode
    return returns.tolist()


num_episodes = 10000
buffer_size = 1000000
episode_length = 25
hidden_dim = 128
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.99
sigma = 0.1
tau = 0.005
batch_size = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 100
update_ratio = 0.1
minimal_size = 10000

env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
env.reset()
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

state_dims = []
action_dims = []
agents = env.agents
for agent in agents:
    action_dims.append(env.action_space(agent).shape[0])
    state_dims.append(env.observation_space(agent).shape[0])
critic_input_dim = sum(state_dims) + sum(action_dims)

maddpg = MADDPG(state_dims, action_dims, critic_input_dim, hidden_dim, actor_lr, critic_lr, device, gamma, sigma, tau)

return_list = []
total_step = 0
for i_episode in range(num_episodes):
    state, _ = env.reset()
    for i in range(episode_length):
        actions = maddpg.take_action(unwrap(state, agents), explore=True)
        actions = wrap(actions, agents)
        next_state, reward, term, trunc, _ = env.step(actions)
        done = {agent: term[agent] or trunc[agent] for agent in agents}
        replay_buffer.add(state, actions, reward, next_state, done)
        state = next_state
        total_step += 1
        if replay_buffer.size(
        ) >= minimal_size and total_step % update_interval == 0:
            for i in range(int(update_interval * update_ratio)):
                sample = replay_buffer.sample(batch_size)

                def process_data(x):
                    processed = []
                    for i in range(len(agents)):
                        processed.append([])
                    for d in x:
                        for i, agent in enumerate(agents):
                            processed[i].append(d[agent])
                    for i in range(len(agents)):
                        processed[i] = torch.FloatTensor(np.array(processed[i])).to(device)
                    return processed

                sample = [process_data(x) for x in sample]
                for a_i in range(len(agents)):
                    maddpg.update(sample, a_i)
            maddpg.update_all_targets()
    if (i_episode + 1) % 200 == 0:
        ep_returns = evaluate(maddpg)
        return_list.append(ep_returns[0])
        print(f"Episode {i_episode+1} returns {ep_returns}")

env.close()

ep_returns = evaluate(maddpg, render=True)

return_array = np.array(return_list)
plt.figure()
plt.plot(
    np.arange(return_array.shape[0]) * 200,
    rl_utils.moving_average(return_array, 9))
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title(f"MADDPG on Speaker-Listener")
plt.savefig("MADDPG-Speaker-Listener.png")