from Environment import Env
from UCB1 import UCB
from RL_brain import DDPG, Memory
import numpy as np
import matplotlib.pyplot as plt

env = Env()
P_R_min = env.P_R[0]  # 得到中继最小发射功率
P_R_max = env.P_R[1]  # 得到中继最大发射功率
action_P = 3

# -------UCB 所用参数----------#
relay_num = env.RELAY_NUM

# --------------- DDPG 所用参数-------------#
s_dim = 1 + relay_num
a_dim = 1
a_bound = (P_R_max-P_R_min)/2.0
a_bias = P_R_max-a_bound
lr_a = 0.00089   # earning rate for actor
lr_c = 0.0065  # learning rate for critic
GAMMA = 0.9    # reward discount
TAU = 0.1      # soft replacement
BATCH_SIZE = 64
var = 1   # control explorationd

# ---------replay buffer---------#
memory_capacity = 5000
dim = s_dim*2+2+1

# --------训练轮数等信息------#
MAX_STEPS = 180    # 每一轮最大步数
MAX_EPISODES = 500   # 训练多少轮

UCB1 = UCB()

RL_DDPG = DDPG(a_dim=a_dim, s_dim=s_dim,
               a_bound=a_bound, a_bias=a_bias,
               lr_a=lr_a, lr_c=lr_c,
               gamma=GAMMA, tau=TAU,
               batch_size=BATCH_SIZE)

MEMORY = Memory(memory_capacity=memory_capacity, dim=dim)

for episode in range(MAX_EPISODES):
    s0, sr, rd, sd, Br = env.initialize_state()

    if episode == 0:
        UCB_relay = UCB1.initialize(sr, rd, sd, Br)
    else:
        UCB1.brush()
        UCB_relay = UCB1.initialize(sr, rd, sd, Br)

    s0 = np.append(s0, UCB_relay)
    s = s0
    for t in range(MAX_STEPS):
        action_UCB, UCB_relay = UCB1.chose_relay(sr, rd, sd, Br, action_P)
        if MEMORY.pointer > MAX_STEPS:
            action_DDPG = RL_DDPG.choose_action(s)
            action_DDPG = action_DDPG+a_bias
            action_DDPG = np.clip(np.random.normal(action_DDPG, var), P_R_min, P_R_max)
        else:
            action_DDPG = np.clip(np.random.normal(loc=3.5, scale=2.5), P_R_min, P_R_max)

        action_P = action_DDPG
        action = np.array([action_UCB, action_DDPG], dtype=float)

        s0_, reward, done, sr, rd, sd = env.step(action, t)

        s0_ = np.append(s0_, UCB_relay)
        s_ = s0_
        Br = env.B_r

        MEMORY.store_transition(s, action, reward, s_)

        batch_memory = MEMORY.sample(BATCH_SIZE)

        if MEMORY.pointer > 5000:
            var *= 0.9995   # decay the action randomnessx
            RL_DDPG.learn(batch_memory)

        s = s_

    if episode % 100 == 0:
        print("第 ", episode, " 轮训练结束")
    print(env.record_c[episode])

plt.plot(np.arange(len(env.record_c)), env.record_c)
plt.xlabel('training_steps')
plt.ylabel('capacity')
plt.legend()
plt.show()

plt.plot(np.arange(len(env.record_interrupt_all)), env.record_interrupt_all, label='all_interrupt')
plt.legend()
plt.show()


