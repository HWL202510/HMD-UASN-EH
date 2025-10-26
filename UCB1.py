import numpy as np
from Environment import Env
import math

env = Env()

class UCB:
    def __init__(self):
        self.alpha = 2 #UCB1参数
        self.relay_num = env.RELAY_NUM #可供选择中继个数
        self.n = np.zeros(self.relay_num) #记录每个中级被选次数
        self.C = np.zeros(self.relay_num) #记录所选中继容量总值
        self.ucb = np.zeros(self.relay_num)

        self.ucb_max = 0    #ucb选出回报最大中继是哪个
        self.arm_max = 0    #真实最大回报中继

        self.p = []   #存储ucb选出最大中继
        self.P = []   #保存真实回值

        self.P_S = 1
        self.B = 10 #信道带宽

    #初始化参数
    def initialize(self,G_sr, G_rd, G_sd, Br):
        C = {}
        sinr_s_d = {}
        sinr_s_r = {}
        sinr_r_d = {}
        for i in range(self.relay_num):
            N = env.environment_noise(env.f, env.s, env.w)
            sinr_s_d = env.SINR_S_D(self.P_S, G_sd, N)
            sinr_s_r[i] = env.SINR_S_R(self.P_S, G_sr[i], N)
            sinr_r_d[i] = env.SINR_R_D(Br[i], G_rd[i], N)
            C[i] = 0.5 * self.B * np.log2(1 + sinr_s_d + (sinr_s_r[i] * sinr_r_d[i]) / (1 + sinr_s_r[i] + sinr_r_d[i]))
            self.C[i] += C[i]
            self.n[i] += 1
            self.ucb[i] = self.C[i] / self.n[i] + np.sqrt(self.alpha * math.log(sum(self.n) / self.n[i]))

        self.ucb_max = np.argmax(self.ucb)
        self.arm_max = np.argmax(self.ucb)

        return self.ucb[self.ucb_max]

    def update(self, G_sr, G_rd, G_sd, Br):
        N = env.environment_noise(env.f, env.s, env.w)
        sinr_s_r = env.SINR_S_R(self.P_S, G_sr, N)
        sinr_r_d= env.SINR_R_D(Br, G_rd, N)
        sinr_s_d = env.SINR_S_D(self.P_S, G_sd, N)
        C = 0.5 * self.B * np.log2(1 + sinr_s_d + (sinr_s_r * sinr_r_d) / (1 + sinr_s_r + sinr_r_d))

        self.P.append(C)

        arm = self.p[-1]

        self.C[arm] += self.P[-1]
        self.n[arm] += 1
        self.ucb[arm] = self.C[arm] / self.n[arm] + np.sqrt(self.alpha * math.log(sum(self.n) / self.n[arm]))

        return self.ucb[arm]

    def chose_relay(self, G_sr, G_rd, G_sd, Br, P):
        g_sr = G_sr
        g_sd = G_sd
        g_rd = G_rd
        P_r = P

        p = np.argmax(self.ucb)
        self.p.append(p)

        self.update(g_sr[p], g_rd[p], g_sd, P_r)

        return p, self.ucb[p]


    def brush(self,):
        self.n = np.zeros(self.relay_num) #记录每个中级被选次数
        self.C = np.zeros(self.relay_num) #记录所选中继容量总值
        self.ucb = np.zeros(self.relay_num)

        self.ucb_max = 0    #ucb选出回报最大中继是哪个
        self.arm_max = 0    #真实最大回报中继

        self.p = []   #存储ucb选出最大中继
        self.P = []   #保存真实回值





