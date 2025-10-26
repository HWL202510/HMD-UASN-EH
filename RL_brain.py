# # coding:utf-8
import numpy as np
import tensorflow as tf

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, a_bias, lr_a, lr_c, gamma, tau, batch_size):
        self.LR_A = lr_a
        self.LR_C = lr_c
        self.GAMMA = gamma
        self.TAU = tau
        self.BATCH_SIZE = batch_size

        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound, self.a_bias = a_dim, s_dim, a_bound, a_bias

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S, )
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        self.a_loss = - tf.reduce_mean(q)  # maximize the q

        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(self.a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + self.GAMMA * q_
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)

            self.ctrain = tf.train.RMSPropOptimizer(self.LR_C).minimize(self.td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

        self.cost_his_C = []
        self.cost_his_A = []

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, bt):  # bt:代表batch_memory
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim+1: self.s_dim + self.a_dim+1]  # 这里加1是因为DDPG做的决策a2前面还有DQN做的决策a1
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        _, self.cost_a = self.sess.run([self.atrain, self.a_loss], {self.S: bs})
        _, self.cost_c = self.sess.run([self.ctrain, self.td_error], {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        self.cost_his_A.append(self.cost_a)
        self.cost_his_C.append(self.cost_c)

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):

            net_1 = tf.layers.dense(s, 30, activation=tf.nn.relu6, name='l1', trainable=trainable)

            a = tf.layers.dense(net_1, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)

            a_scaled = tf.multiply(a, self.a_bound, name="scaled_a")
            return a_scaled #+ self.a_bias

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 32
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_1 = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_2 = tf.layers.dense(net_1, 16, activation=tf.nn.relu6, trainable=trainable)
            return tf.layers.dense(net_2, 1, trainable=trainable)  # Q(s,a)


class Memory(object):

    def __init__(self, memory_capacity, dim):
        self.MEMORY_CAPACITY = memory_capacity
        self.MEMORY = np.zeros((memory_capacity, dim), dtype=np.float32)
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.MEMORY[index, :] = transition
        self.pointer += 1

    def sample(self, size):  # size:表示batch_size
        if self.pointer > self.MEMORY_CAPACITY:
            sample_index = np.random.choice(self.MEMORY_CAPACITY, size=size)
        else:
            sample_index = np.random.choice(self.pointer, size=size)
        return self.MEMORY[sample_index, :]







