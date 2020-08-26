import numpy as np
import tensorflow as tf
from IHiterEnv.parameter import *

np.random.seed(1)
tf.set_random_seed(1)

class DuelingDQN:
    def __init__(self,
                ActionDim = TP.ActionDim,
                StateDim = TP.StateDim,
                LearningRate = 0.01,
                RewardDecay=0.9,
                eGreedy=0.1,
                ReplaceTargetIter=1000,
                MemorySize=10000,
                BatchSize=64,
                eGreedyDecay=0.9,
                train_dir=None):

        self.ActionDim = ActionDim
        self.StateDim = StateDim

        self.Gamma = RewardDecay

        self.LearningRate = LearningRate
        self.EpsilonMin = eGreedy
        self.ReplaceTargetIter = ReplaceTargetIter
        self.MemorySize = MemorySize
        self.BatchSize = BatchSize
        self.eGreedyDecay = eGreedyDecay
        self.Epsilon = 0.98
        
        # 文件保存部分
        self.train_dir = train_dir
        self.checkpoints_dir = self.train_dir + 'CheckpointsFile/'
        self.summary_dir = self.train_dir + 'Summary/'

        # total learning step
        self.LearnStepCounter = 0

        # initialize zero memory [state, action, reward, next_state]
        self.memory = np.zeros((self.MemorySize, StateDim * 2 + ActionDim + 1))

        # consist of [TargetNet, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='TargetNet')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='EvalNet')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


    def _build_net(self):
        # ------------------ all inputs ------------------------
        # 
        self.state = tf.placeholder(tf.float32, [None, self.StateDim], name='state')
        self.next_state = tf.placeholder(tf.float32, [None, self.StateDim], name='nextstate')
        self.reward = tf.placeholder(tf.float32, [None, ], name='reward')  # input Reward

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        def build_layers(state, units, w_initializer, b_initializer):
            with tf.variable_scope('basic_net'):
                layer_1 = tf.layers.dense(state, units, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='layer_1')
                layer_2 = tf.layers.dense(layer_1, units, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='layer_2')

            
            with tf.variable_scope('value'):
                value_1 = tf.layers.dense(layer_2, units, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='value_1')
                value_out = tf.layers.dense(value_1, 1, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='value_out')                

            with tf.variable_scope('advantage'):
                advantage_1 = tf.layers.dense(layer_2, units, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='advantage_1')
                advantage_2 = tf.layers.dense(advantage_1, self.ActionDim, None, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='advantage_2')
                advantage_out = advantage_2 - tf.reduce_mean(advantage_2, axis=1, keep_dims=True)

            # Q = V(s) + A(s, a)
            with tf.variable_scope('Q'):
                out = value_out + advantage_out 
            return out

        # ------------------ build evaluate Net ------------------
        with tf.variable_scope('EvalNet'):
           self.q_eval = build_layers(self.state, 50, w_initializer, b_initializer)

        # ------------------ build target net ------------------
        with tf.variable_scope('TargetNet'):
            self.q_next = build_layers(self.next_state, 50, w_initializer, b_initializer)

        with tf.variable_scope('q_target'):
            self.q_target = tf.stop_gradient(self.reward + self.Gamma * tf.reduce_max(self.q_next, axis=1))

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, tf.reduce_max(self.q_eval, axis=1)))
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('reward', tf.reduce_mean(self.reward, axis=0))
        self.merged = tf.summary.merge_all()

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.LearningRate).minimize(self.loss)


    def StoreTransition(self, state, action, reward, next_state):
        '''
            存储记忆库
            
            @ state: [4x8+6,]
            @ action: [17x4,]
            @ reward: [2,]
        '''
        if not hasattr(self, 'MemoryCounter'):
            self.MemoryCounter = 0
        transition = np.hstack((state, action, reward, next_state))
        # replace the old memory with new memory
        index = self.MemoryCounter % self.MemorySize

        self.memory[index, :] = np.hstack((state, action, reward, next_state))
        self.MemoryCounter += 1


    def TrainDicision(self, State):
        '''
            agent在训练的时候做出action，会有一部分几率进行随机探索

            @ State：[38,]

            return：直接将网络的原始action数据进行输出[2x34,]
        '''
        # 小于self.Epsilon就随机探索
        if np.random.uniform() < self.Epsilon:
            State = State[np.newaxis, :]            
            # forward feed the observation and get q value for every actions
            ActionRaw = self.sess.run(self.q_eval, feed_dict={self.state: State}).reshape(-1)
        else:
            ActionRaw = np.random.randn(self.ActionDim)
        return ActionRaw


    def EvalDicision(self, State):
        State = State[np.newaxis, :]
        ActionRaw = self.sess.run(self.q_eval, feed_dict={self.state: State})
        return ActionRaw


    def Learn(self):
        # check to replace target parameters
        if self.LearnStepCounter % self.ReplaceTargetIter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.MemoryCounter > self.MemorySize:
            sample_index = np.random.choice(self.MemorySize, size=self.BatchSize)
        else:
            # 数据集还没有满
            sample_index = np.random.choice(self.MemoryCounter, size=self.BatchSize)
        batch_memory = self.memory[sample_index, :]

        # 第一方的训练
        _, summary, loss = self.sess.run([self._train_op, self.merged, self.loss],
                        feed_dict={
                        self.state: batch_memory[:, :self.StateDim],
                        self.reward: batch_memory[:, self.StateDim],
                        self.next_state: batch_memory[:, -self.StateDim:]})     

        # increasing Increment
        self.Epsilon = max(self.Epsilon - self.eGreedyDecay, self.EpsilonMin)
        if self.LearnStepCounter % 10 == 0:
            if self.LearnStepCounter % 10000 == 0:
                print('Train Times : ', self.LearnStepCounter, ' cost : ', loss)
            self.writer.add_summary(summary, self.LearnStepCounter)
        self.LearnStepCounter += 1
        
    def Save(self, step):
        dir_name = self.checkpoints_dir + "params" + str(step)
        self.saver.save(self.sess, dir_name, write_meta_graph=False)



if __name__ == '__main__':
    DQN = DuelingDQN()