import tensorflow as tf
import numpy as np

'''
使用DDPG算法.
类Actor动作  输入s 输出a ,训练需要(s,a,dq/da)
类Critic评价
'''


class Actor(object):
    def __init__(self, sess,s_input, s_dim, actions_dim, scope, trainable,action_bound):
        self.sess = sess
        self.s_input = s_input
        self.s_dim = s_dim
        self.actions_dim = actions_dim
        self.trainable = trainable
        self.scope = scope
        self.l1_count = 32
        self.lr = 0.01
        self.action_bound = action_bound
        self.build_net()

    def build_net(self):
        w_init = tf.random_normal_initializer(0.0, 0.300)
        b_init = tf.constant_initializer(0.1)
        with tf.variable_scope(self.scope):
            w1 = tf.get_variable('w1', shape=[self.s_dim,self.l1_count], dtype='float32',
                                 initializer=w_init, trainable=self.trainable)
            b1 = tf.get_variable('b1',shape=[self.l1_count],dtype='float32',initializer=b_init,
                                 trainable=self.trainable)
            l1 = tf.nn.relu( tf.matmul(self.s_input,w1)+b1)

            w2 = tf.get_variable('w2',shape=[self.l1_count,self.actions_dim],dtype='float32',
                                 initializer=w_init, trainable=self.trainable)
            b2 = tf.get_variable('b2',shape=[self.actions_dim],dtype='float32',initializer=b_init,
                                 trainable=self.trainable)
            net = tf.nn.tanh(tf.matmul(l1, w2) + b2)

        self.a_net = tf.multiply(net, self.action_bound)
        self.theta_u = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def buildtrain(self,dqda):
        self.loss = tf.reduce_mean(tf.pow(tf.multiply(self.a_net , dqda),2))
        #grad = da * dqda / dtheta_q
        grad = tf.gradients(ys = self.a_net, xs = self.theta_u, grad_ys=dqda)
        opt = tf.train.AdamOptimizer( - self.lr)
        self.train_op = opt.apply_gradients(zip(grad,self.theta_u))



class Critic(object):
    def __init__(self,sess,s_input,r_input,s_dim, actions_dim, scope, trainable, a_net):
        self.sess = sess
        self.s_input = s_input
        self.r_input = r_input
        self.s_dim = s_dim
        self.a_net = a_net
        self.actions_dim = actions_dim
        self.scope = scope
        self.trainable=trainable
        self.l1_count = 32
        self.lr = 0.01
        self.target_q_net = None
        self.gamma = 0.9
        self.build_net()


    def build_net(self):
        w_init = tf.random_normal_initializer(0., 0.3)
        b_init = tf.constant_initializer(0.10)
        with tf.variable_scope(self.scope):
            ws1 = tf.get_variable('ws1', shape=[self.s_dim,self.l1_count], dtype='float32',
                                  initializer=w_init, trainable=self.trainable)
            b1 = tf.get_variable('b1',shape=[self.l1_count],dtype='float32',initializer=b_init,
                                 trainable=self.trainable)
            wa1 = tf.get_variable('wa1', shape=[self.actions_dim,self.l1_count], dtype='float32',
                                  initializer=w_init, trainable=self.trainable)
            l1 = tf.nn.relu(tf.matmul(self.s_input, ws1) + tf.matmul(self.a_net, wa1) + b1)

            w2 = tf.get_variable('w2',shape=[self.l1_count,1],dtype='float32',
                                 initializer=w_init, trainable=self.trainable)
            b2 = tf.get_variable('b2',shape=[1],dtype='float32',initializer=b_init,
                                 trainable=self.trainable)
            self.q_net = tf.matmul(l1, w2) + b2

        self.grad_a_op = tf.gradients(self.q_net, self.a_net)
        self.theta_q = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.scope)
        self.target_q_net = self.r_input + self.gamma * self.q_net

    def buildTrain(self,target_q_net):
        self.target_q_net = target_q_net
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_q_net, self.q_net))
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(self.loss,var_list=self.theta_q)

    def build_grad_a(self, s):
        dqda = self.sess.run(self.grad_a_op, feed_dict={self.s_input:s})[0]
        return dqda





class Memory(object):
    def __init__(self,s_dim,actions_dim,memorysize,batchsize):
        self.maxmemorysize=memorysize
        self.actions_dim = actions_dim
        self.batchsize=batchsize
        self.s_dim = s_dim
        self.mem = np.zeros((memorysize,self.s_dim+1+1+s_dim),dtype='float32')
        self.index = 0
        self.memcount = 0
        self.gamma = 0.9

    def reset_memory(self):
        self.memcount = 0
        self.index = 0

    def store_memory(self,s,a,r,s_):
        rec = np.hstack((s, a, [r], s_))
        self.mem[self.index, :] = rec
        self.index += 1
        self.memcount += 1
        if self.index >= self.maxmemorysize: self.index = 0

    def rand_choice(self):
        count = min(self.memcount, len(self.mem))
        indexes = range(count)
        indexes = np.random.choice(indexes,self.batchsize)
        batch_mem = self.mem[indexes,:]
        b_s = batch_mem[:, :self.s_dim]
        b_a = batch_mem[:, self.s_dim:self.s_dim+self.actions_dim]
        b_r = batch_mem[:, self.s_dim+self.actions_dim][:,None]
        b_s_ = batch_mem[:, -self.s_dim:]
        return b_s,b_a,b_r,b_s_

    def all_choice(self):
        assert self.memcount >= 1,'样本太少'
        indexes = range(min(self.memcount,self.maxmemorysize))
        batch_mem = self.mem[indexes,:]
        b_s = batch_mem[:, :self.s_dim]
        b_a = batch_mem[:, self.s_dim:self.s_dim+self.actions_dim]
        b_r = batch_mem[:, self.s_dim+self.actions_dim][:,None]
        b_s_ = batch_mem[:, -self.s_dim:]
        return b_s,b_a,b_r,b_s_

    def discount_normal_reward(self):
        b_r = self.mem[:self.memcount, self.s_dim+self.actions_dim]

        sum = 0
        for i in reversed(range(len(b_r))):
            sum = sum * self.gamma + b_r[i]
            b_r[i] = sum

    def append_record(self, rec):
        self.mem[self.index, :] = rec
        self.index += 1
        self.memcount += 1
        if self.index >= self.maxmemorysize: self.index = 0


'''DDPG算法
实例U， Actor估计网络 输入s 输出a ,训练需要(s,a,dq/da)
实例U_， Actor现实网络 输入s_,输出a_
实例Q,  Critic估计网络  输入s,a,输出q,训练需要(s,a,q_)
实例Q_, Critic现实网络  输入s_,a_,输出q_
实例R, 存储状态、动作、奖励

U.dqda = Q偏导 / U编导 = dQ.q_net / dU.a_net
训练U需要偏导 dQ / dtheta_u = (dQ.q_net / dU.a_net) * (dU.a_net / dU.theta_i)

选动作，choice_action(s)，返回动作a,参数s
存储动作:store_memory(s,a,r,s_)保存状态、动作、奖励、下一个状态
选取样本：从保存的内存R中随机选batchsize个样本。支持选择所有样本训练。
训练: learn (batchsize个样本)
   1） 运行U(s_),得动作a_ 由下一个状态得到下一个动作a_
   2)  运行Q_.target_q_net(r,s_,a_),返回得Q现实值q_
   3)  运行Q.train_op(s,a,q_) 误差为估计值q和现值q_的差，进行训练调整theta_Q。
   4） 求偏导数：a_grad = dQ / da
   5)  训练U。 梯度为 (dU  * a_grad) / dtheta_u
   6)  soft方式同步参数 theta_U <-(soft) theta_U ,theta_Q_ <-soft theta_Q

'''
class Brain_DDPG(object):
    def __init__(self, s_dim, actions_dim , memorysize,batchsize,action_bound):
        self.sess = tf.Session()
        #状态
        self.s_input = tf.placeholder(dtype='float32', shape=(None,s_dim))
        #下一个状态
        self.s__input = tf.placeholder(dtype='float32', shape=(None,s_dim))
        #奖励
        self.r_input = tf.placeholder(dtype='float32',shape=(None, 1))
        self.s_dim = s_dim
        self.actions_dim = actions_dim
        self.action_bound = action_bound

        self.U = Actor(self.sess,self.s_input,s_dim,actions_dim,'Actor/eval_net',True,action_bound)
        self.U_ = Actor(self.sess, self.s__input,s_dim,actions_dim,'Actor/target_net',False,action_bound)
        self.Q = Critic(self.sess, self.s_input,self.r_input,s_dim, actions_dim,'Critic/eval_net', True, self.U.a_net)
        self.Q_ = Critic(self.sess,self.s__input,self.r_input,s_dim,actions_dim,'Critic/target_net',False,self.U_.a_net)

        self.U.buildtrain(self.Q.grad_a_op)
        self.Q.buildTrain(self.Q_.target_q_net)


        self.R = Memory(s_dim,actions_dim,memorysize,batchsize)

        self.softcopy_tau = 0.015 #这个值不能太大，越小收敛越好

        self.assign_u_op = [tf.assign(t, e*self.softcopy_tau + (1 - self.softcopy_tau)*t) for (t,e) in
                            zip(self.U_.theta_u,self.U.theta_u)]
        self.assign_q_op = [tf.assign(t, e*self.softcopy_tau + (1 - self.softcopy_tau)*t) for (t,e) in
                            zip(self.Q_.theta_q,self.Q.theta_q)]
        self.sess.run(tf.global_variables_initializer())

    def reset_memory(self):
        self.R.reset_memory()


    def choice_action(self,s):
       s = s[None , :]
       a = self.sess.run(self.U.a_net, feed_dict={self.s_input: s})
       return a

    def store_memory(self,s,a,r,s_):
        self.R.store_memory(s,a,r,s_)

    def soft_assign(self,t_params,e_params,tau):
        [tf.assign(t, e*tau + (1 - tau)*t) for (t,e) in zip(t_params,e_params)]

    def store_memory(self,s,a,r,s_):
        self.R.store_memory(s,a,r,s_)

    def memorycount(self):
        return self.R.memcount

    def discount_normal_reward(self):
        self.R.discount_normal_reward()

    def learn(self,feature):

        if(feature == 'all'):
            b_s, b_a, b_r, b_s_ = self.R.all_choice()
        else:
            b_s, b_a, b_r, b_s_ = self.R.rand_choice()
        #训练Q
        _,q_loss = self.sess.run([self.Q.train_op,self.Q.loss],
                                 feed_dict={self.s_input: b_s, self.r_input: b_r,
                                            self.s__input: b_s_, self.Q.a_net: b_a})
        #print("q_loss={:.4f}".format(q_loss))

        #训练U
        _ , u_value = self.sess.run([self.U.train_op, self.U.loss],
                                    feed_dict={self.s_input: b_s, self.r_input: b_r,
                                               self.s__input: b_s_})
        #print("u_value={:.4f}".format(u_value))

        #soft 同步参数
        self.sess.run([self.assign_u_op,self.assign_q_op])
        return u_value,q_loss

