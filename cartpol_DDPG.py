import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Brain_DDPG import  Brain_DDPG
from Brain_DDPG import Memory

'''使用Brain DDPG运行cartpole
多循环一些回合后,比如episode 2000 以内，可以出现永远不倒的情况，测过100万次杆子也不倒。
为了节约时间，设置为10万次不倒主动退出。
'''

def main():
    ENV_NAME = 'CartPole-v0'
    EPISODE = 3000  #最大回合数
    MAX_EP_STEPS = 7500  # 一个回合最大学习步数 最多学习多少步不倒。可以设置更大一些。
    batchsize = MAX_EP_STEPS   #采样批次数量
    maxmemorysize = int(batchsize)  #记忆状态动作的内存大小

    np.random.seed(1)
    tf.set_random_seed(1)
    env = gym.make(ENV_NAME)
    env.seed(11) #22
    env = env.unwrapped


    s_dim = env.observation_space.shape[0]  #状态维度4
    actions_dim = 1  #动作维度
    action_high = 1  #动作最大值


    #agent为 Brain DDPG,使用DDPG算法。
    agent = Brain_DDPG(s_dim,actions_dim,maxmemorysize,batchsize,action_high)

    var = 1  #随机数的系数
    RENDER = False
    R = Memory(s_dim, actions_dim, MAX_EP_STEPS, batchsize) #记录一个回合的内存
    for episode in range(EPISODE):
        # 回合开始
        # initialize task
        R.reset_memory()
        s = env.reset()
        learned = False
        RENDER = False
        step = 0  #因为步数可能很大，不用for循环
        while True:
            step += 1
            if RENDER: env.render()
            #选动作 加 随机
            action = agent.choice_action(s)[0] + np.random.normal(0, 0.5) * var

            #动作0或1
            if action >= 0.5:
                action = 1
            else:
                action = 0

            s_, r, done, _ = env.step(action)  #在环境中走一步
            if done : r = -1
            R.store_memory(s, action, r , s_)  #保存在回合内存中

            if done :
                if step <= MAX_EP_STEPS:
                    R.discount_normal_reward() #计算本回合的奖励贴现
                    for j in range(R.memcount):
                        record = R.mem[j:j+1,]
                        agent.R.append_record(record)  #将状态存到agent中

                    if(agent.R.memcount >= batchsize ):
                        var *= 0.997
                        agent.learn('all')
                        learned = True
                break
            else:
                #杆子没有倒，已达最大步 训练
                if step == MAX_EP_STEPS:
                    #计算奖励贴现值
                    R.discount_normal_reward()
                    for j in range(R.memcount):
                        record = R.mem[j:j+1,]
                        agent.R.append_record(record)
                    var *= 0.9995              #系数衰减 0.9992 0.9993
                    agent.learn('all')
                    learned = True
                    #RENDER = True
                elif step > MAX_EP_STEPS:
                    #超过9万9千5百步，最后500步显示动画
                    if step>=99500: RENDER = True
                    #越过最大步数，每20000次打印一次。
                    if step % 20000 == 0:
                        print('running now step = {}'.format(step))
                    if step >= 100000:
                        print('已经10万次不倒，可能永远不会倒了，主动退出')
                        break
            s = s_  #当前状态等于下一个状态

        print("episode={:d}，done={},learned={},var={:.4f}, maxstep={}".format(episode,done,learned,var,step))



if __name__ == '__main__':
    main()
