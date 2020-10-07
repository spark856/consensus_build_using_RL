import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import gym
from gym import wrappers
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt

#学習する関数
def run(a,b,c,d,gamma,steps,outname,demand_name):

    #環境の読み込み
    env = gym.make('cobuenv-v0',a=a,b=b,c=c,d=d,gamma=gamma,outname=outname,demand_name=demand_name)
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)

    #DNNの定義
    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    #学習設定
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=0.001)
    dqn = DQNAgent(model=model, nb_actions=nb_actions,gamma=gamma, memory=memory, nb_steps_warmup=100,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    #学習
    history = dqn.fit(env=env, nb_steps=steps, visualize=False, verbose=1, nb_max_episode_steps=1000)

    #重みの保存
    dqn.save_weights(f"{env.datas}/weight",True)
    
    env.close()

#学習した重みのテスト
def wtest(a,b,c,d,gamma,steps,outname,demand_name):

    env = gym.make('cobuenv-v0',a=a,b=b,c=c,d=d,gamma=gamma,outname=outname,demand_name=demand_name,test=True)
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)

    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=0.001)
    dqn = DQNAgent(model=model, nb_actions=nb_actions,gamma=gamma, memory=memory, nb_steps_warmup=100,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    #重みの読み込み
    name = f"{env.datas}"
    dqn.load_weights(f"{name}/weight") 
    
    #テスト
    a = dqn.test(env=env, nb_episodes=1, visualize=True) 
    
    env.close()

#最適制御
def opt_test(a,b,c,d,outname,demand_name,opt):
    env = gym.make('cobuenv-v0',a=a,b=b,c=c,d=d,gamma=0,outname=outname,demand_name=demand_name,opt=opt)
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)
    
    #最適な行動の取得
    ans = []
    env.reset()
    total_rw = 0
    for i in range(env.episode_length):
        s,r,d,info = env.step(0)
        total_rw += r
    #テストモードに切り替え
    env.settest(True)
    
    #テスト
    with open(env.data+f"/opt_act{opt}","r") as f:
        opt_act = f.readlines()
    env.reset()
    for i in range(env.episode_length):
        action = int(opt_act[i])
        s,r,d,info = env.step(action)  
        env.render()
    
    env.close()


