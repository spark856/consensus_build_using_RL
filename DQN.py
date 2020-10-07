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
from myenv.myenv2 import cobuenv

def run(a,b,c,d,gamma,steps,outname,demand_name):

    env = gym.make('cobuenv-v0',a=a,b=b,c=c,d=d,gamma=gamma,outname=outname,demand_name=demand_name)
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)
    #env = cobuenv()
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

    history = dqn.fit(env=env, nb_steps=steps, visualize=False, verbose=1, nb_max_episode_steps=1000)

    #dqn.test(env=env, nb_episodes=1, visualize=True)
    with open(f"{env.datas}/{steps}","w") as f:
        pass

    dqn.save_weights(f"{env.datas}/weight",True)
    
    env.close()

def wtest(a,b,c,d,gamma,steps,outname,demand_name):

    env = gym.make('cobuenv-v0',a=a,b=b,c=c,d=d,gamma=gamma,outname=outname,demand_name=demand_name,test=True)
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)
    #env = cobuenv()
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
    
    name = f"{env.datas}"
    
    dqn.load_weights(f"{name}/weight")
    
    
    
    
    a = dqn.test(env=env, nb_episodes=1, visualize=True)
    
    """
    with open(f"{env.data}/total_rw","a") as f:
            f.write(f"{a.history['episode_reward'][0]}\n")
    """
    env.close()
    
def opt_test(a,b,c,d,outname,demand_name,opt):# 貪欲法
    env = gym.make('cobuenv-v0',a=a,b=b,c=c,d=d,gamma=0,outname=outname,demand_name=demand_name,opt=opt)
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)
    
    #env = cobuenv()
    ans = []
    

    env.reset()
    total_rw = 0
    for i in range(env.episode_length):
        s,r,d,info = env.step(0)
        total_rw += r
    
    """
    with open(f"{env.data}/total_rw","w") as f:
            f.write(f"{total_rw}\n")
            
    """
    env.settest(True)
    
    with open(env.data+f"/opt_act{opt}","r") as f:
        opt_act = f.readlines()
    env.reset()
    for i in range(env.episode_length):
        action = int(opt_act[i])
        s,r,d,info = env.step(action)  
        env.render()
    
    env.close()


def c_test(a,b,c,d,outname,demand_name):# 照度固定
    env = gym.make('cobuenv-v0',a=a,b=b,c=c,d=d,gamma=0,outname=outname,demand_name=demand_name,const=True)
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)
    
    #env = cobuenv()
    ans = []
    
    for j in range(10):
        env.reset()
        total_rw = 0
        for i in range(env.episode_length):

            s,r,d,info = env.step(j)
            total_rw += r
        ans += [(total_rw,20+j)]

    
    const_max = max(ans,key=lambda x:x[0])
    
    env.settest(True)
    env.reset()
    for i in range(env.episode_length):
        s,r,d,info = env.step(const_max[1]-20)


    with open(f"{env.data}/total_rw","a") as f:
        f.write(f"{const_max}\n")
    with open(f"{env.data}/results","a") as f:
        for _ in range(744):
            f.write(str(const_max[1])+"\n")

    env.close()
    """
    with open(env.data+"/total_rw","r") as f:
        total_rw = float(f.read())
        
    if total_rw > const_max[0]:
        name = "OK"
    else:
        name = "NG"
        
    with open(env.data+"/"+name,"w") as f:
        f.write(f"test={total_rw},const={const_max}\n")
    """
    
    
def test_o(a,b,c,d,e): #温度照度全パターン
    env = gym.make('cobuenv-v0',a=a,b=b,c=c,d=d,e=e)
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)
    
    #env = cobuenv()
    ans = []
    
    
    for j in range(40):
            env.reset()
            total_rw = 0
            for i in range(env.episode_length):

                s,r,d,info = env.step(j)
                total_rw += r
                
            #print(j//10,j%10+20,total_rw)

            ans += [total_rw]
            
    const_max = max(ans)

    with open(env.data+"/total_rw","r") as f:
        total_rw = float(f.read())
        
    if total_rw > const_max:
        name = "OK"
    else:
        name = "NG"
        
    with open(env.data+"/"+name,"w") as f:
        f.write(f"test={total_rw},const={const_max}\n")
    
    

    