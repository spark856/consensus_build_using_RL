import numpy as np
import gym
from gym import wrappers
from keras.models import Model
from keras.layers import Dense, Flatten, Input, concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt
import myenv

def build_actor_model(num_action, observation_shape):
    action_input = Input(shape=(1,)+observation_shape)
    x = Flatten()(action_input)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    #x = Dense(16, activation="relu")(x)
    x = Dense(num_action, activation="softsign")(x)
    actor = Model(inputs=action_input, outputs=x)
    return actor

def build_critic_model(num_action, observation_shape):
    action_input = Input(shape=(num_action,))
    observation_input = Input(shape=(1,)+observation_shape)
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    #x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    return (critic, action_input)

def build_agent(num_action, observation_shape,gamma):
    actor = build_actor_model(num_action, observation_shape)
    critic, critic_action_input = build_critic_model(num_action, observation_shape)
    memory = SequentialMemory(limit=5*(10**4), window_length=1)
    agent = DDPGAgent(
        num_action,
        actor,
        critic,
        critic_action_input,
        memory,
        gamma=gamma,

    )
    return agent

def run(a,b,c,d,gamma,steps):
    env = gym.make('cobuenv-v0',a=a,b=b,c=c,d=d,gamma=gamma)
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)

    print("Action Space: %s" % env.action_space)
    print("Observation Space: %s" % env.observation_space)
    #print(env.action_space.shape[0])
    agent = build_agent(env.action_space.shape[0], env.observation_space.shape,gamma)
    agent.compile(Adam(lr=0.001, clipnorm=1.), metrics=["mae"])
    history = agent.fit(env, nb_steps=steps, visualize=False, verbose=1, nb_max_episode_steps=10000)
    #agent.test(env,nb_episodes=1, visualize=True, nb_max_episode_steps=10000)
    #print(history.history["episode_reward"])
    with open(f"{env.data}/{steps}","w") as f:
        pass

    agent.save_weights(f"{env.data}/weight",True)

def wtest(a,b,c,d,gamma,steps):
    env = gym.make('cobuenv-v0',a=a,b=b,c=c,d=d,gamma=gamma,test=True)
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)

    print("Action Space: %s" % env.action_space)
    print("Observation Space: %s" % env.observation_space)
    agent = build_agent(env.action_space.shape[0], env.observation_space.shape,gamma)
    agent.compile(Adam(lr=0.001, clipnorm=1.), metrics=["mae"])
    #print(history.history["episode_reward"])

    agent.load_weights(f"{env.data}/weight")
    agent.test(env,nb_episodes=1, visualize=True, nb_max_episode_steps=10000)
    
    
    
def test():
    env = gym.make('cobuenv-v0')
    env = wrappers.Monitor(env, directory="/tmp/cobuenv", force=True)
    
    
    ans = []
    
    
    for j in range(40):
            env.reset()
            total_rw = 0
            for i in range(24):

                s,r,d,info = env.step(j)
                total_rw += r
                
            print(j//10,j%10+20,total_rw)

            ans += [total_rw]

    print(max(ans))
    
    
if __name__ == "__main__":
    run()
