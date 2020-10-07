from gym.envs.registration import register

register(
    id='cobuenv-v0',
    entry_point='myenv.myenv4:cobuenv',
    max_episode_steps=1000
)