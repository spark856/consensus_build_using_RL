#coding:UTF-8
import sys
import os
import io
import gym
import numpy as np
import gym.spaces
import matplotlib.pyplot as plt


#env for DQN with outside state
class cobuenv(gym.Env):
   
    MAX_STEPS = 1000

    gamma = 0.2
    
    rflag = 1
    pflag = 1
    nflag = 1
    
    tmp_hist = []
    rw_hist = []
    
    total_step = 0
    def __init__(self,a,b,c,d,e):
        super().__init__()
        self.maxn = 5
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(40)
        low = np.array([0,0])
        high = np.array([9,40])
        self.observation_space = gym.spaces.Box( # 室温　照度　時刻　ユーザ 
             #np.hstack((low, np.zeros(self.maxn))),
             #np.hstack((high, np.ones(self.maxn)))
             low,
             high
        )
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

        temp = np.loadtxt("demand.txt")
        self.outside = np.loadtxt("outside/outside_aug.txt")
        self.data = f"data/aug_2018/{self.gamma}_{self.a}_{self.b}_{self.c}_{self.d}_{self.e}"
        os.makedirs(self.data ,exist_ok=True)
        
        self.results = self.data+"/results"
       

        self.episode_length = len(self.outside)
        
        self.demand = []
        for i in range(self.maxn):
            self.demand += [temp[4*i:4*i+4,:]]
        self.reward_range = [-100., 100.]
        self.reset()

    def reset(self):
        # 諸々の変数を初期化する
        self.temp = 25
        self.light = 0
        self.time = 0
        self.rws = []
        #self.steps = self.total_step
        #self.total_step = 0 if self.total_step==23 else self.total_step + 1
        self.total_rw = 0
        #self.time = np.random.randint(0,23)
        self.people = np.ones(self.maxn)
        
        
        
        self.test_temp = []
        self.test_rw = 0
        
        #self.observation = np.hstack((np.array([self.temp,self.light,self.steps*10]),self.people))
        self.observation = np.array([(self.temp-20), self.outside[1]])
        return self.observation

    def step(self, action):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)

        self.p_temp = self.observation[0]
        self.out = self.observation[1]
        

        self.temp = action%10 + 20
        self.light = action//10

        
        r1 = self._get_powco()
        r2,r3 = self._get_satislv()

        self.reward = r1 + self.b*r2 + self.c*r3


        self.tmp_hist += [self.temp]
        self.total_rw += self.reward
        

        self.time += 1
        self.done = self._is_done()
        
        
        
        #print("1",self.done)
        #self.observation = np.hstack((np.array([self.temp,self.light,self.time*10]),self.people))
        self.observation = np.array([(self.temp-20),self.outside[(self.time+1) % self.episode_length]])
        return self.observation, self.reward, self.done, {}

    def render(self, mode='human', close=False):
        temp = self.observation[0]+20
        #light = self.observation[1]
        #time = self.observation[2]
        
        
        if self.rflag:
            with open(self.results,"a") as f:
                f.write(str(temp)+"\n")
        
        self.test_rw += self.reward
        self.test_temp += [temp]
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        #outfile.write(f"[{temp} {light} {time}]"+"\n")
        return outfile

    def close(self):
        
        plt.plot(self.rw_hist)
        plt.xlabel("steps")
        plt.ylabel("reward")
        plt.show()
        
        plt.plot(self.tmp_hist)
        plt.xlabel("steps")
        plt.ylabel("action")
        plt.show()
        
        plt.plot(self.test_temp)
        plt.xlabel("steps")
        plt.ylabel("test_action")
        plt.show()

        
        #print(self.rw_hist)

    def seed(self, seed=None):
        pass




    def _get_satislv(self):
        sati = []
        best = 1
        bad = -1
        good = 0
        
        for i,val in enumerate(self.demand):
            parson = val[self.light]
            if self.temp >= parson[1] and self.temp <= parson[2]:
                tmp = best
            elif self.temp <= parson[0] or self.temp >= parson[3]:
                tmp = bad
            else:
                tmp = good

            sati += [tmp]
            
            if self.pflag:
                with open(f"{self.data}/parson{i}","a") as f:
                    f.write(f"{tmp}\n")
                    
        badnum = sati.count(bad)
        if self.nflag:
            with open(self.data+"/badnum","a") as f:
                f.write(f"{badnum}\n")

        return sum(sati), -badnum
    
    
    def _get_powco(self):

        outdis = self.out - self.temp if self.out > self.temp else 0
        indis = self.p_temp - self.temp if self.p_temp > self.temp else 0

        ans = self.a*(self.temp - 20)  - self.d*outdis - self.e*indis
        return ans
    
        
    def _is_done(self):
        if self.time == self.episode_length:
            self.rw_hist += [self.total_rw]
            return True
        else:
            return False


    
