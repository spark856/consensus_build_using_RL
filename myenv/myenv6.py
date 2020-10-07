#coding:UTF-8
import sys
import os
import io
import gym
import numpy as np
import gym.spaces
import matplotlib.pyplot as plt


#env for DDPG with 個別機器 (softsign)
class cobuenv(gym.Env):
   
    MAX_STEPS = 1000
    tmp_hist = []
    rw_hist = []
    light_hist = []
    
    total_step = 0
    def __init__(self,a,b,c,d,gamma,test=False):
        super().__init__()
        self.maxn = 5
        # action_space, observation_space, reward_range を設定する
        
 
        self.action_space = gym.spaces.Box(
            np.hstack((0, np.zeros(self.maxn))),
            np.hstack((9, np.full(self.maxn,3)))
        )
    
        low = np.array([0])
        high = np.array([40])
        self.observation_space = gym.spaces.Box( # 外気温
             #np.hstack((low, np.zeros(self.maxn))),
             #np.hstack((high, np.full(self.maxn,3)))
             low,
             high
        )
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.gamma = gamma
        self.testf = test
        
        temp = np.loadtxt("demand.txt")
        self.outside = np.loadtxt("outside/outside_aug.txt")
        self.data = f"data/env6/aug_2018/{self.gamma}_{self.a}_{self.b}_{self.c}_{self.d}"
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
        self.observation = np.array([self.outside[1]])
        return self.observation

    def step(self, action):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)

        
        
        
        self.out = self.observation[0]
        self.lights = []
       
        

        flag = True
        for i in range(1,self.maxn+1):
            if not ( -10<= action[i] <=10):
                flag = False
                break
                
                
                
        if -10 <= action[0] <= 10 and flag:    
            tmp = int((action[0]+10)/(20/10))
            self.temp = tmp + 20 if tmp !=10 else tmp +19
            
            for i in range(1,self.maxn+1):
                tmp = int((action[i]+10)/(20/4))
                self.lights += [tmp if tmp !=4 else tmp-1] 

            r1 = self._get_powco()
            r2 = self._get_satislv()
            self.reward = r1 + r2 
        else:
            self.reward = -10**4

            
            
            
        self.tmp_hist += [self.temp]
        self.total_rw += self.reward
        
        self.time += 1
        self.done = self._is_done()
        

        #print("1",self.done)
        #self.observation = np.hstack((np.array([self.temp,self.light,self.time*10]),self.people))
        self.observation = np.array([self.outside[(self.time+1) % self.episode_length]])
        return self.observation, self.reward, self.done, {}

    def render(self, mode='human', close=False):

        if self.testf:
            with open(self.results,"a") as f:
                f.write(str(self.temp)+"\n")
        
            self.light_hist += [self.lights]
        
        self.test_rw += self.reward
        self.test_temp += [self.temp]
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        #outfile.write(f"[{temp} {light} {time}]"+"\n")
        return outfile

    def close(self):
        if self.testf:
            light_hist = np.asarray(self.light_hist)
            np.savetxt(self.data+"/lights",light_hist)
            
        
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
        bad = -self.c
        good = 0
        
        for i,val in enumerate(self.demand):
            parson = val[self.lights[i]]
            if self.temp >= parson[1] and self.temp <= parson[2]:
                tmp = best
            elif self.temp <= parson[0] or self.temp >= parson[3]:
                tmp = bad
            else:
                tmp = good

            sati += [tmp]
            
            if self.testf:
                with open(f"{self.data}/parson{i}","a") as f:
                    f.write(f"{tmp}\n")
                    
        badnum = sati.count(bad)
        if self.testf:
            with open(self.data+"/badnum","a") as f:
                f.write(f"{badnum}\n")

        return self.b*sum(sati)
    
    
    def _get_powco(self):

        outdis = self.out - self.temp if self.out > self.temp else 0
        #indis = self.p_temp - self.temp if self.p_temp > self.temp else 0

        ans = self.a*(self.temp - 20)  - self.d*outdis
        return ans
    
        
    def _is_done(self):
        if self.time == self.episode_length:
            self.rw_hist += [self.total_rw]
            return True
        else:
            return False


    
