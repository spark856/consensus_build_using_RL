#coding:UTF-8
import sys
import os
import io
import gym
import numpy as np
import gym.spaces
import matplotlib.pyplot as plt


#env for DQN with 個別機器
class cobuenv(gym.Env):
   
    MAX_STEPS = 1000
    tmp_hist = []
    rw_hist = []
    light_hist = []
    
    total_step = 0
    def __init__(self,a,b,c,d,gamma,outname,demand_name,test=False,const=False,opt=0):
        super().__init__()
        self.maxn = 5
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(10* (4**self.maxn))
        low = np.array([0,0])
        high = np.array([9,40])
        self.observation_space = gym.spaces.Box( # 室温　照度　時刻　ユーザ 
             #np.hstack((low, np.zeros(self.maxn))),
             #np.hstack((high, np.full(self.maxn,4)))
             low,
             high
        )
        self.a = a
        self.b = b
        self.c = c
        self.d = d #無効化
        self.gamma = gamma
        self.outname = outname
        out = outname.split(":")[0]
        self.test = test
        self.const = const
        self.opt = opt
        

        temp = np.loadtxt("demands/"+demand_name)
        self.outside = np.loadtxt(f"outside/{out}.txt")
        self.datas = f"data/env4/aug_2018/{demand_name}/{self.gamma}_{self.a}_{self.b}_{self.c}_{self.d}"
        if self.const:
            self.data = self.datas + "/" + self.outname+"_const"
        elif self.opt:
            self.data = self.datas + "/" + self.outname+"_opt"+str(self.opt)
        else:
            self.data = self.datas + "/" + self.outname
            
        os.makedirs(self.data ,exist_ok=True)
        
        self.results = self.data+"/results"
        

        self.episode_length = len(self.outside)
        
        self.demand = []
        for i in range(self.maxn):
            self.demand += [temp[4*i:4*i+4,:]]
        self.reward_range = [-100., 100.]
        self.reset()
        
    def settest(self,flag):
        self.test = flag
        self.opt = 0

    def reset(self):
        # 諸々の変数を初期化する
        self.temp = 25
        self.light = 0
        self.time = 0
        self.rws = []
        #self.steps = self.total_step
        #self.total_step = 0 if self.total_step==23 else self.total_step + 1
        self.total_rw = 0
        self.total_pow = 0
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
        
        if self.const:
            self.lights = [0,0,0,0,0]
            self.temp = action + 20
        elif self.opt:
            maxr = -float("inf")
            if self.opt==6:
                data = []
            for j in range(10* (4**self.maxn)):
                tmp = j
                self.lights = []
                for i in range(self.maxn):
                    self.lights += [tmp%4]
                    tmp //= 4

                self.temp = tmp%10 + 20

                r1_1,r1_2 = self._get_powco()
                if self.opt==1:
                    r2 = self._get_satislv()
                    self.reward = r1_1 - r1_2 + r2  
                    if self.reward > maxr:
                        maxr = self.reward
                        opt_act = j
                elif self.opt==6:
                    data.append((self._get_satislv_premethod(),r1_1-r1_2,j))
            if self.opt==6:
                opt_act = min(data,key=lambda x:(-x[0],-x[1]))[2]
                

                        
            with open(self.data+f"/opt_act{self.opt}","a") as f:
                f.write(str(opt_act)+"\n")
                
            for i in range(self.maxn):
                self.lights += [opt_act%4]
                opt_act //= 4

            self.temp = opt_act%10 + 20

        else:
            self.lights = []

            for i in range(self.maxn):
                self.lights += [action%4]
                action //= 4

            self.temp = action%10 + 20
            action //= 10
        
        
        r1_1,r1_2 = self._get_powco()
        r2 = self._get_satislv()

        self.reward = r1_1 - r1_2 + r2 
        
        self.tmp_hist += [self.temp]
        self.total_rw += self.reward
        self.total_pow += r1_1

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
        
        
        if self.test:
            with open(self.results,"a") as f:
                f.write(str(temp)+"\n")
        
            self.light_hist += [self.lights]
        
        self.test_rw += self.reward
        self.test_temp += [temp]
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        #outfile.write(f"[{temp} {light} {time}]"+"\n")
        return outfile

    def close(self):
        if self.test:
            light_hist = np.asarray(self.light_hist)
            np.savetxt(self.data+"/lights",light_hist)
            with open(self.data+"/total_pow","a") as f:
                f.write(str(self.total_pow)+"\n")
            with open(self.data+"/total_rw","a") as f:
                f.write(str(self.total_rw)+"\n")
        else:
            rw_hist = np.asarray(self.rw_hist)
            np.savetxt(self.data+"/rw_hist",rw_hist)
        
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


    def _get_satislv_premethod(self):
        sati = []
        bad = -1
        notbad = 0
        
        for i,val in enumerate(self.demand):
            parson = val[self.lights[i]]
            if self.temp <= parson[0] or self.temp >= parson[3]:
                tmp = bad
            else:
                tmp = notbad
            sati += [tmp]
            

        return sum(sati)

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
            
            if self.test:
                
                with open(f"{self.data}/parson{i}","a") as f:
                    f.write(f"{tmp}\n")
                    
        badnum = sati.count(bad)
        if self.test:
            with open(self.data+"/badnum","a") as f:
                f.write(f"{badnum}\n")

        return self.b*sum(sati)
    
    
    def _get_powco(self):

        outdis = self.out - self.temp if self.out > self.temp else 0
        #indis = self.p_temp - self.temp if self.p_temp > self.temp else 0

        return self.a*(self.temp - 20),  self.a*outdis
    
        
    def _is_done(self):
        if self.time == self.episode_length:
            self.rw_hist += [self.total_rw]
            return True
        else:
            return False


    
