#coding:UTF-8
import sys
import io
import gym
import numpy as np
import gym.spaces
import matplotlib.pyplot as plt



class cobuenv(gym.Env):
   
    MAX_STEPS = 100 
    a = 2
    b = 1
    c = 3
    #d = 0.5
    e = 2
    tmp_hist = []
    rw_hist = []
    
    total_step = 0
    def __init__(self):
        super().__init__()
        self.maxn = 5
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Box( # 温度設定　照明設定
            np.array([20,0]),
            np.array([30,3])
        )
        low = np.array([0,0,0])
        high = np.array([40,3,23])
        self.observation_space = gym.spaces.Box( # 室温　照度　時刻　ユーザ 
             #np.hstack((low, np.zeros(self.maxn))),
             #np.hstack((high, np.ones(self.maxn)))
             low,
             high
        )
        
        temp = np.loadtxt("demand.txt")
        self.outside = np.loadtxt("outside.txt")
        self.demand = []
        for i in range(self.maxn):
            self.demand += [temp[4*i:4*i+4,:]]
        self.reward_range = [-100., 100.]
        self.reset()

    def reset(self):
        # 諸々の変数を初期化する
        self.temp = 30
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
        self.observation = np.array([(self.temp-20)/10,self.light/4,self.time/24])
        return self.observation

    def step(self, action):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)

        if type(action) is int:
            
            self.temp = action%10 + 20
            self.light = action//10
        else:
            index = action.argmax()
            self.temp = index%10 + 20
            self.light = index//10
        '''
        tmp = action[0]
        tmp = 30 if tmp > 30 else tmp
        tmp = 20 if tmp < 20 else tmp
        
        
        tmp = 0 if np.isnan(action[0]) else action[0]
        self.temp = int(tmp)%10+20 
        self.light = int(0 if np.isnan(action[1]) else action[1])%4
        '''
        
        
        
        self.time = int(self.observation[2]*24)

        r1 = self._get_powco()
        r2,r3 = self._get_satislv()
        #r4 = self._get_out()
        
        
        #d4 = (self.d*r4)
        #d4 = 1 if d4 < 0 else d4
        
        
        #print(r1,r2,r3)
        self.reward = self.a*r1 + self.b*r2 + self.c*r3

        #self.reward = self.reward / d4 if self.reward >= 0 else self.reward * d4

        self.tmp_hist += [self.temp]
        self.total_rw += self.reward
        
       
        
        
        
        self.time += 1
        self.done = self._is_done()
        
        
        
        #self.observation = np.hstack((np.array([self.temp,self.light,self.time*10]),self.people))
        self.observation = np.array([(self.temp-20)/10,self.light/4,self.time/24])
        return self.observation, self.reward, self.done, {}

    def render(self, mode='human', close=False):
        temp = self.observation[0]*10+20
        light = self.observation[1]*4
        time = self.observation[2]*24
        
        self.test_rw += self.reward
        self.test_temp += [temp]
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(f"[{temp} {light} {time}]"+"\n")
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

        print(self.total_rw)
        #print(self.rw_hist)

    def seed(self, seed=None):
        pass




    def _get_satislv(self):
        sati = []
        best = 1
        bad = -1
        good = 0
        
        for i in self.demand:
            parson = i[self.light]
            if self.temp >= parson[1] and self.temp <= parson[2]:
                temp = best
            elif self.temp <= parson[0] or self.temp >= parson[3]:
                temp = bad
            else:
                temp = good

            sati += [temp]
            
        return sum(sati),(-sati.count(bad))
    
    
    def _get_powco(self):
        '''
        ulim = 28
        dlim = 25
            
        if self.temp >= ulim:
            ans = 1
        elif self.temp >= dlim:
            ans = 0
        else:
            ans = -1
        '''
        out = self.outside[self.time%len(self.outside)]
        val = out - self.temp if out > self.temp else 0
        #val = out - self.temp
        
        ans = self.temp - 20  - self.e*val
        return ans
    
    def _get_out(self):
        out = int(self.outside[self.time%len(self.outside)])
        
        ans = self.temp - out if self.temp < out else 1
        
        return -ans
        
        
    def _is_done(self):
        if self.time == 24:
            self.rw_hist += [self.total_rw]
            return True
        else:
            return False


    
