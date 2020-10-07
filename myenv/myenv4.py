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
    
"""
初期化関数
a: 消費電力の重み
b: ユーザ満足度の重み
c: bad値を-cとする(good,normalは常に1,0)
d: 無効(外気温の重みだったが、aに統合した)

gamma: 強化学習の割引率
outname: 外気温ファイルの名前
demand_name: ユーザパターンの名前
test: テストモード(ファイル書き込みなど)
const: 一定値制御モード
opt: 最適制御モード(1:最適制御,6:従来手法)

"""
    def __init__(self,a,b,c,d,gamma,outname,demand_name,test=False,const=False,opt=0):
        super().__init__()
        #人数
        self.maxn = 5

        #行動のサイズ
        #温度自由度10 × （照度自由度4^人数)
        self.action_space = gym.spaces.Discrete(10* (4**self.maxn))

        #状態のサイズ
        #室温(20+0~9) 外気温(0~40)
        low = np.array([0,0])
        high = np.array([9,40])
        self.observation_space = gym.spaces.Box(
             low,
             high
        )
        self.a = a
        self.b = b
        self.c = c
        self.d = d 
        self.gamma = gamma
        self.outname = outname
        out = outname.split(":")[0]
        self.test = test
        self.const = const
        self.opt = opt
        

        self.outside = np.loadtxt(f"outside/{out}.txt")

        #出力フォルダ指定
        self.datas = f"data/env4/aug_2018/{demand_name}/{self.gamma}_{self.a}_{self.b}_{self.c}_{self.d}"
        if self.const:
            self.data = self.datas + "/" + self.outname+"_const"
        elif self.opt:
            self.data = self.datas + "/" + self.outname+"_opt"+str(self.opt)
        else:
            self.data = self.datas + "/" + self.outname
        os.makedirs(self.data ,exist_ok=True)
        self.results = self.data+"/results"
        

        #エピソード数は一ヵ月の日数
        self.episode_length = len(self.outside)

        #ユーザパターンの読み込み
        temp = np.loadtxt("demands/"+demand_name)
        self.demand = []
        for i in range(self.maxn):
            self.demand += [temp[4*i:4*i+4,:]]

        self.reward_range = [-100., 100.]
        self.reset()
        
    #テストモードへの切り替え
    def settest(self,flag):
        self.test = flag
        self.opt = 0

    #エピソードのリセット
    def reset(self):
        # 諸々の変数を初期化する
        self.temp = 25
        self.light = 0
        self.time = 0
        self.rws = []
        self.total_rw = 0
        self.total_pow = 0
        self.people = np.ones(self.maxn)
        
        self.test_temp = []
        self.test_rw = 0
        
        self.observation = np.array([(self.temp-20), self.outside[1]])
        return self.observation

    def step(self, action):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)

        self.p_temp = self.observation[0]
        self.out = self.observation[1]
        
        #一定値制御
        if self.const:
            self.lights = [0,0,0,0,0]
            self.temp = action + 20
        #最適値制御
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
                

            #最適行動の保存            
            with open(self.data+f"/opt_act{self.opt}","a") as f:
                f.write(str(opt_act)+"\n")
                
            for i in range(self.maxn):
                self.lights += [opt_act%4]
                opt_act //= 4

            self.temp = opt_act%10 + 20

        #学習時
        else:
            self.lights = []

            for i in range(self.maxn):
                self.lights += [action%4]
                action //= 4

            self.temp = action%10 + 20
            action //= 10
        
        
        #消費電力報酬
        r1_1,r1_2 = self._get_powco()
        #ユーザ満足度報酬
        r2 = self._get_satislv()
        #全体報酬
        self.reward = r1_1 - r1_2 + r2 
        

        #出力したいデータの記録
        self.tmp_hist += [self.temp]
        self.total_rw += self.reward
        self.total_pow += r1_1

        #時刻を進ませる
        self.time += 1

        self.done = self._is_done()
        self.observation = np.array([(self.temp-20),self.outside[(self.time+1) % self.episode_length]])
        return self.observation, self.reward, self.done, {}


    #テストのときに毎ステップ実行される関数
    def render(self, mode='human', close=False):
        temp = self.observation[0]+20
        
        if self.test: 
            with open(self.results,"a") as f:
                f.write(str(temp)+"\n")
        
            self.light_hist += [self.lights]
        
        self.test_temp += [temp]
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        return outfile

    #終了処理
    def close(self):
        #出力処理
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


    #ユーザ満足度(従来手法)
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

    #ユーザ満足度報酬
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
    
    
    #消費電力報酬
    def _get_powco(self):
        #外気温が室温より高いとき、差分を得る
        outdis = self.out - self.temp if self.out > self.temp else 0

        return self.a*(self.temp - 20),  self.a*outdis
    

    #一ヵ月ごとにエピソードを終了する
    def _is_done(self):
        if self.time == self.episode_length:
            self.rw_hist += [self.total_rw]
            return True
        else:
            return False


    
