import sys
sys.path.append("../")
from DQN import wtest
from setproctitle import setproctitle, getproctitle
import tensorflow as tf
from keras.backend import tensorflow_backend
import os
os.chdir("../")
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

setproctitle("proc-yourname")

args = list(map(lambda x:float(x) if "." in x else int(x),sys.argv[1:]))
args.extend([args[0],0,744*1,"outside_aug_2019"])

for i in range(5): # ユーザパターン
    wtest(*(args+["gdemand"+str(i)]))
