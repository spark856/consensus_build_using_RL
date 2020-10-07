import sys
from DQN import opt_test
from setproctitle import setproctitle, getproctitle
import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

setproctitle("proc-oishi")

args = list(map(lambda x:float(x) if "." in x else int(x),sys.argv[1:]))
args.extend([args[0],"outside_aug_2019"])

for j in range(6,7): # 1:総報酬　6:既存手法(不満:-1,それ以外:0)
    for i in range(5): # ユーザパターン
        opt_test(*(args+["gdemand"+str(i)]+[j]))
