import sys
sys.path.append("../")
import os
os.chdir("../")
from DQN import run,wtest
from setproctitle import setproctitle, getproctitle
import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

setproctitle("proc-yourname")

args = list(map(lambda x:float(x) if "." in x else int(x),sys.argv[1:]))
args.append(args[0])

data = [0,744*5000,"outside_aug_2018","gdemand1"] # 消費電力、満足度、bad人数、外気温、割引率 、学習回数F
args.extend(data)

run(*args)
wtest(*args)
