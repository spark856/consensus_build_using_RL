import numpy as np

for i in range(3):
    data = np.loadtxt("gdemand"+str(i))
    for row in data:
        with open("t"+str(i),"a") as f:
            f.write(" ".join(list(map(lambda x:str(int(x)),row)))+"\n")

