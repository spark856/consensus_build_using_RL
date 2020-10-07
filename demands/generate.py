import numpy as np
from itertools import combinations as co

data = np.loadtxt("all")
num = list(range(10))
pat = list(co(num,5))

for a,t in enumerate(np.random.choice(range(len(pat)),3)):
    ans = []
    for i in pat[t]:
        for j in data[4*i:4*i+4]:
            ans.append(j)
    print(ans)
    np.savetxt(f"gdemand{a}",ans)
    with open("data","a") as f:
        #f.write(f"gdemand{a}: {pat[t]}\n")


