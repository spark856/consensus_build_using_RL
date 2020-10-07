import numpy as np

for i in range(1,2):
    for k in range(1,4):
        for j in ["0_3_3_2_3","0_3_2.5_2_3","0_3_2_2.5_3"]:
            temp_ai = np.loadtxt(f"data/env4/aug_2018/gdemand{i}/{k}/{j}/outside_aug_2018/results")
            temp_opt = np.loadtxt(f"data/env4/aug_2018/gdemand{i}/{j}/outside_aug_2018_opt1/results")
            #rw_ai = np.loadtxt(f"data/env4/aug_2018/gdemand{i}/{j}/outside_aug_201{k}/total_rw")
            #rw_opt = np.loadtxt(f"data/env4/aug_2018/gdemand{i}/{j}/outside_aug_201{k}_opt1/total_rw")
            count = 0

            for ta,to in zip(temp_ai,temp_opt):
                if ta==to:count += 1

            """
            try:rw_ai = rw_ai[0]
            except:pass
            try:rw_opt = rw_opt[0]
            except:pass

            print(f"gdemand{i},{j},201{k}")
            print(count*100/744,rw_ai*100/rw_opt)
            """
            print(f"{k},gdemand{i},{j}")
            print(count*100/744)
