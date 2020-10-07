import shutil

for i in range(5):
    for j in ["0_3_2.5_2_3","0_3_2_2.5_3","0_3_3_2_3"]:
        try:shutil.rmtree(f"gdemand{i}/{j}/outside_aug_2018_opt1")
        except:pass

