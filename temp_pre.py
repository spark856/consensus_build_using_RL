import pandas as pd
import numpy as np

print("開始日(年、月、日)-->",end="")
Y, M, D = map(int, input().rstrip().split())

print("日数-->", end="")
length = int(input())

print("ファイル名-->", end="")
name = input()


for i in range(D, D+length):
    url = "http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=66&block_no=47768&year=%d&month=%02d&day=%02d&view=p1" % (Y, M, i)

    soup = pd.read_html(url)[0]["気温(℃)"]

    for j in soup.values:
        with open(name ,"a") as f:
            f.write(str(j[0])+"\n")

    #print(soup.values[0])
