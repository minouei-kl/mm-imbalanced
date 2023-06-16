import shutil
import pandas as pd
import os
import numpy as np
import csv

["letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication",
    "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]


cls_num = 16
img_max = 20000
imb_factor = 0.01
img_num_per_cls = []
for cls_idx in range(cls_num):
    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
    img_num_per_cls.append(int(num))

print(img_num_per_cls)
nc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
txtfile = r"train.txt"
csvfile = r"train100.csv"
outl = []

with open(txtfile, 'r') as infile, open(csvfile, 'w') as outfile:
    stripped = (line.strip() for line in infile)
    lines = (line.split(",") for line in stripped if line)
    # print(len(stripped))
    for li in lines:
        ll = li[0].split(' ', 1)
        # print(ll[0])
        # exit()
        cc = int(ll[1])
        nc[cc] = nc[cc]+1
        if nc[cc] <= img_num_per_cls[cc]:
            outl.append(str(li[0]))

# print(np.sum(nc))
# nc=map(str,nc)
# print('\n'.join(nc))
# exit()

    outfile.write('\n'.join(outl))
    outfile.close()
    # writer = csv.writer(outfile)
    # writer.writerows(outl)
