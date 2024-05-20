import cv2 as cv
import os
import numpy as np

dir = "../data/pics"
save_path = "../data/map_pics_merge.png"
picts = {}
for i in os.listdir(dir):
    if i[-3:] != "png":
        continue
    num = int(i.split(".png")[0])
    key_num = num // 32
    if key_num not in picts.keys():
        picts[key_num] = cv.imread(os.path.join(dir, i))
    else:
        picts[key_num] = np.vstack([cv.imread(os.path.join(dir, i)), picts[key_num]])

stack = []
keys = sorted(picts.keys())
for k in keys:
    stack.append(picts[k])
output = np.hstack(stack)
cv.imencode(".png", output)[1].tofile(save_path)