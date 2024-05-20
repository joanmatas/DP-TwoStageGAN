import cv2 as cv
import os
import math

dir = "../data/pics_raw"
save_dir = "../data/pics"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

cut_h = 526
cut_w = 467
h = 1001
w = 1920
dy = (h - cut_h) / 2
dx = (w - cut_w) / 2
for i in range(1024):
    pict = cv.imread(os.path.join(dir, "0" * (4 - len(str(i))) + str(i) + '.png'))
    pict = pict[math.floor(dy) : - math.ceil(dy), math.floor(dx) : - math.ceil(dx), :]
    pic_name = str(i // 32) + '_' + str(i % 32) + '.png'
    cv.imencode(".png", pict)[1].tofile(
        os.path.join(save_dir, pic_name)
    )