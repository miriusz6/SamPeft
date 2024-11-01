
from PIL import Image
import numpy as np

from os import listdir
from os.path import isfile, join
from os import getcwd

from train_for_real import choose_bg_points, choose_target_points

curr_dit = getcwd()
im = Image.open(curr_dit + '/data/DataSmall/train/mask/0.png')
#im.show()

im_arr = np.array(im)

msk = np.asarray(im)
msk = np.where(msk>0, 1, 0)
# t_points = choose_target_points(msk, 5, min_dist=50)
# bg_points = choose_bg_points(msk, 5, min_dist=50)

tps = choose_target_points(msk, 5, min_dist=50)
bps = choose_bg_points(msk, 5, min_dist=50)
# plot points on image
import matplotlib.pyplot as plt
plt.imshow(msk)
plt.scatter(tps[:,1], tps[:,0], color='red')
plt.scatter(bps[:,1], bps[:,0], color='blue')
plt.show()