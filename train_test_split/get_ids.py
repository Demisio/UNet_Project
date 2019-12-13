## Script for getting the indices of images and storing them ##

import numpy as np
import pandas as pd
import os
import re

##Todo: currently only 1 folder considered, have to add some measurement in number for multiple folders, e.g. add 1000 for each new folder considered

## Do same thing for labels and images (even if GT has exact same label as raw images) for sake of proper completion
im_path = './../Data/Heart/Raw/06_WK1_03_Cropabs/'
gt_path = './../Data/Heart/Segmented/06_WK1_03_Fusion/'

save_path = './Heart/'
im_list = []
gt_list = []

regex = re.compile(r'\d+')

for el in os.listdir(im_path):
    reg_list = regex.findall(im_path + el)
    im_list.append(int(reg_list[-1]))

for el in os.listdir(gt_path):
    reg_list = regex.findall(gt_path + el)
    gt_list.append(int(reg_list[-1]))

im_list = sorted(im_list)
gt_list = sorted(gt_list)

print('Amount of images in folder ' + im_path + ' is: ' + str(len(im_list)))
print('')
print('Amount of segmented images in folder ' + gt_path + ' is: ' + str(len(gt_list)))

im_array = np.asarray(im_list)
gt_array = np.asarray(gt_list)

np.savetxt(save_path + 'heart_ids.txt', im_array, fmt='%i', delimiter=',')
np.savetxt(save_path + 'heart_gt_ids.txt', im_array, fmt='%i', delimiter=',')
