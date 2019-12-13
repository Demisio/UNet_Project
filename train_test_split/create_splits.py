import numpy as np
import os
from sklearn import model_selection
import pandas as pd
import re

## Here we only split the volumes, not the actual image slices --> Train / Test volumes

# How many images do we have?
split_path = './splits/'
data_path = './../Data/Heart/3D/Raw/'

n_splits = 5

## labels here are the regions of interest: Apex, Lateral Ventricle & Basal Septum coded with (1, 2, 3) respectively
## Do stratified split to have a balanced number in the splits
regex = re.compile(r'\d+')
ind_list = []
for el in os.listdir(data_path):
    reg_list = regex.findall(el)
    ind_list.append(int(reg_list[-1]))

## ids are the indices in the hdf5 array (1-11), labels are as mentioned above
## sorting just because of the order in the folder; might change for different filenames
ids = np.arange(1,len(ind_list) + 1)
labels = np.array(sorted(ind_list, reverse=True))


# kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20)

kfold =model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=20)

i = 1
for train, test in kfold.split(ids, labels):
    train_ids = ids[train]
    test_ids = ids[test]

    with open(os.path.join(split_path, 'train_fold_{}.txt'.format(i)), 'w') as file:
        for el in train_ids:
            file.write(str(el)+'\n')
    with open(os.path.join(split_path, 'test_fold_{}.txt'.format(i)), 'w') as file:
        for el in test_ids:
            file.write(str(el)+'\n')
    i += 1
