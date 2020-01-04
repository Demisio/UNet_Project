'''
Script creates volumes of same size as full heart data but repeats the same slice several times (defined according to parameters)
step_size (initial spacing of slices), number_new_slices (how many slices per volume), factor (factor to help spacing of slices)
'''

import numpy as np
import nibabel as nib
import logging
import h5py
import os

np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

## change settings here
data_path = './../Data/Heart/3D/Segmented_og_labels/'
save_path = './../Data/Heart/3D/Segmented_og_labels_very_lim/'
#hardcoded for a volume of 460 slices, want 20 slices per volume with 23 in between
step_size = 92 #23
number_new_slices = 5 #20
factor = 14  #1

files = os.listdir(data_path)

filepaths = []
filenames = []

##TODO: Ordering of both has to be the same, possibly better to use numbered indexing and sorting rather than this iteration
for el in files:
    filepaths.append(data_path + el)
    filenames.append(el)
sort_paths = sorted(filepaths)
sort_names = sorted(filenames)
assert len(sort_paths) == len(sort_names)


# Load data for all images, put it into list
for idx in range(len(sort_paths)):
    it = idx + 1

    ## raw files
    im_path = sort_paths[idx]
    file_name = sort_names[idx]

    img = nib.load(im_path)
    data = img.get_fdata()
    dimensions = data.shape

    lim_data = np.zeros(shape=dimensions)

    it_idx = 0
    help_idx = 0
    curr_idx_list = []

    for i in range(0, dimensions[0], step_size):
        # if ((i + 1) % (dimensions[0] / percentage_slices) == 0) or (i == 0):
        term = factor * help_idx
        for j in range(step_size):
            #add same image at this interval
            curr_idx = int(j * number_new_slices + it_idx)
            old_idx = term + i
            lim_data[curr_idx, ...] = data[old_idx, ...]

            curr_idx_list.append(curr_idx)
            print(old_idx, curr_idx)
        it_idx += 1
        help_idx += 1
    print(sorted(curr_idx_list))
    print(len(curr_idx_list))
    print(len(set(curr_idx_list)))

    nib_img = nib.Nifti1Image(lim_data, np.eye(4))
    nib.save(nib_img, save_path + file_name)
    print('INFO:   Saved 3D image into: ' + save_path + ' as: ' + file_name)


