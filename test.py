import numpy as np
import h5py
from matplotlib import pyplot as plt
import nibabel as nib
# path = './data_processing/aug_heart_data.h5'
#
# data = h5py.File(path, 'r')
# img_a = np.array(data['A/data_7'][13456,:,:,0])
# img_b = np.array(data['B/data_7'][13456,:,:,0])
#
# fig, axes = plt.subplots(1, 2)
#
# plt.gray()
#
# axes[0].imshow(img_a, interpolation='nearest')
# axes[1].imshow(img_b, interpolation='nearest')
#
# plt.show()

path1 = './Data/Heart/3D/Raw_lim_data/06_WK1_03_Cropabs.nii.gz'
raw_img1 = nib.load(path1)
raw_data1 = raw_img1.get_fdata()

path2 = './Data/Heart/3D/Segmented_og_labels_lim_data/06_WK1_03_Fusion.nii.gz'
raw_img2 = nib.load(path2)
raw_data2 = raw_img2.get_fdata()

fig, axes = plt.subplots(1, 2)

plt.gray()

axes[0].imshow(raw_data1[19,:,:,0], interpolation='nearest')
axes[1].imshow(raw_data2[19,:,:,0], interpolation='nearest')

plt.show()