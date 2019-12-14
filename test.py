import numpy as np
import h5py
from matplotlib import pyplot as plt

path = './data_processing/aug_heart_data.h5'

data = h5py.File(path, 'r')
img_a = np.array(data['A/data_7'][13456,:,:,0])
img_b = np.array(data['B/data_7'][13456,:,:,0])

fig, axes = plt.subplots(1, 2)

plt.gray()

axes[0].imshow(img_a, interpolation='nearest')
axes[1].imshow(img_b, interpolation='nearest')

plt.show()

# toimage(img_a).show()
# toimage(img_b).show()