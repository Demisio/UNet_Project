import numpy as np


class BatchProvider_Heart():
    """
    Batch Provider class to provide batches of data for the network
    """
    def __init__(self, data, sample_indices, aug_factor, nr_img, imshape):
        """
        :param data: h5 data array
        :param sample_indices: Sample indices for the split
        :param aug_factor: how many times were images augmented?
        :param nr_img: the number of non-augmented images, calculated by: total_images / aug_factor
        :param imshape: shape of individual slices
        """
        self.data = data
        self.sample_indices = sample_indices
        self.aug_factor = aug_factor
        self.nr_img = nr_img
        self.imshape = imshape


    def next_batch(self, batch_size):
        """
        Get a single random batch, takes a pseudo-random slice from a random image.
        Pseudo-randomness: Slices will be from different sets if available, also since augmentation was performed,
        we have n different versions of an image, do not take the "same" image multiple times for an epoch
        """

        batch_a = np.zeros(shape=(batch_size, self.imshape[0], self.imshape[1], self.imshape[2]), dtype=np.uint8)
        batch_b = np.zeros(shape=(batch_size, self.imshape[0], self.imshape[1], self.imshape[2]), dtype=np.uint8)

        # get the batch indices, here these correspond to different sample volumes
        # if batch size is larger than sample volumes, work with resampling
        if batch_size <= self.sample_indices.shape[0]:
            batch_indices = np.random.choice(self.sample_indices, size=batch_size, replace=False)
        elif batch_size > self.sample_indices.shape[0]:
            batch_indices = np.random.choice(self.sample_indices, size=batch_size, replace=True)

        #get image indices, these correspond to slices
        ### do arange to get img idx array from 0 to 459, then sample from this, then perform operations

        batch_img = np.random.choice(int(self.nr_img), size=batch_size, replace=False)

        ## Access desired image as follows:
        img_indices = self.aug_factor * batch_img + np.random.choice(self.aug_factor)

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(batch_indices)
        img_indices = np.sort(img_indices)

        i = 0
        for idx, img in zip(batch_indices, img_indices):
            batch_a[i,:,:,:] = self.data['A' + '/data_' + str(idx)][img, :, :, :]
            batch_b[i,:,:,:] = self.data['B' + '/data_' + str(idx)][img, :, :, :]
            i += 1

        # special case for Unet implementation, no channel dimension for GT
        batch_b = np.squeeze(batch_b, axis=-1)

        return batch_a, batch_b


    def iterate_batches(self, batch_size, shuffle=True):
        """
        Get batch iterator, can be used in a for loop, e.g. when iterating over epochs to sample batches in a dataset,
        can be used in same way as "in range()" argument of a for loop
        """

        # shuffle img indices, then use iterator to provide sample batches
        img_indices = np.arange(self.nr_img)
        if shuffle:
            np.random.shuffle(img_indices)
        N = img_indices.shape[0]

        # iterate over datasets and number of images, provide batches for a total number of N images
        # with N = number of total non-augmented images
        for batch in range(0, N, batch_size):
            batch_a = np.zeros(shape=(batch_size, self.imshape[0], self.imshape[1], self.imshape[2]), dtype=np.uint8)
            batch_b = np.zeros(shape=(batch_size, self.imshape[0], self.imshape[1], self.imshape[2]), dtype=np.uint8)

            # get the batch indices, here these correspond to different sample volumes
            if batch_size <= self.sample_indices.shape[0]:
                batch_indices = np.random.choice(self.sample_indices, size=batch_size, replace=False)
            elif batch_size > self.sample_indices.shape[0]:
                batch_indices = np.random.choice(self.sample_indices, size=batch_size, replace=True)

            # get image indices, these correspond to slices
            ### do arange to get img idx array from 0 to 459, then sample from this, then perform operations

            batch_img = img_indices[batch:batch + batch_size]
            ## Access desired image as follows:
            img_ind = self.aug_factor * batch_img + np.random.choice(self.aug_factor)

            # HDF5 requires indices to be in increasing order
            batch_indices = np.sort(batch_indices)
            img_ind = np.sort(img_ind)

            i = 0
            for idx, img in zip(batch_indices, img_ind):
                batch_a[i, :, :, :] = self.data['A' + '/data_' + str(idx)][img, :, :, :]
                batch_b[i, :, :, :] = self.data['B' + '/data_' + str(idx)][img, :, :, :]
                # print('Use Sample / Slice: {} / {}'.format(idx, img))
                i += 1

            # special case for Unet implementation, no channel dimension for GT
            batch_b = np.squeeze(batch_b, axis=-1)

            yield batch_a, batch_b

    def test_image(self,img_idx, batch_size=1, sample_vol=0):
        """
        Get a single test batch (= image) in non-random order
        here, batch a has the following shape: [number of sample volumes, x, y, channels]
        --> each specific first index represents a specific volume
        """
        # here, batch a has the following shape: [number of sample volumes, x, y, channels]
        # --> each specific first index represents a specific volume
        batch_a = np.zeros(shape=(batch_size, self.imshape[0], self.imshape[1], self.imshape[2]), dtype=np.uint8)
        batch_b = np.zeros(shape=(batch_size, self.imshape[0], self.imshape[1], self.imshape[2]), dtype=np.uint8)

        # get the batch indices, here these correspond to different sample volumes
        # if batch size is larger than sample volumes, work with resampling

        batch_indices = np.sort(self.sample_indices)
        batch_indices = batch_indices[sample_vol]

        ## Access desired image as follows:
        img_indices = img_idx

        # HDF5 requires indices to be in increasing order
        batch_a[0, :, :, :] = self.data['A' + '/data_' + str(batch_indices)][img_indices, :, :, :]
        batch_b[0, :, :, :] = self.data['B' + '/data_' + str(batch_indices)][img_indices, :, :, :]

        # special case for Unet implementation, no channel dimension for GT
        batch_b = np.squeeze(batch_b, axis=-1)

        return batch_a, batch_b, batch_indices