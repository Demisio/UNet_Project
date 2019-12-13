import numpy as np
import nibabel as nib
import logging
import h5py
import os

np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def rotate_90_flip(arr, dimensions, rotate=True, flip=True):
    """
    rotate and flip the input images (90 degree, horizontal & vertical flips)
    :param arr:
    :param dimensions:
    :param rotate:
    :param flip:
    :return:
    """

    #TODO: Integrate conditionals for flips & rotations

    # if rotate and flip:
    aug_array = np.zeros(shape=[8*dimensions[0], dimensions[1], dimensions[2], dimensions[3]], dtype=np.uint8)
    # elif rotate and not flip or flip and not rotate:
    #     aug_array = np.zeros(shape=[4 * dimensions[0], dimensions[1], dimensions[2], dimensions[3]], dtype=np.uint8)
    # else:
    #     aug_array = np.zeros(shape=dimensions, dtype=np.uint8)

    aug_dimensions = aug_array.shape
    logging.info('New dimensions of array, after augmentation: {}'.format(aug_dimensions))

    ## Indices of this part:
    ## Augmentation of 1 image into 8 --> 8 * original index = original image in augmented array
    ## Rotations are at indices current_index + 1 / 2 / 3
    ## Flips are at current_index + 4 / 5 / 6 / 7
    for ind in range(dimensions[0]):
        aug_ind = 8 * ind
        aug_array[aug_ind,:,:,0] = arr[ind,:,:,0]
        for i in range(1,4):
            aug_array[aug_ind + i, :, :, 0] = np.rot90(arr[ind,:,:,0], i)
            if i == 2:
                aug_array[aug_ind + 4, :, :, 0] = np.fliplr(arr[ind,:,:,0])
                aug_array[aug_ind + 5, :, :, 0] = np.flipud(arr[ind,:,:,0])
            elif i == 3:
                aug_array[aug_ind + 6, :, :, 0] = np.fliplr(aug_array[aug_ind + i,:,:,0])
                aug_array[aug_ind + 7, :, :, 0] = np.flipud(aug_array[aug_ind + i,:,:,0])

        assert np.array_equal(aug_array[aug_ind,:,:,:], arr[ind,:,:,:])

    return aug_array, aug_dimensions

def crop(arr, dimensions, syn_arr, syn_dim, crop_size, nr_crops):
    """
    crops an input array to the desired size (only squares currently)
    :param arr: raw / real images
    :param dimensions: dimensions of raw images pre-cropping
    :param syn_arr: synthetic images / GT images
    :param syn_dim: dimensions of synth. images pre-cropping
    :param crop_size: what's the desired crop size (square crop)
    :param nr_crops: how many crops from 1 image?
    :return: crop_arr, crop_dim, syn_crop, syn_crop_dim: crops and dimensions
    """

    crop_arr = np.zeros(shape=[nr_crops*dimensions[0], crop_size, crop_size, dimensions[3]], dtype=np.uint8)
    syn_crop = np.zeros(shape=[nr_crops*syn_dim[0], crop_size, crop_size, syn_dim[3]], dtype=np.uint8)

    for ind in range(dimensions[0]):
        new_ind = nr_crops*ind
        for i in range(nr_crops):
            # do random cropping of rotated and flipped images --> add something like translation invariance but without border problems
            start_x = int(np.random.uniform(0, dimensions[1] - crop_size))
            start_y = int(np.random.uniform(0, dimensions[2] - crop_size))
            stop_x = start_x + crop_size
            stop_y = start_y + crop_size
            crop_arr[new_ind + i,:,:,:] = arr[ind, start_x:stop_x, start_y:stop_y, :]
            syn_crop[new_ind + i,:,:,:] = syn_arr[ind, start_x:stop_x, start_y:stop_y, :]

    crop_dim = crop_arr.shape
    syn_crop_dim = syn_crop.shape

    logging.info('New dimensions of array, after cropping: {}'.format(crop_dim))

    return crop_arr, crop_dim, syn_crop, syn_crop_dim

def create_hdf5(file , raw_file, syn_file, raw_dim, syn_dim, group_a, group_b, iteration, aug_factor):

    assert len(raw_file.shape) == len(syn_file.shape)
    assert raw_dim == syn_dim

    it = iteration

    raw_num_samples = raw_dim[0] / aug_factor
    raw_num_channel = raw_dim[3]
    dtype = np.uint8

    if it == 1:

        group_a.create_dataset(name='num_samples', data=raw_num_samples)
        group_a.create_dataset(name='num_channel', data=raw_num_channel)
        group_a.create_dataset(name='aug_factor', data=aug_factor)


    data_A = raw_file
    group_a.create_dataset(name='data_' + str(it), data=(data_A), dtype=dtype)
    print('Created raw dataset: data_' + str(it))
    print('Shape of raw dataset: {}'.format(data_A.shape))
    print('Number of raw, non-augmented samples: {}'.format(raw_num_samples))
    print('Number of raw channels: {}'.format(raw_num_channel))
    print('Augmentation Factor: {}'.format(aug_factor))

    ## Synthetic data

    syn_num_samples = syn_dim[0] / aug_factor
    syn_num_channel = syn_dim[3]
    dtype = np.uint8

    if it == 1:
        group_b.create_dataset(name='num_samples', data=syn_num_samples)
        group_b.create_dataset(name='num_channel', data=syn_num_channel)
        group_b.create_dataset(name='aug_factor', data=aug_factor)


    data_B = syn_file
    group_b.create_dataset(name='data_' + str(it), data=(data_B), dtype=dtype)
    print('Created synth. dataset: data_' + str(it))
    print('Shape of synth dataset: {}'.format(data_B.shape))
    print('Number of synth., non-augmented samples: {}'.format(syn_num_samples))
    print('Number of synth. channels: {}'.format(syn_num_channel))
    print('Augmentation Factor: {}'.format(aug_factor))


if __name__ == '__main__':

    rotate = True
    flip = True
    nr_crops = 4
    filename = 'aug_heart_data.h5'

    raw_data_path = './../Data/Heart/3D/Raw/'
    # syn_data_path = './../Data/Heart/3D/Segmented/'
    syn_data_path = './../Data/Heart/3D/Segmented_og_labels/'
    # filename = '06_WK1_03_Segm_3D.nii.gz'

    raw_files = os.listdir(raw_data_path)
    syn_files = os.listdir(syn_data_path)

    raw_filepaths = []
    raw_filenames = []
    raw_aug_arr_list = []
    raw_aug_dim_list = []

    syn_filepaths = []
    syn_filenames = []
    syn_aug_arr_list = []
    syn_aug_dim_list = []

    ##TODO: Ordering of both has to be the same, possibly better to use numbered indexing and sorting rather than this iteration
    for el in raw_files:
        raw_filepaths.append(raw_data_path + el)
        raw_filenames.append(el)
    sort_raw_paths = sorted(raw_filepaths)
    sort_raw_names = sorted(raw_filenames)

    assert len(sort_raw_paths) == len(sort_raw_names)

    for el in syn_files:
        syn_filepaths.append(syn_data_path + el)
        syn_filenames.append(el)
    sort_syn_paths = sorted(syn_filepaths)
    sort_syn_names = sorted(syn_filenames)
    assert len(sort_syn_paths) == len(sort_syn_names)


    # Create the output hdf5 file & initialize iterator
    f = h5py.File(filename, "w")
    group_a = f.create_group('A')
    group_b = f.create_group('B')

    #Load data for all images, put it into list
    for idx in range(len(sort_raw_paths)):

        it = idx + 1

        ## raw files
        raw_im_path = sort_raw_paths[idx]
        raw_file_name = sort_raw_names[idx]

        raw_img = nib.load(raw_im_path)
        raw_data = raw_img.get_fdata()
        raw_dimensions = raw_data.shape
        logging.info('Data dimensions: {}'.format(raw_dimensions))

        ##Augmentations (cropping at same time, below)
        raw_aug_array, raw_aug_dim = rotate_90_flip(raw_data,raw_dimensions, rotate=rotate, flip=flip)

        ## synthetic files / ground truth
        syn_im_path = sort_syn_paths[idx]
        syn_file_name = sort_syn_names[idx]

        syn_img = nib.load(syn_im_path)
        syn_data = syn_img.get_fdata()
        syn_dimensions = syn_data.shape
        logging.info('Data dimensions: {}'.format(syn_dimensions))

        ##Augmentations
        syn_aug_array, syn_aug_dim = rotate_90_flip(syn_data, syn_dimensions, rotate=rotate, flip=flip)

        ## cropping for both at the same time to get same areas
        raw_crop_array, raw_crop_dim, syn_crop_array, syn_crop_dim = crop(raw_aug_array, raw_aug_dim, syn_aug_array, syn_aug_dim, 240, 4)

        #get augmentation factor for later indexing
        aug_factor = int(raw_crop_dim[0] / raw_dimensions[0])
        print('INFO:    Augmentation factor is: {}'.format(aug_factor))

        create_hdf5(f, raw_crop_array, syn_crop_array, raw_crop_dim, syn_crop_dim, group_a, group_b, it, aug_factor)
        print('########### Next Iteration ###########')
    # Close the file
    f.close()