
### Creates test data, meaning no rotation or flips. Crops are not random but rather chosen to crop the whole image symmetrically ###

import numpy as np
import nibabel as nib
import logging
import h5py
import os

np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')



def crop(arr, dimensions, syn_arr, syn_dim, crop_size, nr_crops):
    """
    here, crop only works properly if you choose crop size as [total image size / N + 10],
    e.g. total size = 460 -> N = 2 --> crop size = 230 + 10 = 240
    ensure that nr_crops has a normal number as output (proper integer), e.g. 4, 9, 16,...
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

    mod_crop = crop_size - 10
    for ind in range(dimensions[0]):
        new_ind = nr_crops*ind
        help_idx = 0
        for i in range(int(np.sqrt(nr_crops))):
            if i > 0:
                start_x = (i * mod_crop) - 10
            else: start_x = 0
            for l in range(int(np.sqrt(nr_crops))):
                if l > 0:
                    start_y = (l * mod_crop) - 10
                else:
                    start_y = 0
                stop_x = start_x + crop_size
                stop_y = start_y + crop_size
                crop_arr[new_ind + help_idx,:,:,:] = arr[ind, start_x:stop_x, start_y:stop_y, :]
                syn_crop[new_ind + help_idx,:,:,:] = syn_arr[ind, start_x:stop_x, start_y:stop_y, :]

                # if ind % 500 == 0:
                #     print('Help index: {}'.format(help_idx))
                help_idx += 1
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


    nr_crops = 4
    filename = 'aug_heart_data_test.h5'

    raw_data_path = './../Data/Heart/3D/Raw/'
    # syn_data_path = './../Data/Heart/3D/Segmented/'
    syn_data_path = './../Data/Heart/3D/Segmented_noisy/'
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

        ## synthetic files / ground truth
        syn_im_path = sort_syn_paths[idx]
        syn_file_name = sort_syn_names[idx]

        syn_img = nib.load(syn_im_path)
        syn_data = syn_img.get_fdata()
        syn_dimensions = syn_data.shape
        logging.info('Data dimensions: {}'.format(syn_dimensions))


        ## cropping for both at the same time to get same areas
        raw_crop_array, raw_crop_dim, syn_crop_array, syn_crop_dim = crop(raw_data, raw_dimensions, syn_data, syn_dimensions, 240, 4)

        #get augmentation factor for later indexing
        aug_factor = int(raw_crop_dim[0] / raw_dimensions[0])
        print('INFO:    Augmentation factor is: {}'.format(aug_factor))

        create_hdf5(f, raw_crop_array, syn_crop_array, raw_crop_dim, syn_crop_dim, group_a, group_b, it, aug_factor)
        print('########### Next Iteration ###########')
    # Close the file
    f.close()