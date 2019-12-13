import numpy as np
import cv2
import imageio as io

# cannot import from parent module in python, work around from:  https://stackoverflow.com/a/11158224
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from create_h5_dataset import get_file_list

# old functions used for images, now everything is done in numpy directly

def rotateImage(image, angle):
    ## Taken from:   https://stackoverflow.com/a/9042907
    image = image
    angle = angle

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def flipImage(image):
    image = image
    horizontal_img = cv2.flip(image, 0)
    vertical_img = cv2.flip(image, 1)
    return horizontal_img, vertical_img


def modifications(files, dimensions, flag, save_path,
                  angle, nr_crops):
    num_samples = len(files)
    num_channel = dimensions[2]
    dtype = np.uint8

    data_A = np.zeros([num_samples, \
                       dimensions[0], \
                       dimensions[1], \
                       num_channel], dtype=dtype)

    for idx, fname in enumerate(files):
        if flag:  # This means, the images are gray scale
            # print('Images are grayscale')
            data_A[idx, :, :, 0] = np.array(cv2.imread(fname, cv2.IMREAD_GRAYSCALE))
        else:
            # print('Images are not grayscale')
            data_A[idx, :, :, :] = np.flip(np.array(cv2.imread(fname, cv2.IMREAD_COLOR)), 2)

    #increase contrast of images
    data_A[data_A == 1] = 0
    data_A[data_A == 2] = 120
    data_A[data_A == 3] = 255

    # define some stuff for rotations
    rotate_factor = int((360 / angle))

    for idx in range(num_samples):
        img = data_A[idx]
        img_list = []

        for rot in range(rotate_factor):
            rot_img = rotateImage(img, rot * angle)
            img_list.append(rot_img)
            hor_rot_img, ver_rot_img = flipImage(rot_img)
            img_list.append(hor_rot_img)
            img_list.append(ver_rot_img)

        print('Saving image number: ' + str(idx))
        for el in range(len(img_list)):
            #do random cropping of rotated and flipped images --> add something like translation invariance but without border problems
            for i in range(nr_crops):
                start_x = int(np.random.uniform(0,img_list[el].shape[0] - 256))
                start_y = int(np.random.uniform(0, img_list[el].shape[1] - 256))
                stop_x = start_x + 256
                stop_y = start_y + 256
                img_crop = img_list[el][start_x:stop_x, start_y:stop_y]

                io.imsave(save_path + str(idx) + '_' + str(el) + '_crop_' + str(i) + '.png', img_crop)

    aug_list = []
    for element in os.listdir(save_path):
        if '.tif' in element or '.png' in element:
            aug_list.append(data_path + element)
    num_aug_samples = len(aug_list)

    print('Increased number of pictures from initially: ' + str(num_samples) + ' to now: ' + str(num_aug_samples))

if __name__ == '__main__':
    data_path = "./Train_Images_Heart/"
    save_path = './Augmented_Heart/'

    angle = 90
    assert 360 % angle == 0
    nr_crops = 4

    files, dimensions, flag = get_file_list(data_path)
    modifications(files, dimensions, flag, save_path,
                  angle, nr_crops)
