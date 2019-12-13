# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import nibabel as nib
import numpy as np
import os
import logging
from skimage import measure, transform
import SimpleITK as sitk
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

try:
    import cv2
except:
    logging.warning('Could not import opencv. Augmentation functions will be unavailable.')
else:
    def rotate_image(img, angle, interp=cv2.INTER_LINEAR):

        rows, cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)

    def rotate_image_as_onehot(img, angle, nlabels, interp=cv2.INTER_LINEAR):

        onehot_output = rotate_image(convert_to_onehot(img, nlabels=nlabels), angle, interp)
        return np.argmax(onehot_output, axis=-1)

    def resize_image(im, size, interp=cv2.INTER_LINEAR):

        im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
        return im_resized

    def resize_image_as_onehot(im, size, nlabels, interp=cv2.INTER_LINEAR):

        onehot_output = resize_image(convert_to_onehot(im, nlabels), size, interp=interp)
        return np.argmax(onehot_output, axis=-1)


    def deformation_to_transformation(dx, dy):

        nx, ny = dx.shape

        grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))

        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        return map_x, map_y

    def dense_image_warp(im, dx, dy, interp=cv2.INTER_LINEAR, do_optimisation=True):

        map_x, map_y = deformation_to_transformation(dx, dy)

        # The following command converts the maps to compact fixed point representation
        # this leads to a ~20% increase in speed but could lead to accuracy losses
        # Can be uncommented
        if do_optimisation:
            map_x, map_y = cv2.convertMaps(map_x, map_y, dstmap1type=cv2.CV_16SC2)
        return cv2.remap(im, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT) #borderValue=float(np.min(im)))


    def dense_image_warp_as_onehot(im, dx, dy, nlabels, interp=cv2.INTER_LINEAR, do_optimisation=True):

        onehot_output = dense_image_warp(convert_to_onehot(im, nlabels), dx, dy, interp, do_optimisation=do_optimisation)
        return np.argmax(onehot_output, axis=-1)


def convert_to_onehot(lblmap, nlabels):

    output = np.zeros((lblmap.shape[0], lblmap.shape[1], nlabels))
    for ii in range(nlabels):
        output[:,:,ii] = (lblmap == ii).astype(np.uint8)
    return output

def ncc(a,v, zero_norm=True):

    a = a.flatten()
    v = v.flatten()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a,v)


def norm_l2(a,v):

    a = a.flatten()
    v = v.flatten()

    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) / np.std(v)

    return np.mean(np.sqrt(a**2 + v**2))



def all_argmax(arr, axis=None):

    return np.argwhere(arr == np.amax(arr, axis=axis))


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


def create_and_save_nii(data, img_path):

    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, img_path)



class Bunch:
    # Useful shortcut for making struct like contructs
    # Example:
    # mystruct = Bunch(a=1, b=2)
    # print(mystruct.a)
    # >>> 1
    def __init__(self, **kwds):
        self.__dict__.update(kwds)



def convert_to_uint8(image):
    image = image - image.min()
    image = 255.0*np.divide(image.astype(np.float32),image.max())
    return image.astype(np.uint8)

def normalise_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)


def map_image_to_intensity_range(image, min_o, max_o, percentiles=0):

    # If percentile = 0 uses min and max. Percentile >0 makes normalisation more robust to outliers.

    if image.dtype in [np.uint8, np.uint16, np.uint32]:
        assert min_o >= 0, 'Input image type is uintXX but you selected a negative min_o: %f' % min_o

    if image.dtype == np.uint8:
        assert max_o <= 255, 'Input image type is uint8 but you selected a max_o > 255: %f' % max_o

    min_i = np.percentile(image, 0 + percentiles)
    max_i = np.percentile(image, 100 - percentiles)

    image = (np.divide((image - min_i), max_i - min_i) * (max_o - min_o) + min_o).copy()

    image[image > max_o] = max_o
    image[image < min_o] = min_o

    return image


def map_images_to_intensity_range(X, min_o, max_o, percentiles=0):

    X_mapped = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,...]
        X_mapped[ii,...] = map_image_to_intensity_range(Xc, min_o, max_o, percentiles)

    return X_mapped.astype(np.float32)

def center_images(X):

    X_centered = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):
        Xc = X[ii, ...]
        X_centered[ii, ...] = Xc - np.mean(Xc)

    return X_centered.astype(np.float32)

def normalise_images(X):
    '''
    Helper for making the images zero mean and unit standard deviation i.e. `white`
    '''

    X_white = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,...]
        X_white[ii,...] = normalise_image(Xc)

    return X_white.astype(np.float32)


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1, 2, 3]:

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img


SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def resample_sitk_image(sitk_image,
                        new_spacing,
                        interpolator=None,
                        fill_value=0,
                        anti_aliasing=True,
                        return_factor=False):
    """
    Resamples an ITK image to a new grid.
    :param sitk_image: SimpleITK image
    :param new_spacing: tuple, specifying the output spacing
    :param interpolator: str, for example 'nearest' or 'linear'
    :param fill_value: int
    :param anti_aliasing: bool, whether to use smoothing before downsampling
    :param return_factor: bool
    :return: SimpleITK image
    """

    assert interpolator in SITK_INTERPOLATOR_DICT.keys(),\
        "'interpolator' should be one of {}".format(SITK_INTERPOLATOR_DICT.keys())
    sitk_interpolator = SITK_INTERPOLATOR_DICT[interpolator]

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = sitk_image.GetSpacing()
    orig_size = sitk_image.GetSize()

    size_factor = np.array(orig_spacing) / np.array(new_spacing)

    new_size = np.array(orig_size) * size_factor
    new_size = [int(el) for el in new_size]  # image dimensions are in integers, SimpleITK expects lists
    new_spacing = [float(el) for el in new_spacing]

    if anti_aliasing:
        # just like in http://scikit-image.org/docs/dev/api/skimage.transform.html#resize
        sigma = (1 - size_factor) / 2
        # TODO: find a more elegant way to disable smoothing (it's working though)
        sigma[sigma <= 0] = 1e-12  # didn't find a prettier way to disable smoothing

        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        sitk_image = gaussian.Execute(sitk_image,
                                      sigma,
                                      False)

    resample_filter = sitk.ResampleImageFilter()
    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   sitk.Transform(),
                                                   sitk_interpolator,
                                                   orig_origin,
                                                   new_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)
    if return_factor:
        return resampled_sitk_image, size_factor
    return resampled_sitk_image


def clip_and_rescale_sitk_image(sitk_image, clip_range, data_type='float32'):
    """
    Clips a SimpleITK at a given intensity range and rescales the intensity scale to [0, 1]
    :param sitk_image: SimpleITK image
    :param clip_range: tuple, intensity interval to clip at
    :param data_type: data type of output image, for float32 image will be in range [0, 1],
                      for uint8 image will be in range [0, 255]
    :return: SimpleITK image
    """
    assert data_type in ['float32', 'uint8'], 'data_type must be float32 or uint8'
    data_type_dict = {'float32': sitk.sitkFloat32,
                      'uint8': sitk.sitkUInt8}
    range_dict = {'float32': [0, 1],
                  'uint8': [0, 255]}

    clip = sitk.ClampImageFilter()
    clipped_image = clip.Execute(sitk_image, data_type_dict[data_type], *clip_range)

    rescale = sitk.RescaleIntensityImageFilter()
    rescaled_image = rescale.Execute(clipped_image, *range_dict[data_type])

    return rescaled_image

# from Christine Tanner's DeepMicroIA
class seedKeeper:
    """ Simple class to manage seeds for experimental repeatability."""
    def __init__(self, initial_seed=1991):
        self.seed = initial_seed
    def fetch(self):
        s = self.seed
        self.seed += 1
        return s
    def setSeed(self):
        seed = self.fetch()
        np.random.seed(seed)

