from models import networks
import tensorflow as tf
import os
import config.system as sys_config
from tfwrapper import normalisation
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

#often changed parameters
experiment_name = 'heart' # some name
cv_fold = 3 # fold of cross validation
set = 'test' # 'train' / 'validation' / 'test' # at test time, which data should be used
mode = 'no_aug' ## change this depending on how much data you want to use: 'limited' / 'full' / 'GAN' / 'no_aug' / 'no_aug_lim'


#number of volumes
if set == 'test' or set == 'validation':
    num_sample_volumes = 2
elif set == 'train':
    num_sample_volumes = 6

gen_img = False # generate image or not


fold_name = 'fold' + str(cv_fold) # fold name in log directory

nr_folds = 1 # how many folds in cv, total


# Model settings
network = networks.unet2D_bn_dropout
normalisation = normalisation.batch_norm

# Data settings
data_identifier = 'heart'
preproc_folder = os.path.join(sys_config.project_root, 'data/preproc_data/acdc')

if mode == 'full':
    data_root = './data_processing/aug_heart_data.h5'
    log_name = 'Heart'  # dict name in ./logs
    test_save_path = './Results/Heart'
    logging.info('Using all Data')
elif mode == 'limited':
    data_root = './data_processing/aug_heart_data_very_limited.h5' #'./data_processing/aug_heart_data_limited.h5'
    log_name = 'Heart_very_lim' #'Heart_limited'  # dict name in ./logs
    test_save_path = './Results/Heart_very_lim' #'./Results/Heart_limited'
    logging.info('Using only limited Data')
elif mode == 'GAN':
    data_root = './data_processing/gan_data.h5'
    log_name = 'Heart_GAN'  # dict name in ./logs
    test_save_path = './Results/Heart_GAN'
    logging.info('Using GAN augmented Data')
elif mode == 'no_aug':
    data_root = './data_processing/aug_heart_data_test.h5'
    log_name = 'Heart_no_aug'  # dict name in ./logs
    test_save_path = './Results/Heart_no_aug'
    logging.info('Using non-augmented Data (only symmetric crops)')
elif mode == 'no_aug_lim':
    data_root = './data_processing/no_aug_heart_data_limited.h5'
    log_name = 'Heart_no_aug_lim'  # dict name in ./logs
    test_save_path = './Results/Heart_no_aug_lim'
    logging.info('Using non-augmented, limited Data (only symmetric crops)')
else:
    raise ValueError

test_data_root = './data_processing/aug_heart_data_test.h5'
# test_data_root = './data_processing/aug_heart_data.h5'

dimensionality_mode = '2D'
image_size = (240, 240)
nlabels = 3


# Network settings
n0 = 32 #32
log_loss = False
# Cost function
weight_decay = 0.0
loss_type = 'dice_macro_robust'  # 'dice_micro'/'dice_macro'/'dice_macro_robust'/'crossentropy'

# Training settings
batch_size = 2
n_accum_grads = 1
learning_rate = 1e-4
optimizer_handle = tf.train.AdamOptimizer
beta1=0.9
beta2=0.999
schedule_lr = False
divide_lr_frequency = None
warmup_training = False
momentum = None

# Rarely changed settings
use_data_fraction = False  # Should normally be False
# max_iterations = 1000000
num_epochs = 200
train_eval_frequency = 1
val_eval_frequency = 1