from models import networks
import tensorflow as tf
import os
import config.system as sys_config
from tfwrapper import normalisation

experiment_name = 'heart_unet'

cv_fold = 1
log_name = 'Heart'
fold_name = 'fold' + str(cv_fold)

# Model settings
network = networks.unet2D_bn_dropout
normalisation = normalisation.batch_norm

# Data settings
data_identifier = 'heart'
preproc_folder = os.path.join(sys_config.project_root, 'data/preproc_data/acdc')
data_root = './data_processing/aug_heart_data.h5'
dimensionality_mode = '2D'
image_size = (240, 240)
nlabels = 3


# Network settings
n0 = 64 #32
log_loss = True
# Cost function
weight_decay = 0.0
loss_type = 'dice_macro_robust'  # 'dice_micro'/'dice_macro'/'dice_macro_robust'/'crossentropy'

# Training settings
batch_size = 4
n_accum_grads = 1
learning_rate = 1e-3
optimizer_handle = tf.train.AdamOptimizer
beta1=0.9
beta2=0.999
schedule_lr = False
divide_lr_frequency = None
warmup_training = False
momentum = None

# Augmentation
do_augmentations = True
augmentation_options = { 'do_rotations': True,
                         'do_scaleaug': True,
                         'do_fliplr': True,
                         'do_flipud': True,
                         'do_elasticaug': True,
                         'augment_every_nth': 2}

# Rarely changed settings
use_data_fraction = False  # Should normally be False
# max_iterations = 1000000
num_epochs = 200
train_eval_frequency = 10
val_eval_frequency = 1