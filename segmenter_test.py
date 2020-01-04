import numpy as np
import time
from switch.data_switch import data_switch
from segmenter.model_segmenter import segmenter
from segmenter.experiments import heart_config as exp_config
import pickle
import h5py
import os
import gc

## for modification
save_path = exp_config.test_save_path
mode = exp_config.set
gen_img = exp_config.gen_img
datatype = exp_config.data_identifier
num_sample_volumes = exp_config.num_sample_volumes

# for later use
label = None
pred = None

# Define the model
real_start_time = time.time()
for fold in range(3,4): #(1, exp_config.nr_folds + 1)
    start_time = time.time()

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config.test_data_root, fold)

    # Build model
    segmenter_model = segmenter(exp_config=exp_config, data=data, fixed_batch_size=1)

    assert (mode == 'test') or (mode == 'validation') or (mode == 'train')

    print('Chosen set for fold {} is : {}'.format(fold, mode))
    print('')
    print('INFO:   Beginning testing phase')

    # specific for heart data currently

    if gen_img and fold == 1:
        filename = './Results/' + exp_config.log_name + '/Images/pred_' + str(mode) + '.h5'
        f = h5py.File(filename, "w")
        label = f.create_group('B_real')
        pred = f.create_group('B_fake')


    summary_dict = segmenter_model.test(
        batch_size=1,
        num_sample_volumes=num_sample_volumes,
        checkpoint='best_dice',
        datatype=datatype,
        gen_img=gen_img,
        set=mode,
        group_label=label,
        group_pred=pred)

    if mode == 'test':
        pickle.dump(summary_dict, open(os.path.join(save_path, 'test_summary_dicts_fold{}.p'.format(fold)), 'wb'))
    elif mode == 'validation':
        pickle.dump(summary_dict, open(os.path.join(save_path, 'validation_summary_dicts_fold{}.p'.format(fold)), 'wb'))
    elif mode == 'train':
        pickle.dump(summary_dict, open(os.path.join(save_path, 'train_summary_dicts_fold{}.p'.format(fold)), 'wb'))

    elapsed_time = time.time() - start_time
    print('Predicting fold %d took %.3f seconds' % (fold, elapsed_time))

    del segmenter_model
    gc.collect()

total_elapsed_time = time.time() - real_start_time
print('Total elapsed time: %.3f minutes' % (elapsed_time / 60))
