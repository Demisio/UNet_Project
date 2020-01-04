# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from skimage import transform  # used for CAM saliency

from tfwrapper import losses
from tfwrapper import utils as tf_utils
import config.system as sys_config
from grad_accum_optimizers import grad_accum_optimizer_classifier
from sklearn.preprocessing import OneHotEncoder
import pickle
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment(GPU=0)

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))


class segmenter:

    def __init__(self, exp_config, data, fixed_batch_size=None, do_checkpoints=True):

        self.seed = 42
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        self.exp_config = exp_config
        self.data = data
        #use log loss or not
        self.log_loss = exp_config.log_loss

        self.nlabels = exp_config.nlabels

        self.image_tensor_shape = [fixed_batch_size] + list(exp_config.image_size) + [1]
        self.labels_tensor_shape = [fixed_batch_size] + list(exp_config.image_size)

        self.x_pl = tf.placeholder(tf.float32, shape=self.image_tensor_shape, name='images')
        self.y_pl = tf.placeholder(tf.uint8, shape=self.labels_tensor_shape, name='labels')

        self.lr_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.training_pl = tf.placeholder(tf.bool, shape=[], name='training_time')

        self.l_pl_ = exp_config.network(self.x_pl,
                                        training=self.training_pl,
                                        nlabels=self.nlabels,
                                        num_filters_first_layer=self.exp_config.n0,
                                        padding_type='SAME')

        self.y_pl_ = tf.nn.softmax(self.l_pl_)
        self.p_pl_ = tf.cast(tf.argmax(self.y_pl_, axis=-1), tf.int32)

        # Add to the Graph the Ops for loss calculation.

        self.task_loss = self.loss()
        self.weights_norm = self.weight_norm()

        self.total_loss = self.task_loss + self.exp_config.weight_decay * self.weights_norm

        self.global_step = tf.train.get_or_create_global_step()  # Used in batch renormalisation

        self.opt = grad_accum_optimizer_classifier(loss=self.total_loss,
                                                   optimizer=self._get_optimizer(),
                                                   variable_list=tf.trainable_variables(),
                                                   n_accum=exp_config.n_accum_grads)

        self.global_step = tf.train.get_or_create_global_step()
        self.increase_global_step = tf.assign(self.global_step, tf.add(self.global_step, 1))

        self.do_checkpoints = do_checkpoints
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        self.saver_best_xent = tf.train.Saver(max_to_keep=2)
        self.saver_best_dice = tf.train.Saver(max_to_keep=2)

        # Settings to optimize GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0

        # For evaluation
        self.eval_dice_per_structure = losses.get_dice(self.l_pl_, tf.one_hot(self.y_pl, depth=self.nlabels), sum_over_labels=False)
        self.eval_dice_per_image = losses.get_dice(self.l_pl_, tf.one_hot(self.y_pl, depth=self.nlabels), sum_over_labels=True)

        # Create a session for running Ops on the Graph.
        self.sess = tf.Session(config=config)

    def loss(self):

        y_for_loss = tf.one_hot(self.y_pl, depth=self.nlabels)

        if self.exp_config.loss_type == 'crossentropy':
            task_loss = losses.cross_entropy_loss(logits=self.l_pl_, labels=y_for_loss)
        elif self.exp_config.loss_type == 'dice_micro':
            task_loss = losses.dice_loss(logits=self.l_pl_, labels=y_for_loss, log_loss=self.log_loss, mode='micro')
        elif self.exp_config.loss_type == 'dice_macro':
            task_loss = losses.dice_loss(logits=self.l_pl_, labels=y_for_loss, log_loss=self.log_loss, mode='macro')
        elif self.exp_config.loss_type == 'dice_macro_robust':
            task_loss = losses.dice_loss(logits=self.l_pl_, labels=y_for_loss, log_loss=self.log_loss, mode='macro_robust')
        else:
            raise ValueError("Unknown loss_type in exp_config: '%s'" % self.exp_config.loss_type)

        return task_loss

    def weight_norm(self):

        weights_norm = tf.reduce_sum(
            input_tensor= tf.stack(
                [tf.nn.l2_loss(ii) for ii in tf.get_collection('weight_variables')]
            ),
            name='weights_norm'
        )

        return weights_norm


    def train(self):

        # Sort out proper logging
        self._setup_log_dir_and_continue_mode()

        # Create tensorboard summaries
        self._make_tensorboard_summaries()

        self.curr_lr = self.exp_config.learning_rate
        schedule_lr = True if self.exp_config.divide_lr_frequency is not None else False

        logging.info('===== RUNNING EXPERIMENT ========')
        logging.info(self.exp_config.experiment_name)
        logging.info('=================================')

        real_start_time = time.time()

        # initialise all weights etc..
        self.sess.run(tf.global_variables_initializer())

        # Restore session if there is one
        if self.continue_run:
            self.saver.restore(self.sess, self.init_checkpoint_path)

        logging.info('Starting training:')

        best_val = np.inf
        best_dice_score = 0

        # use deque for smoothing the validation score
        val_deque = deque([np.inf] * 5, maxlen=5)
        dice_deque = deque([0] * 5, maxlen=5)

        num_samples = self.data.nr_images
        iterations_per_epoch = num_samples / self.exp_config.batch_size

        for step in range(self.init_step, int(self.exp_config.num_epochs * iterations_per_epoch)):

            # If learning rate is scheduled
            if self.exp_config.warmup_training:
                if step < 50:
                    self.curr_lr = self.exp_config.learning_rate / 10.0
                elif step == 50:
                    self.curr_lr = self.exp_config.learning_rate

            if schedule_lr and step > 0 and step % self.exp_config.divide_lr_frequency == 0:
                self.curr_lr /= 10.0
                logging.info('Updating learning rate to: %f' % self.curr_lr)

            batch_x_dims = [self.exp_config.batch_size] + list(self.exp_config.image_size) + [1]
            batch_y_dims = [self.exp_config.batch_size] + list(self.exp_config.image_size) + [1]
            feed_dict = {self.x_pl: np.zeros(batch_x_dims),  # dummy variables will be replaced in optimizer
                         self.y_pl: np.zeros(batch_y_dims),
                         self.training_pl: True,
                         self.lr_pl: self.curr_lr}

            start_time = time.time()
            loss_value = self.opt.do_training_step(sess=self.sess,
                                                   sampler=self.data.train.next_batch,
                                                   batch_size=self.exp_config.batch_size,
                                                   feed_dict=feed_dict,
                                                   img_pl=self.x_pl,
                                                   lbl_pl=self.y_pl,
                                                   loss=self.total_loss)
            elapsed_time = time.time() - start_time


            ###  Tensorboard updates, Model Saving, and Validation

            # Update tensorboard
            if step % 5 == 0:

                logging.info('Step %d: loss = %.2f (One update step took %.3f sec)' % (step, loss_value, elapsed_time))

                summary_str = self.sess.run(self.summary, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()

            # Do training evaluation
            if (step + 1) % (self.exp_config.train_eval_frequency * iterations_per_epoch) == 0:

                # Evaluate against the training set
                logging.info('Training Data Eval:')
                self._do_validation(self.data.train,
                                    self.train_summary,
                                    self.train_error,
                                    self.train_tot_dice_score,
                                    self.train_mean_dice_score,
                                    self.train_lbl_dice_scores)

            # Do validation set evaluation
            if (step + 1) % (self.exp_config.val_eval_frequency * iterations_per_epoch)== 0:

                if self.do_checkpoints:
                    checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')
                    self.saver.save(self.sess, checkpoint_file, global_step=step)

                # Evaluate against the validation set.
                logging.info('Validation Data Eval:')

                val_loss, val_dice = self._do_validation(self.data.validation,
                                                         self.val_summary,
                                                         self.val_error,
                                                         self.val_tot_dice_score,
                                                         self.val_mean_dice_score,
                                                         self.val_lbl_dice_scores)

                dice_deque.append(val_dice)
                val_deque.append(val_loss)

                smoothed_dice = np.mean(dice_deque)
                smoothed_val = np.mean(val_deque)

                if smoothed_dice >= best_dice_score:
                    best_dice_score = smoothed_dice
                    best_step = step
                    best_file = os.path.join(self.log_dir, 'model_best_dice.ckpt')
                    self.saver_best_dice.save(self.sess, best_file, global_step=step)
                    logging.info( 'Found new best Dice score on validation set! - %f -  Saving model_best_dice.ckpt' % smoothed_dice)

                if smoothed_val < best_val:
                    best_val = smoothed_val
                    best_file = os.path.join(self.log_dir, 'model_best_xent.ckpt')
                    self.saver_best_xent.save(self.sess, best_file, global_step=step)
                    logging.info('Found new best crossentropy on validation set! - %f -  Saving model_best_xent.ckpt' % val_loss)

            self.sess.run(self.increase_global_step)

        final_time = time.time() - real_start_time
        print('Training took a total of {} hours'.format(final_time / 3600.0))
        # print('Number of trainable Parameters in model: ' + self.sess.run(str(count_params)))
        print('')
        print('Best Dice Score on Validation set is: {} at Epoch {}'.format(best_dice_score, int(best_step / iterations_per_epoch)))

        # loading weights and other stuff, function from Christian Baumgartner's discriminative learning toolbox

    def load_weights(self, log_dir=None, type='latest', **kwargs):

        if not log_dir:
            log_dir = self.log_dir

        if type == 'latest':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
        elif type == 'best_dice':
            init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(log_dir, 'model_best_dice.ckpt')
        else:
            raise ValueError('Argument type=%s is unknown. type can be latest/best_dice' % type)

        print('Loaded checkpoint of type ' + str(type) + ' from log directory ' + str(self.log_dir))
        self.saver.restore(self.sess, init_checkpoint_path)


    def predict(self, images):

        prediction, softmax = self.sess.run([self.p_pl_, self.y_pl_],
                                            feed_dict={self.x_pl: images, self.training_pl: False})

        return prediction, softmax

    def test(self, batch_size=1, num_sample_volumes=2, checkpoint = 'best_dice', datatype=None, set='test',
             gen_img=False, group_label=None, group_pred=None):
        """
        Used when testing the trained model, calculates the defined metrics for the input set (train / val / test) and has option
        to create new fake datasets.
        :param batch_size:
        :param num_sample_volumes:
        :param checkpoint:
        :param heart_data:
        :param datatype:
        :param gen_img:
        :return:
        """
        self.log_dir = os.path.join('./logs', self.exp_config.log_name, self.exp_config.fold_name)
        datatype = datatype
        dtype = np.uint8

        summary_dict = {}
        summary_dict['nr_samples'] = num_sample_volumes

        if set == 'train':
            print('INFO:   Evaluating Test results for Training data')
            print('')
        elif set == 'validation':
            print('INFO:   Evaluating Test results for Validation data')
            print('')
        elif set == 'test':
            print('INFO:   Evaluating Test results for Test data')
            print('')
        else:
            raise ValueError('Need to specify either train, validation or test split')

        if gen_img:
            fold_grp = group_label.create_group(self.exp_config.fold_name)
            fake_fold_grp = group_pred.create_group(self.exp_config.fold_name)

        self.load_weights(type=checkpoint)

        num_samples = self.data.nr_images * self.data.aug_factor
        num_iterations = num_samples // batch_size

        tot_dice = 0
        tot_dice_all = np.zeros(shape=(3))
        tot_corr = 0
        sample_vol_idx_list = []

        for sample in range(num_sample_volumes):
            print('INFO:   Currently iterating through sample volume {} (non-coded)'.format(sample))
            dice = 0
            dice_all = np.zeros(shape=(3))
            frac_list_b = []
            frac_list_fake_b = []
            num_batches = 0

            labels = np.zeros(shape=self.image_tensor_shape, dtype=np.uint8)
            pred = np.zeros(shape=self.labels_tensor_shape, dtype=np.uint8)

            # generate nested dict with information
            summary_dict[sample] = {}

            for iteration in range(int(num_iterations)):

                if iteration % 500 == 0:
                    print('INFO:   Currently at iteration {} / {}'.format(iteration, int(num_iterations)))

                # iterate also over number of sample volumes
                if set == 'train':
                    x, y, sample_vol_idx = self.data.train.test_image(img_idx=iteration,
                                                                                   batch_size=batch_size,
                                                                                   sample_vol=sample)
                elif set == 'validation':
                    x, y, sample_vol_idx = self.data.validation.test_image(img_idx=iteration,
                                                                                   batch_size=batch_size,
                                                                                   sample_vol=sample)
                elif set == 'test':
                    x, y, sample_vol_idx = self.data.test.test_image(img_idx=iteration,
                                                                                   batch_size=batch_size,
                                                                                   sample_vol=sample)
                else:
                    raise ValueError('Need to specify either train, validation or test split')

                assert x.shape[0] == y.shape[0] == 1, print('Shape should be 1 in first dimension')


                prediction, softmax = self.sess.run([self.p_pl_, self.y_pl_],
                                                    feed_dict={self.x_pl: x, self.training_pl: False})


                if gen_img:
                    pred[iteration, :, :, ] = prediction
                    labels[iteration,:,:,] = y

                flat_label = y.flatten()
                flat_pred = prediction.flatten()

                sk_f1_macro = f1_score(flat_label, flat_pred, average='macro')
                sk_f1_all = f1_score(flat_label, flat_pred, average=None)

                # obtain pearson corr for collagen fraction
                if datatype == 'heart':
                    epsi = 1e-10
                    assert y.shape[0] == 1

                    for i in range(y.shape[0]):
                        coll_b = np.count_nonzero(y[i, :, :] == 1)
                        coll_fake_b = np.count_nonzero(prediction[i, :, :] == 1)

                        cells_b = np.count_nonzero(y[i, :, :] == 2)
                        cells_fake_b = np.count_nonzero(prediction[i, :, :] == 2)

                        fraction_b = coll_b / (epsi + cells_b)
                        fraction_fake_b = coll_fake_b / (epsi + cells_fake_b)

                    frac_list_b.append(fraction_b)
                    frac_list_fake_b.append(fraction_fake_b)

                dice += sk_f1_macro
                dice_all += sk_f1_all

                num_batches += 1

                # get the mean dice score for the dataset
            mean_dice_score = dice / num_batches
            mean_dice_all = dice_all / num_batches

            corr, _ = pearsonr(frac_list_b, frac_list_fake_b)

            print('INFO:  Indiv. Dice Scores of Sample {} are: {}'.format(sample, mean_dice_all))
            print('INFO:  Mean Dice Score of Sample {} is: {}'.format(sample, mean_dice_score))
            print('INFO:  Mean Pearson Corr. of Sample {} is: {}'.format(sample, corr))

            sample_vol_idx_list.append(sample_vol_idx)

            summary_dict[sample]['dice'] = mean_dice_score
            summary_dict[sample]['all_dice'] = mean_dice_all
            summary_dict[sample]['corr'] = corr
            summary_dict[sample]['real_frac'] = frac_list_b
            summary_dict[sample]['fake_frac'] = frac_list_fake_b
            summary_dict[sample]['sample_idx'] = sample_vol_idx

            tot_dice += mean_dice_score
            tot_dice_all += mean_dice_all
            tot_corr += corr

            if gen_img:
                fold_grp.create_dataset(name='data_' + str(sample_vol_idx), data=labels, dtype=dtype)
                fake_fold_grp.create_dataset(name='data_' + str(sample_vol_idx), data=pred, dtype=dtype)

        tot_dice = tot_dice / num_sample_volumes
        tot_dice_all = tot_dice_all / num_sample_volumes

        tot_corr = tot_corr / num_sample_volumes

        print('INFO:  Total Indiv. Dice: {}'.format(tot_dice_all))
        print('INFO:  Total Mean Dice: {}'.format(tot_dice))
        print('INFO:  Total Mean Pearson Corr: {}'.format(tot_corr))

        return summary_dict


    ### HELPER FUNCTIONS ###################################################################################

    def _make_tensorboard_summaries(self):

        ### Batch-wise summaries

        tf.summary.scalar('learning_rate', self.lr_pl)

        tf.summary.scalar('task_loss', self.task_loss)
        tf.summary.scalar('weights_norm', self.weights_norm)
        tf.summary.scalar('total_loss', self.total_loss)

        def _image_summaries(prefix, x, y, y_gt):

            if len(self.image_tensor_shape) == 5:
                data_dimension = 3
            elif len(self.image_tensor_shape) == 4:
                data_dimension = 2
            else:
                raise ValueError('Invalid image dimensions')

            if data_dimension == 3:
                y_disp = tf.expand_dims(y[:, :, :, self.exp_config.tensorboard_slice], axis=-1)
                y_gt_disp = tf.expand_dims(y_gt[:, :, :, self.exp_config.tensorboard_slice], axis=-1)
                x_disp = x[:, :, :, self.exp_config.tensorboard_slice, :]
            else:
                y_disp = tf.expand_dims(y, axis=-1)
                y_gt_disp = tf.expand_dims(y_gt, axis=-1)
                x_disp = x

            sum_y = tf.summary.image('%s_mask_predicted' % prefix, tf_utils.put_kernels_on_grid(
                y_disp,
                batch_size=self.exp_config.batch_size,
                mode='mask',
                nlabels=self.exp_config.nlabels))
            sum_y_gt = tf.summary.image('%s_mask_groundtruth' % prefix, tf_utils.put_kernels_on_grid(
                y_gt_disp,
                batch_size=self.exp_config.batch_size,
                mode='mask',
                nlabels=self.exp_config.nlabels))
            sum_x = tf.summary.image('%s_input_image' % prefix, tf_utils.put_kernels_on_grid(
                x_disp,
                min_int=None,
                max_int=None,
                batch_size=self.exp_config.batch_size,
                mode='image'))

            return tf.summary.merge([sum_y, sum_y_gt, sum_x])

        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()

        ### Validation summaries

        self.val_error = tf.placeholder(tf.float32, shape=[], name='val_task_loss')
        val_error_summary = tf.summary.scalar('validation_task_loss', self.val_error)

        # Note: Total dice is the Dice over all pixels of an image
        #       Mean Dice is the mean of the per label dices, which is not affected by class imbalance

        self.val_tot_dice_score = tf.placeholder(tf.float32, shape=[], name='val_dice_total_score')
        val_tot_dice_summary = tf.summary.scalar('validation_dice_tot_score', self.val_tot_dice_score)

        self.val_mean_dice_score = tf.placeholder(tf.float32, shape=[], name='val_dice_mean_score')
        val_mean_dice_summary = tf.summary.scalar('validation_dice_mean_score', self.val_mean_dice_score)

        self.val_lbl_dice_scores = []
        val_lbl_dice_summaries = []
        for ii in range(self.nlabels):
            curr_pl = tf.placeholder(tf.float32, shape=[], name='val_dice_lbl_%d' % ii)
            self.val_lbl_dice_scores.append(curr_pl)
            val_lbl_dice_summaries.append(tf.summary.scalar('validation_dice_lbl_%d' % ii, curr_pl))

        val_image_summary = _image_summaries('validation', self.x_pl, self.p_pl_, self.y_pl)

        self.val_summary = tf.summary.merge([val_error_summary,
                                             val_tot_dice_summary,
                                             val_mean_dice_summary,
                                             val_image_summary] + val_lbl_dice_summaries)

        ### Train summaries

        self.train_error = tf.placeholder(tf.float32, shape=[], name='train_task_loss')
        train_error_summary = tf.summary.scalar('training_task_loss', self.train_error)

        self.train_tot_dice_score = tf.placeholder(tf.float32, shape=[], name='train_dice_tot_score')
        train_tot_dice_summary = tf.summary.scalar('train_dice_tot_score', self.train_tot_dice_score)

        self.train_mean_dice_score = tf.placeholder(tf.float32, shape=[], name='train_dice_mean_score')
        train_mean_dice_summary = tf.summary.scalar('train_dice_mean_score', self.train_mean_dice_score)

        self.train_lbl_dice_scores = []
        train_lbl_dice_summaries = []
        for ii in range(self.nlabels):
            curr_pl = tf.placeholder(tf.float32, shape=[], name='train_dice_lbl_%d' % ii)
            self.train_lbl_dice_scores.append(curr_pl)
            train_lbl_dice_summaries.append(tf.summary.scalar('train_dice_lbl_%d' % ii, curr_pl))

        train_image_summary = _image_summaries('train', self.x_pl, self.p_pl_, self.y_pl)

        self.train_summary = tf.summary.merge([train_error_summary,
                                               train_tot_dice_summary,
                                               train_mean_dice_summary,
                                               train_image_summary] + train_lbl_dice_summaries)


    def _get_optimizer(self):

        if self.exp_config.optimizer_handle == tf.train.AdamOptimizer:
            return self.exp_config.optimizer_handle(learning_rate=self.lr_pl,
                                                    beta1=self.exp_config.beta1,
                                                    beta2=self.exp_config.beta2)
        if self.exp_config.momentum is not None:
            return self.exp_config.optimizer_handle(learning_rate=self.lr_pl,
                                                    momentum=self.exp_config.momentum)
        else:
            return self.exp_config.optimizer_handle(learning_rate=self.lr_pl)

    def _setup_log_dir_and_continue_mode(self):

        # Default values
        self.log_dir = os.path.join('./logs', self.exp_config.log_name, self.exp_config.fold_name)
        self.init_checkpoint_path = None
        self.continue_run = False
        self.init_step = 0

        if self.do_checkpoints:
            # If a checkpoint file already exists enable continue mode
            if tf.gfile.Exists(self.log_dir):
                init_checkpoint_path = tf_utils.get_latest_model_checkpoint_path(self.log_dir, 'model.ckpt')
                if init_checkpoint_path is not False:
                    self.init_checkpoint_path = init_checkpoint_path
                    self.continue_run = True
                    self.init_step = int(self.init_checkpoint_path.split('/')[-1].split('-')[-1])
                    self.log_dir += '_cont'

                    logging.info(
                        '--------------------------- Continuing previous run --------------------------------')
                    logging.info('Checkpoint path: %s' % self.init_checkpoint_path)
                    logging.info('Latest step was: %d' % self.init_step)
                    logging.info(
                        '------------------------------------------------------------------------------------')

        tf.gfile.MakeDirs(self.log_dir)
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)


    def _eval_predict(self, images, labels):

        prediction, loss, dice_per_img_and_lbl, dice_per_img = self.sess.run([self.p_pl_,
                                                                              self.total_loss,
                                                                              self.eval_dice_per_structure,
                                                                              self.eval_dice_per_image],
                                                                             feed_dict={self.x_pl: images,
                                                                             self.y_pl: labels,
                                                                             self.training_pl: False})

        # We are getting per_image and per_structure dice separately because the average of per_structure
        # will be wrong for labels that do not appear in an image.

        return prediction, loss, dice_per_img_and_lbl, dice_per_img


    def _do_validation(self, data_handle, summary, tot_loss_pl, tot_dice_pl, mean_dice_pl, lbl_dice_pl_list):

        diag_loss_ii = 0
        num_batches = 0
        all_dice_per_img_and_lbl = []
        all_dice_per_img = []


        for batch in data_handle.iterate_batches(self.exp_config.batch_size):

            x, y = batch

            # Skip incomplete batches
            if y.shape[0] < self.exp_config.batch_size:
                continue

            c_d_preds, c_d_loss, dice_per_img_and_lbl, dice_per_img = self._eval_predict(x, y)

            num_batches += 1
            diag_loss_ii += c_d_loss

            all_dice_per_img_and_lbl.append(dice_per_img_and_lbl)
            all_dice_per_img.append(dice_per_img)

        avg_loss = (diag_loss_ii / num_batches)

        dice_per_lbl_array = np.asarray(all_dice_per_img_and_lbl).reshape((-1, self.nlabels))

        per_structure_dice = np.mean(dice_per_lbl_array, axis=0)

        dice_array = np.asarray(all_dice_per_img).flatten()

        avg_dice = np.mean(dice_array)

        ### Update Tensorboard
        x, y = data_handle.next_batch(self.exp_config.batch_size)
        feed_dict = {}
        feed_dict[self.training_pl] = False
        feed_dict[self.x_pl] = x
        feed_dict[self.y_pl] = y
        feed_dict[tot_loss_pl] = avg_loss
        feed_dict[tot_dice_pl] = avg_dice
        feed_dict[mean_dice_pl] = np.mean(per_structure_dice)
        for ii in range(self.nlabels):
            feed_dict[lbl_dice_pl_list[ii]] = per_structure_dice[ii]

        summary_msg = self.sess.run(summary, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_msg, global_step=self.sess.run(tf.train.get_global_step()))

        ### Output logs
        # Note: Total dice is the Dice over all pixels of an image
        #       Mean Dice is the mean of the per label dices, which is not affected by class imbalance
        logging.info('  Average loss: %0.04f' % avg_loss)
        logging.info('  Total Dice: %0.04f' % avg_dice)
        logging.info('  Mean Dice: %0.04f' % np.mean(per_structure_dice))
        for ii in range(self.nlabels):
            logging.info('  Dice lbl %d: %0.04f' % (ii, per_structure_dice[ii]))
        logging.info('---')

        return avg_loss, np.mean(per_structure_dice)



