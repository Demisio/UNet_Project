# Author:
# Christine Tanner
# mostly from
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import numpy as np
import code

##########
#TODO: add Focal Loss with crossentropy
##########

def dice_loss(logits, labels, epsilon=1e-10, from_label=1, to_label=-1):
    '''
    Calculate a dice loss defined as `1-foreround_dice`. Default mode assumes that the 0 label
     denotes background and the remaining labels are foreground. 
    :param logits: Network output before softmax
    :param labels: ground truth label masks
    :param epsilon: A small constant to avoid division by 0
    :param from_label: First label to evaluate 
    :param to_label: Last label to evaluate
    :return: Dice loss
    '''
    with tf.name_scope('dice_loss'):

        prediction = tf.nn.softmax(logits)

        intersection = tf.multiply(prediction, labels)
        intersec_per_lab = tf.reduce_sum(intersection, axis=[0, 1, 2])

        l = tf.reduce_sum(prediction, axis=[0, 1, 2])
        r = tf.reduce_sum(labels, axis=[0, 1, 2])


        dices_per_lbl = 2 * intersec_per_lab / (l + r + epsilon)

        loss = 1- tf.reduce_mean(dices_per_lbl[from_label:to_label])

    return loss

def log_dice_loss(logits, labels, epsilon=1e-10, from_label=1, to_label=-1):
    
    with tf.name_scope('log_dice_loss'):

        prediction = tf.nn.softmax(logits)

        intersection = tf.multiply(prediction, labels)
        intersec_per_lab = tf.reduce_sum(intersection, axis=[0, 1, 2])

        l = tf.reduce_sum(prediction, axis=[0, 1, 2])
        r = tf.reduce_sum(labels, axis=[0, 1, 2])

        log_dices_per_lbl = -tf.log(2 * intersec_per_lab / (l + r + epsilon))

        loss = tf.reduce_mean(log_dices_per_lbl[from_label:to_label])
    
    return loss




def mean_dice(prediction, ground_truth, nr_labels, epsilon=1e-10, sum_over_batches=True, partial_dice=False, from_label=0, to_label=-1):
    '''
    Mean dice calculated from all voxels from minibatch, integrates macro averaging over batches. This makes the method more robust
    than with a single image as chance of missing label is smaller. Macro averaging automatically handles class imbalance,
    as each label contributes equally towards final average score

    :param prediction: network output
    :param ground_truth: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :return: scalar Dice
    '''

    sum_over_batches = sum_over_batches
    struct_dice = get_dice(prediction=prediction,
                           ground_truth=ground_truth,
                           nr_labels= nr_labels,
                           epsilon=epsilon,
                           sum_over_batches=sum_over_batches)

    if partial_dice:
        selected_dice = struct_dice[..., from_label:to_label]
    # foreground_dice = tf.slice(struct_dice, (0, from_label),(-1, to_label))
    else:
        selected_dice = struct_dice

    mean_dice_score = tf.reduce_mean(selected_dice)

    return mean_dice_score


def get_dice(prediction, ground_truth, nr_labels, epsilon=1e-10, sum_over_batches=False):
    '''
    Dice coefficient per label

    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :param sum_over_batches: Calculate intersection and union over whole batch rather than single images
    :return: tensor shaped (tf.shape(logits)[0], tf.shape(logits)[-1])
    '''

    ndims = prediction.get_shape().ndims

    ## From original implementation, had 3 channels with probabilities for each class, now single channel with numbers
    ## Therefore, only one hot needed
    # prediction = tf.nn.softmax(logits)
    # hard_pred = tf.one_hot(tf.argmax(prediction, axis=-1), depth=tf.shape(prediction)[-1])
    int_pred = tf.cast(prediction, dtype=tf.int32)
    int_labels = tf.cast(ground_truth, dtype=tf.int32)

    # have to one hot encode both predictions and labels for this to work
    hard_pred = tf.one_hot(int_pred, nr_labels)
    labels = tf.one_hot(int_labels, nr_labels)

    intersection = tf.multiply(hard_pred, labels)

    if ndims == 5:
        reduction_axes = [1, 2, 3]
    else:
        reduction_axes = [1, 2]

    if sum_over_batches:
        reduction_axes = [0] + reduction_axes # 0 indicates reduction over batches

    intersec_per_lab = tf.reduce_sum(intersection, axis=reduction_axes)  # was [1,2]

    l = tf.reduce_sum(hard_pred, axis=reduction_axes)
    r = tf.reduce_sum(labels, axis=reduction_axes)

    dices_per_lab = 2 * intersec_per_lab / (l + r + epsilon)

    return dices_per_lab


def mean_dice_fancy(prediction, ground_truth, epsilon=1e-10, **kwargs):
    '''
    The dice loss is always 1 - dice, however, there are many ways to calculate the dice. Basically, there are
    three sums involved: 1) over the pixels, 2) over the labels, 3) over the images in a batch. These sums
    can be arranged differently to obtain different behaviour. The behaviour can be controlled either by providing
    the 'mode' variable, or by setting the parameters directly.

    Selecting the parameters directly:
    :param per_structure: <True|False> If True the Dice is calculated for each label separately first and then averaged.
    :param sum_over_batches: <True|False> If True the Dice is calculated for each batch separately then averaged.

    Selecting the mode:
    :param mode: <'macro'|'macro_robust'|'micro'>
                     macro: Calculate Dice for each label separately then average. This may cause problems
                            if a structure is completely missing from the image. Even if correctly predicted
                            the dice will evaluate to 0/epsilon = 0. However, this method automatically tackles
                            class imbalance, because each structure contributes equally to the final Dice.
                     macro_robust: The above calculation can be made more robust by summing over all images in a
                                   minibatch. If the label appear at least in one image in the batch and is perfectly
                                   predicted, the Dice will evaluate to 1 as expected.
                     micro: Calculate Dice for all labels together. This doesn't have the problems of macro for missing
                            labels. However, it is sensitive to class imbalance because each label contributes
                            by how often it appears in the data.

    The above are equivalent to F1 score in macro/micro mode (see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

    Other parameters:
    :param logits: Network output
    :param labels: Ground-truth labels
    :param only_foreground: <True|False> Sometimes it can be beneficial to ignore label 0 for the optimisation
    :epsilon: <float> To avoid division by zero in Dice calculation.

    '''

    only_foreground = kwargs.get('only_foreground', False)
    mode = kwargs.get('mode', None)
    if mode == 'macro':
        sum_over_labels = False
        sum_over_batches = False
    elif mode == 'macro_robust':
        sum_over_labels = False
        sum_over_batches = True
    elif mode == 'micro':
        sum_over_labels = True
        sum_over_batches = False
    elif mode is None:
        sum_over_labels = kwargs.get('per_structure')  # Intentionally no default value
        sum_over_batches = kwargs.get('sum_over_batches', False)
    else:
        raise ValueError("Encountered unexpected 'mode' in dice_loss: '%s'" % mode)

    with tf.name_scope('dice_loss'):

        dice_per_img_per_lab = get_dice(prediction=prediction,
                                        ground_truth=ground_truth,
                                        epsilon=epsilon,
                                        sum_over_labels=sum_over_labels,
                                        sum_over_batches=sum_over_batches,
                                        use_hard_pred=False)

        if only_foreground:
            if sum_over_batches:
                loss = 1 - tf.reduce_mean(dice_per_img_per_lab[1:])
            else:
                loss = 1 - tf.reduce_mean(dice_per_img_per_lab[:, 1:])
        else:
            loss = 1 - tf.reduce_mean(dice_per_img_per_lab)

    return loss



def pixel_wise_cross_entropy_loss(logits, labels):
    '''
    Simple wrapper for the normal tensorflow cross entropy loss 
    '''

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss


def pixel_wise_cross_entropy_loss_weighted(logits, labels, class_weights):
    '''
    Weighted cross entropy loss, with a weight per class
    :param logits: Network output before softmax
    :param labels: Ground truth masks
    :param class_weights: A list of the weights for each class
    :return: weighted cross entropy loss
    '''

    n_class = len(class_weights)

    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(labels, [-1, n_class])

    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

    weight_map = tf.multiply(flat_labels, class_weights)
    weight_map = tf.reduce_sum(weight_map, axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
    weighted_loss = tf.multiply(loss_map, weight_map)

    loss = tf.reduce_mean(weighted_loss)

    return loss


def abs_criterion(in_, target):
    # mean absolute difference
    #code.interact(local=dict(globals(), **locals()))
    return tf.reduce_mean(tf.abs(in_ - target))


def msd_criterion(in_, target):
    # mean squared difference
    return tf.reduce_mean((in_-target)**2)
