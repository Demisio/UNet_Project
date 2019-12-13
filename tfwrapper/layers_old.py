# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import numpy as np
import logging
from tfwrapper import utils
from tfwrapper import normalisation as tfnorm
# import high_dim_filter_loader

# custom_module = high_dim_filter_loader.custom_module

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Will be used as default in all the layers below
STANDARD_NONLINEARITY = tf.nn.relu

## Pooling layers ##

def maxpool2D(x, kernel_size=(2, 2), strides=(2, 2), padding="SAME"):
    '''
    nets2D max pooling layer with standard 2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], 1]
    strides_aug = [1, strides[0], strides[1], 1]

    op = tf.nn.max_pool(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op


def maxpool3D(x, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="SAME"):
    '''
    nets3D max pooling layer with 2x2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], kernel_size[2], 1]
    strides_aug = [1, strides[0], strides[1], strides[2], 1]

    op = tf.nn.max_pool3d(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op


def avgpool3D(x, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME'):
    '''
    nets3D avg pooling layer with 2x2x2 pooling as default
    '''
    kernel_size_aug = [1, kernel_size[0], kernel_size[1], kernel_size[2], 1]
    strides_aug = [1, strides[0], strides[1], strides[2], 1]

    op = tf.nn.avg_pool3d(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op


def global_averagepool2D(x, name=None):
    '''
    nets3D max pooling layer with 2x2x2 pooling as default
    '''

    op = tf.reduce_mean(x, axis=(1,2), keepdims=False, name=name)
    tf.summary.histogram(op.op.name + '/activations', op)

    return op


def global_averagepool3D(x, name=None):
    '''
    nets3D max pooling layer with 2x2x2 pooling as default
    '''

    op = tf.reduce_mean(x, axis=(1,2,3), keepdims=False, name=name)
    tf.summary.histogram(op.op.name + '/activations', op)

    return op


## Standard feed-forward layers ##

def conv2D(x,
           name,
           kernel_size=(3,3),
           num_filters=32,
           strides=(1,1),
           activation=STANDARD_NONLINEARITY,
           normalisation=tf.identity,
           normalise_post_activation=False,
           dropout_p=None,
           padding="SAME",
           weight_init='he_normal',
           add_bias=True,
           **kwargs):

    '''
    Standard nets2D convolutional layer
    kwargs can have training, and potentially other normalisation paramters
    '''

    bottom_num_filters = x.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], 1]

    with tf.variable_scope(name):

        weights = utils.get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        op = tf.nn.conv2d(x, filter=weights, strides=strides_augm, padding=padding)

        biases = None  # so there is always something for summary
        if add_bias:
            biases = utils.get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)

        if not normalise_post_activation:
            op = activation(normalisation(op, **kwargs))
        else:
            op = normalisation(activation(op), **kwargs)

        if dropout_p is not None:
            op = dropout(op, keep_prob=dropout_p, **kwargs)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def conv3D(x,
           name,
           kernel_size=(3,3,3),
           num_filters=32,
           strides=(1,1,1),
           activation=STANDARD_NONLINEARITY,
           normalisation=tf.identity,
           normalise_post_activation=False,
           dropout_p=None,
           padding="SAME",
           weight_init='he_normal',
           add_bias=True,
           **kwargs):

    '''
    Standard nets3D convolutional layer
    '''

    bottom_num_filters = x.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.variable_scope(name):
        stddev = kwargs.pop('stddev', 0.02)
        init_weights = kwargs.pop('init_weights', None)
        trainable = kwargs.pop('trainable', True)
        weights = utils.get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True, stddev=stddev, init_weights=init_weights, trainable=trainable)
        op = tf.nn.conv3d(x, filter=weights, strides=strides_augm, padding=padding)

        biases = None
        if add_bias:
            init_biases = kwargs.pop('init_biases', None)
            biases = utils.get_bias_variable(bias_shape, name='b', init_biases=init_biases, trainable=trainable)
            op = tf.nn.bias_add(op, biases)

        if normalisation == tf.identity:
            kwargs.pop('training', None)
        if not normalise_post_activation:
            op = activation(normalisation(op, **kwargs))
        else:
            op = normalisation(activation(op), **kwargs)

        if dropout_p is not None:
            op = dropout(op, keep_prob=dropout_p, **kwargs)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def transposed_conv2D(bottom,
                      name,
                      kernel_size=(4,4),
                      num_filters=32,
                      strides=(2,2),
                      output_shape=None,
                      activation=STANDARD_NONLINEARITY,
                      normalisation=tf.identity,
                      normalise_post_activation=False,
                      dropout_p=None,
                      padding="SAME",
                      weight_init='he_normal',
                      add_bias=True,
                      **kwargs):

    '''
    Standard nets2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2.
    '''

    bottom_shape = bottom.get_shape().as_list()
    if output_shape is None:
        batch_size = tf.shape(bottom)[0]
        output_shape = tf.stack([batch_size, bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], num_filters])

    bottom_num_filters = bottom_shape[3]

    weight_shape = [kernel_size[0], kernel_size[1], num_filters, bottom_num_filters]
    bias_shape = [num_filters]
    strides_augm = [1, strides[0], strides[1], 1]

    with tf.variable_scope(name):

        weights = utils.get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.nn.conv2d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)

        # The line below is hack necessary to fix a bug with tensorflow. The same operation is not required
        # for the 3D equivalent of this layer.
        op = tf.reshape(op, output_shape)

        biases = None
        if add_bias:
            biases = utils.get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)

        if not normalise_post_activation:
            op = activation(normalisation(op, **kwargs))
        else:
            op = normalisation(activation(op), **kwargs)

        if dropout_p is not None:
            op = dropout(op, keep_prob=dropout_p, **kwargs)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def transposed_conv3D(bottom,
                      name,
                      kernel_size=(4,4,4),
                      num_filters=32,
                      strides=(2,2,2),
                      output_shape=None,
                      activation=STANDARD_NONLINEARITY,
                      normalisation=tf.identity,
                      normalise_post_activation=False,
                      dropout_p=None,
                      padding="SAME",
                      weight_init='he_normal',
                      add_bias=True,
                      **kwargs):

    '''
    Standard nets2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2.
    '''

    bottom_shape = bottom.get_shape().as_list()

    if output_shape is None:
        batch_size = tf.shape(bottom)[0]
        output_shape = tf.stack([batch_size, bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], bottom_shape[3]*strides[2], num_filters])

    bottom_num_filters = bottom_shape[4]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], num_filters, bottom_num_filters]

    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.variable_scope(name):

        weights = utils.get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.nn.conv3d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)

        # op = tf.reshape(op, output_shape)

        biases = None
        if add_bias:
            biases = utils.get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)

        if not normalise_post_activation:
            op = activation(normalisation(op, **kwargs))
        else:
            op = normalisation(activation(op), **kwargs)

        if dropout_p is not None:
            op = dropout(op, keep_prob=dropout_p, **kwargs)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def bilinear_upsample2D(x, name, factor):

    bottom_shape = x.get_shape().as_list()
    output_size = tf.stack([bottom_shape[1] * factor, bottom_shape[2] * factor])

    with tf.variable_scope(name):

        op = tf.image.resize_images(x, output_size)

    return op


def bilinear_upsample3D(x, name, factor):

    # Taken from: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html

    with tf.variable_scope(name):

        b_size, x_size, y_size, z_size, c_size =  x.shape.as_list()

        x_size_new = x_size*factor
        y_size_new = y_size*factor
        z_size_new = z_size*factor

        # resize y-z
        squeeze_b_x = tf.reshape(x, [-1, y_size, z_size, c_size])
        resize_b_x = tf.image.resize_bilinear( squeeze_b_x, [y_size_new, z_size_new])
        resume_b_x = tf.reshape(resize_b_x, [b_size, x_size, y_size_new, z_size_new, c_size])

        # resize x
        #   first reorient
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])

        #   squeeze and 2d resize
        squeeze_b_z = tf.reshape( reoriented, [-1, y_size_new, x_size, c_size])
        resize_b_z = tf.image.resize_bilinear(squeeze_b_z, [y_size_new, x_size_new])
        resume_b_z = tf.reshape(resize_b_z, [b_size, z_size_new, y_size_new, x_size_new, c_size])

        output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])

    return output_tensor

def dilated_conv2D(bottom,
                   name,
                   kernel_size=(3,3),
                   num_filters=32,
                   rate=2,
                   activation=STANDARD_NONLINEARITY,
                   normalisation=tf.identity,
                   normalise_post_activation=False,
                   dropout_p=None,
                   padding="SAME",
                   weight_init='he_normal',
                   add_bias=True,
                   **kwargs):

    '''
    nets2D dilated convolution layer. This layer can be used to increase the receptive field of a network.
    It is described in detail in this paper: Yu et al, Multi-Scale Context Aggregation by Dilated Convolutions,
    2015 (https://arxiv.org/pdf/1511.07122.pdf)
    '''

    bottom_num_filters = bottom.get_shape().as_list()[3]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    with tf.variable_scope(name):

        weights = utils.get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.nn.atrous_conv2d(bottom, filters=weights, rate=rate, padding=padding)

        biases = None
        if add_bias:
            biases = utils.get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)

        if not normalise_post_activation:
            op = activation(normalisation(op, **kwargs))
        else:
            op = normalisation(activation(op), **kwargs)

        if dropout_p is not None:
            op = dropout(op, keep_prob=dropout_p, **kwargs)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def residual_unit2D(x,
                    name,
                    num_filters=32,
                    down_sample=False,
                    projection=False,
                    activation=STANDARD_NONLINEARITY,
                    normalisation=tfnorm.batch_norm,
                    add_bias=True,
                    **kwargs):

    """
    See https://arxiv.org/abs/1512.03385
    """

    bottom_num_filters = x.get_shape().as_list()[-1]

    adjust_nfilters = True if not bottom_num_filters == num_filters or down_sample else False

    if down_sample:
        first_strides = (2,2)
    else:
        first_strides = (1,1)

    with tf.variable_scope(name):

        conv1 = conv2D(x, 'conv1', num_filters=num_filters, strides=first_strides, activation=tf.identity, add_bias=add_bias)
        conv1 = normalisation(conv1, scope='bn1', **kwargs)
        conv1 = activation(conv1)

        conv2 = conv2D(conv1, 'conv2', num_filters=num_filters, activation=tf.identity, add_bias=add_bias)
        conv2 = normalisation(conv2, scope='bn2', **kwargs)
        # conv2 = activation(conv2)

        if adjust_nfilters:
            if projection:
                projection = conv2D(x, 'projection', num_filters=num_filters, strides=first_strides, kernel_size=(1, 1), activation=tf.identity, add_bias=add_bias)
                projection = normalisation(projection, scope='bn_projection', **kwargs)
                skip = activation(projection)
            else:
                pad_size = (num_filters - bottom_num_filters) // 2
                identity = tf.pad(x, paddings=[[0, 0], [0, 0], [0, 0], [pad_size, pad_size]])
                if down_sample:
                    identity = identity[:,::2,::2,:]
                skip = identity
        else:
            skip = x

        block = tf.add(skip, conv2, name='add')
        block = activation(block)

        return block


def identity_residual_unit2D(x,
                             name,
                             num_filters,
                             down_sample=False,
                             projection=True,
                             activation=STANDARD_NONLINEARITY,
                             normalisation=tfnorm.batch_norm,
                             add_bias=True,
                             **kwargs):

    """
    Better residual unit which should allow better gradient flow.
    See Identity Mappings in Deep Residual Networks (https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38)
    """

    bottom_num_filters = x.get_shape().as_list()[-1]

    if not projection:
        assert (bottom_num_filters == num_filters) or (bottom_num_filters*2 == num_filters), \
            'Number of filters must remain constant, or be increased by a ' \
            'factor of 2. In filters: %d, Out filters: %d' % (bottom_num_filters, num_filters)

    increase_nfilters = True if not bottom_num_filters == num_filters or down_sample else False

    if down_sample:
        first_strides = (2,2)
    else:
        first_strides = (1,1)

    with tf.variable_scope(name):

        op1 = normalisation(x, scope='bn1', **kwargs)
        op1 = activation(op1)
        op1 = conv2D(op1, 'conv1', num_filters=num_filters, strides=first_strides, activation=tf.identity, add_bias=add_bias)

        op2 = normalisation(op1, scope='bn2', **kwargs)
        op2 = activation(op2)
        op2 = conv2D(op2, 'conv2', num_filters=num_filters, activation=tf.identity, add_bias=add_bias)

        if increase_nfilters:
            if projection:
                projection = conv2D(x, 'projection', num_filters=num_filters, strides=first_strides, kernel_size=(1, 1), activation=tf.identity, add_bias=add_bias)
                projection = normalisation(projection, scope='bn_projection', **kwargs)
                skip = activation(projection)
            else:
                pad_size = (num_filters - bottom_num_filters) // 2
                identity = tf.pad(x, paddings=[[0, 0], [0, 0], [0, 0], [pad_size, pad_size]])
                if down_sample:
                    identity = identity[:,::2,::2,:]
                skip = identity
        else:
            skip = x

        block = tf.add(skip, op2, name='add')

        return block


def dense_layer(bottom,
                name,
                hidden_units=512,
                activation=STANDARD_NONLINEARITY,
                normalisation=tfnorm.batch_norm,
                normalise_post_activation=False,
                dropout_p=None,
                weight_init='he_normal',
                add_bias=True,
                **kwargs):

    '''
    Dense a.k.a. fully connected layer
    '''

    bottom_flat = utils.flatten(bottom)
    bottom_rhs_dim = utils.get_rhs_dim(bottom_flat)

    weight_shape = [bottom_rhs_dim, hidden_units]
    bias_shape = [hidden_units]

    with tf.variable_scope(name):

        weights = utils.get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.matmul(bottom_flat, weights)

        biases = None
        if add_bias:
            biases = utils.get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)

        if normalisation == tf.identity:
            kwargs.pop('training', None)
        if not normalise_post_activation:
            op = activation(normalisation(op, **kwargs))
        else:
            op = normalisation(activation(op), **kwargs)

        if dropout_p is not None:
            op = dropout(op, keep_prob=dropout_p, **kwargs)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op

## Other layers ##

def crop_and_concat(inputs, axis=-1):

    '''
    Layer for cropping and stacking feature maps of different size along a different axis.
    Currently, the first feature map in the inputs list defines the output size.
    The feature maps can have different numbers of channels.
    :param inputs: A list of input tensors of the same dimensionality but can have different sizes
    :param axis: Axis along which to concatentate the inputs
    :return: The concatentated feature map tensor
    '''

    output_size = inputs[0].get_shape().as_list()[1:]
    # output_size = tf.shape(inputs[0])[1:]
    concat_inputs = [inputs[0]]

    for ii in range(1, len(inputs)):

        larger_size = inputs[ii].get_shape().as_list()[1:]
        # larger_size = tf.shape(inputs[ii])

        # Don't subtract over batch_size because it may be None
        start_crop = np.subtract(larger_size, output_size) // 2

        if len(output_size) == 4:  # nets3D images
            cropped_tensor = tf.slice(inputs[ii],
                                      (0, start_crop[0], start_crop[1], start_crop[2], 0),
                                      (-1, output_size[0], output_size[1], output_size[2], -1))
        elif len(output_size) == 3:  # nets2D images
            cropped_tensor = tf.slice(inputs[ii],
                                      (0, start_crop[0], start_crop[1], 0),
                                      (-1, output_size[0], output_size[1], -1))
        else:
            raise ValueError('Unexpected number of dimensions on tensor: %d' % len(output_size))

        concat_inputs.append(cropped_tensor)

    return tf.concat(concat_inputs, axis=axis)


def pad_to_size(bottom, output_size):

    '''
    A layer used to pad the tensor bottom to output_size by padding zeros around it
    TODO: implement for nets3D data
    '''

    input_size = bottom.get_shape().as_list()
    size_diff = np.subtract(output_size, input_size)

    pad_size = size_diff // 2
    odd_bit = np.mod(size_diff, 2)

    if len(input_size) == 4:

        padded = tf.pad(bottom, paddings=[[0, 0],
                                          [pad_size[1], pad_size[1] + odd_bit[1]],
                                          [pad_size[2], pad_size[2] + odd_bit[2]],
                                          [0, 0]])

        return padded

    elif len(input_size) == 5:
        raise NotImplementedError('This layer has not yet been extended to nets3D')
    else:
        raise ValueError('Unexpected input size: %d' % input_size)


def dropout(bottom, keep_prob, training):
    '''
    Performs dropout on the activations of an input
    '''

    with tf.variable_scope('dropout_layer'):
        keep_prob_pl = tf.cond(training,
                               lambda: tf.constant(keep_prob, dtype=bottom.dtype),
                               lambda: tf.constant(1.0, dtype=bottom.dtype))

        # The tf.nn.dropout function takes care of all the scaling
        # (https://www.tensorflow.org/get_started/mnist/pros)
        return tf.nn.dropout(bottom, keep_prob=keep_prob_pl, name='dropout')


# def crfrnn(unary, rgb_g, num_classes,
#               theta_alpha, theta_beta, theta_gamma,
#               num_iterations, scope='crfrnn', **kwargs):
#
#     # Code by Yigit Baran Can (see https://link.springer.com/chapter/10.1007/978-3-030-00889-5_27)
#
#
#     #    shape=[num_classes,num_classes]
#     #    tf.initializers.random_uniform
#     #
#     #    shape_unary = tf.shape(unary)
#     #    shape_rgb = tf.shape(rgb)
#     #    print(str(shape_unary) + " : " + str(shape_rgb))
#     with tf.variable_scope(scope):
#         weights = {
#
#             'spatial_ker_weights': tf.get_variable('spatial_ker_weights',
#                                                    shape=[num_classes, num_classes],
#                                                    initializer=tf.random_uniform_initializer),
#             'bilateral_ker_weights': tf.get_variable('bilateral_ker_weights',
#                                                      shape=[num_classes, num_classes],
#                                                      initializer=tf.random_uniform_initializer),
#             'compatibility_matrix': tf.get_variable('compatibility_matrix',
#                                                     initializer=np.float32(
#                                                         (np.ones((num_classes, num_classes)) - np.eye(num_classes)))
#                                                     )
#         }
#
#         tf.add_to_collection('crf_weights', weights['spatial_ker_weights'])
#         tf.add_to_collection('crf_weights', weights['compatibility_matrix'])
#         tf.add_to_collection('crf_weights', weights['bilateral_ker_weights'])
#
#         if 'batch_size' in kwargs:
#             batch_size = kwargs.get('batch_size')
#         else:
#             batch_size = rgb_g.get_shape().as_list()[0]
#
#         for batch in range(batch_size):
#
#             unaries = tf.transpose(unary[batch, :, :, :], perm=(2, 0, 1))
#             rgb = tf.transpose(rgb_g[batch, :, :, :], perm=(2, 0, 1))
#
#             image_dims = rgb.get_shape().as_list()[-2:]
#
#             c, h, w = num_classes, image_dims[0], image_dims[1]
#             all_ones = np.ones((c, h, w), dtype=np.float32)
#
#             # Prepare filter normalization coefficients
#             spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
#                                                               theta_gamma=theta_gamma)
#             bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
#                                                                 theta_alpha=theta_alpha,
#                                                                 theta_beta=theta_beta)
#             q_values = unaries
#
#             for i in range(num_iterations):
#                 softmax_out = tf.nn.softmax(q_values, dim=0)
#
#                 # Spatial filtering
#                 spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
#                                                             theta_gamma=theta_gamma)
#                 spatial_out = spatial_out / spatial_norm_vals
#
#                 # Bilateral filtering
#                 bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
#                                                               theta_alpha=theta_alpha,
#                                                               theta_beta=theta_beta)
#                 bilateral_out = bilateral_out / bilateral_norm_vals
#
#                 # Weighting filter outputs
#                 message_passing = (tf.matmul(weights["spatial_ker_weights"],
#                                              tf.reshape(spatial_out, (c, -1))) +
#                                    tf.matmul(weights["bilateral_ker_weights"],
#                                              tf.reshape(bilateral_out, (c, -1))))
#
#                 # Compatibility transform
#                 pairwise = tf.matmul(weights["compatibility_matrix"], message_passing)
#
#                 # Adding unary potentials
#                 pairwise = tf.reshape(pairwise, (c, h, w))
#                 q_values = unaries - pairwise
#
#             if batch == 0:
#                 output = tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))
#             else:
#                 output = tf.concat([output, tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))], 0)
#
#         return output

### HELPER FUNCTIONS ####################################################################################

def _add_summaries(op, weights, biases):

    # Tensorboard variables
    tf.summary.histogram(weights.name, weights)
    if biases:
        tf.summary.histogram(biases.name, biases)
    tf.summary.histogram(op.op.name + '/activations', op)