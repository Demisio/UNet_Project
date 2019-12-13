# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
from tfwrapper import layers
import numpy as np


def unet2D_bn_dropout(images, training, nlabels, num_filters_first_layer=64, padding_type='VALID', test_dropout_rate=1.0):
    dropout_rate = tf.where(training, 0.5, test_dropout_rate)

    nf = num_filters_first_layer

    if str.lower(padding_type) == 'same':
        # images:0' shape=(8, 240, 240, 1)
        # images:0' shape=(8, 240, 240, 2, 1)
        images_padded = images
    else:
        # images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')
        images_padded = tf.pad(images, [[0, 0], [92, 92], [92, 92], [0, 0]], 'CONSTANT')

    if len(images_padded.shape) == 5:
        # both image modalities, combine first
        tmpI = images_padded[:, :, :, :, 0]
        conv1_1 = layers.conv2D_layer_bn(tmpI, 'conv1_1', num_filters=nf, training=training, padding=padding_type)
    else:
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=nf, training=training,
                                         padding=padding_type)

    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=nf, training=training, padding=padding_type)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=nf * 2, training=training, padding=padding_type)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=nf * 2, training=training, padding=padding_type)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=nf * (2 ** 2), training=training,
                                     padding=padding_type)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=nf * (2 ** 2), training=training,
                                     padding=padding_type)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=nf * (2 ** 3), training=training,
                                     padding=padding_type)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=nf * (2 ** 3), training=training,
                                     padding=padding_type)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=nf * (2 ** 4), training=training,
                                     padding=padding_type)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=nf * (2 ** 4), training=training,
                                     padding=padding_type)
    # conv5_2/Relu:0 shape=(8, 15, 15, 512)

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512,
                                       training=training)
    upconv4 = tf.nn.dropout(upconv4, dropout_rate)

    concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=nf * (2 ** 3), training=training,
                                     padding=padding_type)

    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=nf * (2 ** 3), training=training,
                                     padding=padding_type)
    # conv6_2/Relu:0 shape=(8, 30, 30, 256)

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256,
                                       training=training)
    upconv3 = tf.nn.dropout(upconv3, dropout_rate)

    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=nf * (2 ** 2), training=training,
                                     padding=padding_type)

    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=nf * (2 ** 2), training=training,
                                     padding=padding_type)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128,
                                       training=training)
    upconv2 = tf.nn.dropout(upconv2, dropout_rate)

    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=nf * (2), training=training,
                                     padding=padding_type)

    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=nf * (2), training=training, padding=padding_type)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64,
                                       training=training)
    upconv1 = tf.nn.dropout(upconv1, dropout_rate)


    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=nf, training=training, padding=padding_type)

    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=nf, training=training, padding=padding_type)


    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1, 1), activation=tf.identity,
                                      training=training, padding='VALID')

    return pred
