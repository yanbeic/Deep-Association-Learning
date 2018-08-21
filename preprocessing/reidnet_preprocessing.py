from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def preprocess_for_eval(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def preprocess_image(image, height, width, is_training=False):
    if is_training:
        # no data augmentation
        return preprocess_for_eval(image, height, width)
    else:
        return preprocess_for_eval(image, height, width)
