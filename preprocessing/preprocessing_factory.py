from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from preprocessing import reidnet_preprocessing


def get_preprocessing(name, is_training=False):

    preprocessing_fn_map = \
        {
            'reidnet': reidnet_preprocessing,
        }

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    def preprocessing_fn(image, output_height, output_width):
        return preprocessing_fn_map[name].preprocess_image(
            image, output_height, output_width, is_training=is_training)

    return preprocessing_fn
