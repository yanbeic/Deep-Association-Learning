from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import data


datasets_map = \
{
    'data': data,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None,
                num_classes=None, num_samples=None):

    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    if num_classes is None:
        raise ValueError('Number of classes is unknown %s' % name)
    if num_samples is None:
        raise ValueError('Number of samples is unknown %s' % name)

    return datasets_map[name].get_split(
        split_name=split_name,
        dataset_dir=dataset_dir,
        file_pattern=file_pattern,
        reader=reader,
        num_classes=num_classes,
        num_samples=num_samples)
