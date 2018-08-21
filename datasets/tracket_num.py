from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



def get_tracket_num(dataset_name):
        if dataset_name == 'MARS':
            return [1816, 1957, 1321, 750, 1880, 574]
        elif dataset_name == 'PRID2011':
            return [89, 89]
        elif dataset_name == 'iLIDS-VID':
            return [150, 150]
        else:
            raise ValueError('You must supply the dataset name as '
                             '-- MARS, PRID2011, iLIDS-VID')


