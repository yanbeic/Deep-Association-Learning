#!/bin/bash
export CUDA_VISIBLE_DEVICES=
################################################################################################
# TODO: supply your project path root here
PATH_ROOT=YOUR_PROJECT_PATH
################################################################################################
# convert training data to tfrecords
for i in 1 #{1..10};
do
	for DATA_TYPE in train
	do
		# Convert tfrecords of PRID2011
		# TODO: supply your image data path here
		# Where the original image data is stored
		DATA_DIR=YOUR_DATA_PATH/PRID2011/video_data/
		# Where the tfrecords should be stored
		OUT_DIR=${PATH_ROOT}/${DATA_TYPE}data/PRID2011/split$i/
		# The txt file that lists all the data splits
		FILENAME=${PATH_ROOT}/evaluation/datasplits/PRID2011/split$i/${DATA_TYPE}data.txt

		python convert_data_to_tfrecords.py \
		    --data_type=${DATA_TYPE} \
		    --dataset_dir=${DATA_DIR} \
		    --output_dir=${OUT_DIR} \
		    --filename=${FILENAME} \
		    --num_tfrecords=10

		# Convert tfrecords of iLIDS-VID
		# TODO: supply your image data path here
		# Where the original image data is stored
		DATA_DIR=YOUR_DATA_PATH/iLIDS-VID/video_data/
		# Where the tfrecords should be stored
		OUT_DIR=${PATH_ROOT}/${DATA_TYPE}data/iLIDS-VID/split$i/
		# The txt file that lists all the data information
		FILENAME=${PATH_ROOT}/evaluation/datasplits/iLIDS-VID/split$i/${DATA_TYPE}data.txt

		python convert_data_to_tfrecords.py \
		    --data_type=${DATA_TYPE} \
		    --dataset_dir=${DATA_DIR} \
		    --output_dir=${OUT_DIR} \
		    --filename=${FILENAME} \
		    --num_tfrecords=10

	done
done

DATA_TYPE=train
# Convert tfrecords of MARS
# TODO: supply your image data path here
# Where the original image data is stored
DATA_DIR=YOUR_DATA_PATH/MARS/video_data/
# Where the tfrecords should be stored
OUT_DIR=${PATH_ROOT}/${DATA_TYPE}data/MARS/
# The txt file that lists all the data splits
FILENAME=${PATH_ROOT}/evaluation/datasplits/MARS/${DATA_TYPE}data.txt

python convert_data_to_tfrecords.py \
    --data_type=${DATA_TYPE} \
    --dataset_dir=${DATA_DIR} \
    --output_dir=${OUT_DIR} \
    --filename=${FILENAME} \
    --num_tfrecords=100


################################################################################################
# convert test data to tfrecords
for i in 1 #{1..10};
do
	for DATA_TYPE in test
	do
		# Convert tfrecords of PRID2011
		# TODO: supply your image data path here
		# Where the original image data is stored
		DATA_DIR=YOUR_DATA_PATH/PRID2011/video_data/
		# Where the tfrecords should be stored
		OUT_DIR=${PATH_ROOT}/${DATA_TYPE}data/PRID2011/split$i/
		# The txt file that lists all the data splits
		FILENAME=${PATH_ROOT}/evaluation/datasplits/PRID2011/split$i/${DATA_TYPE}data.txt

		python convert_data_to_tfrecords.py \
		    --data_type=${DATA_TYPE} \
		    --dataset_dir=${DATA_DIR} \
		    --output_dir=${OUT_DIR} \
		    --filename=${FILENAME} \
		    --num_tfrecords=1

		# Convert tfrecords of iLIDS-VID
		# TODO: supply your image data path here
		# Where the original image data is stored
		DATA_DIR=YOUR_DATA_PATH/iLIDS-VID/video_data/
		# Where the tfrecords should be stored
		OUT_DIR=${PATH_ROOT}/${DATA_TYPE}data/iLIDS-VID/split$i/
		# The txt file that lists all the data information
		# The txt file that lists all the training data information
		FILENAME=${PATH_ROOT}/evaluation/datasplits/iLIDS-VID/split$i/${DATA_TYPE}data.txt

		python convert_data_to_tfrecords.py \
		    --data_type=${DATA_TYPE} \
		    --dataset_dir=${DATA_DIR} \
		    --output_dir=${OUT_DIR} \
		    --filename=${FILENAME} \
		    --num_tfrecords=1

	done
done

DATA_TYPE=test
# Convert tfrecords of MARS
# TODO: supply your image data path here
# Where the original image data is stored
DATA_DIR=YOUR_DATA_PATH/MARS/video_data/
# Where the tfrecords should be stored
OUT_DIR=${PATH_ROOT}/${DATA_TYPE}data/MARS/
# The txt file that lists all the data splits
FILENAME=${PATH_ROOT}/evaluation/datasplits/MARS/${DATA_TYPE}data.txt

python convert_data_to_tfrecords.py \
    --data_type=${DATA_TYPE} \
    --dataset_dir=${DATA_DIR} \
    --output_dir=${OUT_DIR} \
    --filename=${FILENAME} \
    --num_tfrecords=1
    