#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
################################################################################################
# TODO: supply your project path root here
PATH_ROOT=YOUR_PROJECT_PATH
################################################################################################
# where to store the checkpoints
TRAIN_DIR=${PATH_ROOT}save_model/MARS/mobilenet_b64_dal/
# where the pretrained model is saved
PRE_TRAIN=${PATH_ROOT}mobilenet/mobilenet_v1_1.0_224.ckpt
# dir of training data
DATASET_DIR=${PATH_ROOT}traindata/MARS/
# dir stores the pre-extracted feature for initialisation
# FEATURE_DIR=${PATH_ROOT}DAL/traindata_feature/MARS/

python train_dal.py \
      --train_dir=${TRAIN_DIR} \
      --dataset_name=MARS \
      --dataset_dir=${DATASET_DIR} \
      --model_name=mobilenet_v1 \
      --train_image=224 \
      --max_steps=100000 \
      --batch_size=64 \
      --optimizer=sgd \
      --pretrained_model_checkpoint_path=${PRE_TRAIN} \
      --num_gpus=1 \
      --input_queue_memory_factor=4 \
      --num_preprocessing_threads=16 \
      --margin=0.5 \
      --num_classes=8298 \
      --num_samples=509914 \
      --num_cams=6 \
      --warm_up_epochs=1 \
      --feature_dim=1024 \
      # --feature_dir=${FEATURE_DIR}
################################################################################################
NUM_SAMPLES=681089
echo "${NUM_SAMPLES}"
# where the tfrecords are saved: DATA_DIR
DATA_DIR=${PATH_ROOT}testdata/MARS/
# where to stored the extracted feature: OUT_DIR
OUT_DIR=${PATH_ROOT}evaluation/feature/MARS/mobilenet_b64_dal/
# where the checkpoint is saved: CHECKPOINT_PATH
CHECKPOINT_PATH=${PATH_ROOT}save_model/MARS/mobilenet_b64_dal/
# name of the model: MODLE_NAME
MODLE_NAME=mobilenet_v1
# name of the activation layer to extract feature: FEATURE_NAME
FEATURE_NAME=AvgPool_1a 

python extract_features.py \
  --dataset_name=data \
  --model_name=${MODLE_NAME} \
  --feature_type=${FEATURE_NAME} \
  --batch_size=1 \
  --num_readers=1 \
  --train_image=224 \
  --feature_dim=1024 \
  --checkpoint_dir=${CHECKPOINT_PATH} \
  --dataset_dir=${DATA_DIR} \
  --feature_dir=${OUT_DIR} \
  --num_classes=8298 \
  --num_samples=${NUM_SAMPLES} \
  --num_matfiles=2 
################################################################################################
# evaluation in matlab
# clear; model_name = 'mobilenet_b64_dal'; CMC_mAP_MARS
