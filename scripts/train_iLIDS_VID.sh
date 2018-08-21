#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
################################################################################################
# TODO: supply your project path root here
PATH_ROOT=YOUR_PROJECT_PATH
################################################################################################
i=0
for NUM_SAMPLES  in 21625 #{21625,20708,21095,21038,21612,20850,21700,21722,21607};
do
      let i++
      echo "${i}"
      echo "${NUM_SAMPLES}"
      # where to store the checkpoints
      TRAIN_DIR=${PATH_ROOT}save_model/iLIDS-VID/split$i/mobilenet_b64_dal/
      # where the pretrained model is saved
      PRE_TRAIN=${PATH_ROOT}mobilenet/mobilenet_v1_1.0_224.ckpt
      # dir of training data
      DATASET_DIR=${PATH_ROOT}traindata/iLIDS-VID/split$i/
      # dir stores the pre-extracted feature for initialisation
      # FEATURE_DIR=${PATH_ROOT}DAL/traindata_feature/iLIDS-VID/split$i/

      python train_dal.py \
            --train_dir=${TRAIN_DIR} \
            --dataset_name=iLIDS-VID \
            --dataset_dir=${DATASET_DIR} \
            --model_name=mobilenet_v1 \
            --train_image=224 \
            --max_steps=20000 \
            --batch_size=64 \
            --optimizer=rmsprop \
            --pretrained_model_checkpoint_path=${PRE_TRAIN} \
            --num_gpus=1 \
            --input_queue_memory_factor=4 \
            --num_preprocessing_threads=16 \
            --margin=0.5 \
            --num_classes=300 \
            --num_samples=${NUM_SAMPLES} \
            --num_cams=2 \
            --warm_up_epochs=2 \
            --feature_dim=1024 \
            # --feature_dir=${FEATURE_DIR}
done
################################################################################################
i=0
for NUM_SAMPLES in 20653 #{20653,20834,21751,21364,21421,20847,21609,20759,20737,20852};
do
    let i++
    echo "${i}"
    echo "${NUM_SAMPLES}"
    # where the tfrecords are saved: DATA_DIR
    DATA_DIR=${PATH_ROOT}testdata/iLIDS-VID/split$i/
    # where to stored the extracted feature: OUT_DIR
    # name of the checkpoint: CHECKPOINT_INDEX
    OUT_DIR=${PATH_ROOT}evaluation/feature/iLIDS-VID/split$i/mobilenet_b64_dal/
    # where the checkpoint is saved: CHECKPOINT_PATH
    CHECKPOINT_PATH=${PATH_ROOT}save_model/iLIDS-VID/split$i/mobilenet_b64_dal/
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
        --num_classes=300 \
        --num_samples=${NUM_SAMPLES} 
done
################################################################################################
# evaluation in matlab
# clear; model_name='mobilenet_b64_dal'; CMC_iLIDS_VID_max_10splits
