# Deep-Association-Learning

[Tensorflow](https://www.tensorflow.org/) Implementation of the paper [Chen et al. Deep Association Learning for Unsupervised Video Person Re-identification. BMVC2018](https://arxiv.org/pdf/1808.07301.pdf). You may refer to our [poster](https://github.com/yanbeic/Deep-Association-Learning/blob/master/poster/bmvc18-poster.pdf) for a quick overview.


## Getting Started

### Prerequisites:

- Datasets: [PRID2011](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/) [3], [iLIDS-VIDS](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html) [4], [MARS](http://www.liangzheng.com.cn/Project/project_mars.html) [5].
- Python 2.7. 
- Tensorflow version >= 1.4.0. (For model training)
- Matlab. (For model evaluation)


### Data preparation:

1. Download ImageNet pretrained models: [mobilenet_v1](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) [1], [resnet_v1_50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) [2].


2. Convert image data to tfrecords. 
(Need to supply your paths in the following `.sh` file. Check the `TODO` comments in the `.sh` file.)

```
bash scripts/tf_convert_data.sh
```


## Running Experiments

### Training: 

Train models and extract features. 
(Need to supply your paths in the following `.sh` file. Check the `TODO` comments in the `.sh` file.)

Model implementation include the following `.py` files:
* `train_dal.py`: build and run the training graph.
* `association.py`: build the anchor learning graph and compute the association losses.
* `network.py`: define the network.
* `utils.py`: data preparation.

For example, to train the DAL model using `mobilenet_v1` on MARS, run the the following scripts.

```
bash scripts/train_MARS.sh
```

**Note** that you may modify the type of deep model by changing the flag `--model_name` (eg. `--model_name=resnet_v1_50`). 
You can also modify the number of gpus by changing the flag `--num_gpus`. (eg. `--num_gpus=2`).


### Testing: 

Test model performance in matlab.
Evaluation codes are placed under the directory `evaluation`.

For examples, to test the DAL model performance trained on MARS in matlab, run the following command.

```
clear; model_name = 'mobilenet_b64_dal'; CMC_mAP_MARS
```

## Citation
Please refer to the following if this repository is useful for your research.

### Bibtex:

```
@inproceedings{chen2018bmvc,
  title={Deep Association Learning for Unsupervised Video Person Re-identification},
  author={Chen, Yanbei and Zhu, Xiatian and Gong, Shaogang},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


## References
[1] [Howard et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv 2017.](https://arxiv.org/pdf/1704.04861.pdf) <br />
[2] [He et al. Deep Residual Learning for Image Recognition. CVPR 2016.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) <br />
[3] [Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.](https://files.icg.tugraz.at/seafhttp/files/ba284964-6e03-4261-bb39-e85280707598/hirzer_scia_2011.pdf) <br />
[4] [Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.](http://www.eecs.qmul.ac.uk/~xiatian/papers/ECCV14/WangEtAl_ECCV14.pdf) <br />
[5] [Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.](http://www.liangzheng.com.cn/Project/project_mars.html) <br /> 


## Acknowledgements

This repository is partially built upon the [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/slim) repository. The evaluation code (cmc & mAP) is partially borrowed from the [MARS-evaluation](https://github.com/liangzheng06/MARS-evaluation) repository. 


