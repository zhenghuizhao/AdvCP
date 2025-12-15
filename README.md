# Advancing Weakly-Supervised Change Detection in Satellite Images via Adversarial Class Prompting
## :notebook_with_decorative_cover: Code for Paper: Advancing Weakly-Supervised Change Detection in Satellite Images via Adversarial Class Prompting [[arXiv]](https://arxiv.org/abs/2508.17186)

<p align="center">
    <img src="./tutorials/introduction1_01.png" width="95%" height="95%">
</p>

## Abastract <p align="justify">
#### a) AdvCP address co-occuring background noise in image-level weakly-supervised change detection.
#### b) It rectifies adversarial samples via an online prototype with exponential moving average.
#### c) As a plug-and-play module, AdvCP boosts baselines without extra inference cost.


## :speech_balloon: DISep Overview:
<p align="center">
    <img src="./tutorials/method1_01.png" width="95%" height="95%">
</p>


##
## A. Preparations
### 1. Dataset Structure 

``` bash
WSCD dataset with image-level labels:
├─A
├─B
├─label
├─imagelevel_labels.npy
└─list
```

### 2.Create and activate conda environment

```bash
conda create --name xxx python=3.6
conda activate xxx
pip install -r requirments.txt
```

##
## B. Train and Test
The DISep module is located in the [`./AdvCP_module`](./AdvCP_module) directory. It can be seamlessly integrated into any training process without modifying the inference process, as shown in the code example we provide.

```bash
# train 
python train.py

```

###
```bash
# test
python test.py
```


##
## C. Performance
<p align="center">
    <img src="./tutorials/experiment0_01.png" width="95%" height="95%">
</p>


## Citation
If it's helpful to your research, please kindly cite. Here is an example BibTeX entry:

```
@ARTICLE{11217335,
  author={Zhao, Zhenghui and Wu, Chen and Wang, Di and Chen, Hongruixuan and Chen, Cuiqun and Zheng, Zhuo and Du, Bo and Zhang, Liangpei},
  journal={IEEE Transactions on Image Processing}, 
  title={Advancing Weakly-Supervised Change Detection in Satellite Images via Adversarial Class Prompting}, 
  year={2025},
  volume={34},
  pages={7065-7078},
  keywords={Prototypes;Training;Noise;Predictive models;Annotations;Remote sensing;Location awareness;Weak supervision;Robustness;Perturbation methods;Remote sensing;change detection;satellite imagery;high resolution;weak supervision},
  doi={10.1109/TIP.2025.3623260}
}

@misc{zhao2025advancingweaklysupervisedchangedetection,
      title={Advancing Weakly-Supervised Change Detection in Satellite Images via Adversarial Class Prompting}, 
      author={Zhenghui Zhao and Chen Wu and Di Wang and Hongruixuan Chen and Cuiqun Chen and Zhuo Zheng and Bo Du and Liangpei Zhang},
      year={2025},
      eprint={2508.17186},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.17186}, 
}
```
