# Advancing Weakly-Supervised Change Detection in Satellite Images via Adversarial Class Prompting
## :notebook_with_decorative_cover: Code for Paper: Advancing Weakly-Supervised Change Detection in Satellite Images via Adversarial Class Prompting [[arXiv]](https://arxiv.org)

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

``` bibtex

```
