# scMinerva: an Unsupervised Graph Learning Framework with Label-efficient Fine-tuning for Single-cell Multi-omics Integrated Analysis

Repository for manuscript [scMinerva](https://www.biorxiv.org/content/10.1101/2022.05.28.493838v1.abstract)

Authors: [Tingyang Yu](https://yistyu.github.io/), Yongshuo Zong, Yixuan Wang, Xuesong Wang, [Yu Li](https://liyu95.com/)

scMinerva is an unsupervised framework for single-cell multi-omics integrated analysis. The learned embeddings from the multi-omics data enable accurate integrated classification of cell types and stages. The power of scMinerva is sparkled by easy fine-tuning and is not sensitive to the using label size for fine-tuning a separate classifier. Out method could achieve excellent performance even with only 5% labeled data. 

![image](https://github.com/YistYU/scMinerva/blob/main/scMinerva_github.jpg)

To use scMinerva, do the following: 
- Install the environment
- Prepare the data
- Train and evaluate scMinerva


## Install the Environment
We provide a yml file containing the necessary packages for scMinerva. Once you have [conda](https://docs.anaconda.com/anaconda/install/) installed, you can create an environment as follows:
```bash
conda env create --file scMinerva.yml 
```

## Prepare the data

Required Files: 

*Omics*: .csv file with shape (a,b) where a is the number of sample and b is the number of feature.\
*label*: labels should be indexed start from *0* and be consecutive natural integers. \
*name the omics files*: Files for omics features are supposed to be named as "i.csv" where *i* is an integer to distinguish omics. i.e. "2.csv".\
*name the label files*: Label file is supposed to be named as "labels.csv" under the corresponding dataset directory.\


## Train and evaluate scMinerva

Demo command on datase GSE156478_CITE:

```bash
python main.py --data_folder 'GSE156478_CITE' --num_omics 2 --num_class 7 --labeled_ratio 0.05
```

Parameters:

1. data_folder: the folder that contains prepared omics data and label data. 
2. num_omics: the amount of omics inputed. 
3. num_class: The number of class to classify.
4. labeled_ratio: Float number from 0 to 1, it means the proportion of labeled data for fine-tuning. 


## How to Cite
```
@article{yu2022scminerva,
  title={scMinerva: an Unsupervised Graph Learning Framework with Label-efficient Fine-tuning for Single-cell Multi-omics Integrated Analysis},
  author={Yu, Tingyang and Zong, Yongshuo and Wang, Yixuan and Wang, Xuesong and Li, Yu},
  journal={bioRxiv},
  pages={2022--05},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contact Us
Please open an issue or contact tyyistyu@gmail.com with any questions.# scMinerva
