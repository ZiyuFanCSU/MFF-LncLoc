# MFF-LncLoc: Subcellular Localization Prediction of lncRNAs Based on Multi-Feature Fusion Using Transformers
## To the BiBM reviewers
Due to the system's requirement that the number of uploaded paper pages should be within 8, we have placed the appendix section under directory /paper/Supplementary Materials.pdf on GitHub.

## Introduction
we propose MFF-LncLoc, a prediction model based on Transformers. Utilizing the Transformer's capability to capture contextual and positional information, MFF-LncLoc processes embedding matrices from a non-overlapping trinucleotide method. While traditional models focus on k-mer frequency features, MFF-LncLoc integrates transcript composition, sequence properties, secondary structure features, and nucleotide structural information. Combining all the aforementioned features, the resulting data is processed by a CNN module to extract local sequence features, and a fully connected layer predicts subcellular localization.
## Requirements

To run the codes, You can configure dependencies by restoring our environment:
```
conda env create -f environment.yml -n $Your_env_name$
```

and then：

```
conda activate $Your_env_name$
```

## Structure
The code includes data clean, data processing, model construction, model training, experimental results, case stydies, and various visualisations and interpretive analyses. The directory structure of our uploaded code is as follows:

```
MFF-LncLoc
├── Data_process                # Data clean and data processing
├── feature                     # LncRNA features used in the paper
├── img                         # Visualisations and interpretive analyses
├── models                      # Code of models
├── paper
│   ├── image                   # Model overview
│   └── Supplementary Materials # Appendix of this paper
├── result_logger               # Training procedure and the logger files
├── main.py                     # Training and validation code
├── FocalLoss.py                # Focal Loss
└── environment.yml             # Dependent Package Version
``` 


## Model
Overview of MFF-LncLoc. (a)Non-Overlapping Trinucleotide Embedding (NOLTE).(c) Model Process.
![1.png](paper%2Fmodeloverview.png)

## Training and testing

Run `main.py` using the following command:
```bash
python main.py --device <your_device>
```

Other configurations need to be changed inside `main.py`, including model settings and the data directory.


## Contact

We thank all the researchers who contributed to this work.

If you have any questions, please contact fzychina@csu.edu.cn.
