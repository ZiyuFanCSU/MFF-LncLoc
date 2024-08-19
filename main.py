import torch
import os
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.model_selection import train_test_split,KFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from configs import CONFIG
from configs import *
from sklearn.model_selection import StratifiedKFold
from configs import REBUILD_DATA_PATH0, REBUILD_DATA_PATH1, REBUILD_DATA_PATH2, MODEL_PATH, DEVICE, CONFIG, LE_PATH, PART_NUM,VOCAB_PATH0,Random_state
from datasets import Datasets
from models.model import Model
from train import train
from utils import load_json, load_pkl
from evaluation import Evaluation
import numpy as np
import pandas as pd
from FocalLoss import *
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def train_run():
    data0 = load_json(REBUILD_DATA_PATH0)
    data1 = load_json(REBUILD_DATA_PATH1)
    data2 = load_json(REBUILD_DATA_PATH2)
    data = [r for r in zip(data0, data1, data2)]

    data_l = []
    for ll in data:
        data_l.append(ll[0]['label'])
   
    traindata = []
    test_data= []
    labelss1 = []
    labelss2 = []
    havetest = True
    if havetest:
        restIdList, testIdList = train_test_split(range(len(data)), test_size=0.1,stratify =data_l,random_state=42)
    else:
        restIdList = range(len(data))
    for m in restIdList:
        traindata.append(np.array(data)[m])
        labelss1.append(np.array(data)[m][0]['label'])

    data_labels = []
    for l in traindata:
        data_labels.append(l[0]['label'])
    if havetest:   
        for mm in testIdList:
            test_data.append(np.array(data)[mm])
            labelss2.append(np.array(data)[mm][0]['label'])

    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_data = []
    valid_data = []

    for k, (train_index, test_index) in enumerate(splits.split(traindata,data_labels)):
        train_data.append(np.array(traindata)[train_index].tolist())
        valid_data.append(np.array(traindata)[test_index].tolist())


    train_data = [[tuple(j) for j in train_data[i]] for i in range(5)]
    valid_data = [[tuple(j) for j in valid_data[i]] for i in range(5)]

    k_fold_evaluation_valid = []
    k_fold_evaluation_test = []

    
    for i in range(CONFIG['num_fold']):

        train_datasets = Datasets(train_data[i], repeat=CONFIG['repeat'])
        labels = []
        for l in train_data[i]:
            labels.append(l[0]['label'])
        val_datasets = Datasets(valid_data[i])
        if havetest:   
            test_datasets = Datasets(test_data)

        if CONFIG["w_sampler"]:
            weights = []
            for ll in labels:
                if ll == 0:
                    weights.append(0.2)
                elif ll == 1:
                    weights.append(0.3)
                elif ll == 2:
                    weights.append(0.2)
                else:
                    weights.append(0.3)
            torch.manual_seed(SEED) 
            train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            train_loader = DataLoader(
                train_datasets,
                batch_size=CONFIG["batch_size"],
                num_workers=1,
                sampler=train_sampler,
                drop_last=False,
                worker_init_fn=worker_init_fn
            )
        else:
            train_loader = DataLoader(
                train_datasets,
                batch_size=CONFIG["batch_size"],
                shuffle=True,
                num_workers=1,
                drop_last=False,
                worker_init_fn=worker_init_fn
            )

        val_loader = DataLoader(
            val_datasets,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=1,
            drop_last=False,
            worker_init_fn=worker_init_fn
        )
        if havetest:   
            test_loader = DataLoader(
                test_datasets,
                batch_size=CONFIG["batch_size"],
                shuffle=False,
                num_workers=1,
                drop_last=False,
                worker_init_fn=worker_init_fn
            )
        else:
            test_loader=val_loader
        
        model = Model(
            vocab_size=len(load_json(VOCAB_PATH0)),
            emb_dim=PART_NUM,
            part_num=PART_NUM,
            features_num=15,
            p_drop=0.3,
            h=1,
            hidden_size=64,
            outputs_size=4,
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)
        criterion = FocalLoss(gamma=2)
        
        print(f'第{i+1}次训练')
        valid_one_fold_result, test_one_fold_result  = train(train_loader, val_loader,test_loader, model, optimizer,scheduler, criterion, CONFIG["bestforwhat"])
        k_fold_evaluation_valid.append(valid_one_fold_result)
        k_fold_evaluation_test.append(test_one_fold_result)

    print(k_fold_evaluation_valid)
    performance_result = np.mean(k_fold_evaluation_valid, axis=0)
    print('valid五折交叉验证的最终结果为:')
    print(f'acc:{round(performance_result[0],4)}  macro-pre:{round(performance_result[1],4)}  macro-recall:{round(performance_result[2],4)}  macro-F1-score:{round(performance_result[3],4)}  auc:{round(performance_result[4],4)}  aupr:{round(performance_result[5],4)}  auc2:{round(performance_result[6],4)}  aupr2:{round(performance_result[7],4)}')
    print(k_fold_evaluation_test)
    performance_result = np.mean(k_fold_evaluation_test, axis=0)
    print('test五折交叉验证的最终结果为:')
    print(f'acc:{round(performance_result[0],4)}  macro-pre:{round(performance_result[1],4)}  macro-recall:{round(performance_result[2],4)}  macro-F1-score:{round(performance_result[3],4)}  auc:{round(performance_result[4],4)}  aupr:{round(performance_result[5],4)}  auc2:{round(performance_result[6],4)}  aupr2:{round(performance_result[7],4)}')

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id) 

if __name__ == "__main__":
    setup_seed(SEED)
    train_run()

