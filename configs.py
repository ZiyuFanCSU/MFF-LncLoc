import os
import torch

def makedir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

# root
SEED = 888
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
TOKEN_LEN = 3  # word size
PART_NUM = 64
PART_LEN = 128
Random_state = 10
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
log_title = str(PART_LEN)+str(PART_NUM)+str(TOKEN_LEN)+str(Random_state)

# model configs
CONFIG = {
    "batch_size": 16,
    "loss_func":"cv", # "dice_loss" "w_cv" "Focal_loss" "cv"
    "w_sampler":True,
    "lr": 0.001,
    "bestforwhat": "f1", # "auc" "auc" "f1"
    "epoch": 200,
    "min_epoch": 5,
    "patience": 0.00002,  # early stopping
    "patience_num": 10,
    "repeat":1,
    "num_fold":5,
}

# data
DATA_DIR = os.path.join(ROOT_DIR, "Data_process") #/data_process
DATASET_FILE = os.path.join(DATA_DIR, "data.txt")
DATA_PATH = os.path.join(DATA_DIR, "data.csv")
REBUILD_DATA_PATH0 = os.path.join(DATA_DIR, "data_process/final_data_"+str(PART_LEN)+"_"+str(PART_NUM)+"_"+str(TOKEN_LEN)+"/final_benchmark_data/rebuild_data0.json")#"benchmark_data0.json")
REBUILD_DATA_PATH1 = os.path.join(DATA_DIR, "data_process/final_data_"+str(PART_LEN)+"_"+str(PART_NUM)+"_"+str(TOKEN_LEN)+"/final_benchmark_data/rebuild_data1.json")#"benchmark_data1.json")
REBUILD_DATA_PATH2 = os.path.join(DATA_DIR, "data_process/final_data_"+str(PART_LEN)+"_"+str(PART_NUM)+"_"+str(TOKEN_LEN)+"/final_benchmark_data/rebuild_data2.json")#"benchmark_data2.json") final_benchmark_data

# model
OUTPUTS_DIR = os.path.join(ROOT_DIR, "middle")
VOCAB_PATH0 = os.path.join(DATA_DIR, "data_process/final_data_"+str(PART_LEN)+"_"+str(PART_NUM)+"_"+str(TOKEN_LEN)+"/vocabulary.json")
NORM_PATH = os.path.join(OUTPUTS_DIR, "norm.pkl")
LE_PATH = os.path.join(OUTPUTS_DIR, "le.pkl")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.pkl")


# remkdir
makedir(OUTPUTS_DIR)
 

