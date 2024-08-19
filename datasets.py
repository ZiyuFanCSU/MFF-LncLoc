import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from configs import PART_NUM
from utils import load_json
from configs import REBUILD_DATA_PATH0


class Datasets(Dataset):
    def __init__(self, data, repeat=1):
        self.data = data * repeat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return (
            torch.LongTensor([self.data[idx][0][f"#{seq_idx}"] for seq_idx in range(PART_NUM)]),
            torch.LongTensor([self.data[idx][1][f"#{seq_idx}"] for seq_idx in range(PART_NUM)]),
            torch.LongTensor([self.data[idx][2][f"#{seq_idx}"] for seq_idx in range(PART_NUM)]),
            torch.FloatTensor(self.data[idx][0]["features"]),
            torch.LongTensor([self.data[idx][0]["label"]]),
        )


if __name__ == "__main__":
    data = load_json(REBUILD_DATA_PATH0)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    datasets = Datasets(train_data)
    print("Amount of training data:", len(datasets))

    # output data format
    for idx, dataset in enumerate(datasets):
        if idx >= 1:
            break
        print(dataset[0].shape, dataset[1].shape, dataset[2].shape)
        print(dataset)
