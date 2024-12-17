import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self,x1,x2,y = None):
        self.x1 = torch.from_numpy(x1).type(torch.int64)
        self.x2 = torch.from_numpy(x2)
        if y is not None:
            self.y = torch.from_numpy(y)
        else:
            self.y = None

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, index):
        if self.y is not None:
            return self.x1[index], self.x2[index], self.y[index]
        else:
            return self.x1[index], self.x2[index]

def load_data(data_path="./data/train.csv", encoder_list = None, train=True):
    df = pd.read_csv(data_path)
    # data pre-process
    df = df[df['fuel_type'] != 'Electric']
    # data load
    x1 = np.array(df[['manufacturer','model','gearbox_type','fuel_type']]) # discrete attributes
    x2 = np.array(df[['year','operating_hours','registration_fees','efficiency','engine_capacity']],dtype=np.float64) # continuous attributes
    if train:
        y = np.array(df['price'],dtype=np.float32)
        encoder_list = []
        for i in range(4):
            encoder = LabelEncoder()
            if i==1:
                encoder.fit(x1[:,i].tolist()+["mask"])
            else:
                encoder.fit(x1[:, i])
            encoder_list.append(encoder)
            x1[:,i] = encoder.transform(x1[:,i])
        x1 = x1.astype(np.int64)
        return  MyDataset(x1,x2,y), encoder_list
    else:
        for i in range(4):
            encoder = encoder_list[i]
            if i==1:
                known_classes = encoder.classes_
                x1[:,i] = encoder.transform(np.array([
                    label if label in known_classes else 'mask' for label in x1[:,i]
                ]))
            else:
                x1[:,i] = encoder.transform(x1[:,i])
        x1 = x1.astype(np.int64)
        id = df['id']
        return  MyDataset(x1,x2,None), id

if __name__ == '__main__':
    # test
    dataset, encoder_list = load_data()
    print(len(dataset))
    data_loader = DataLoader(dataset,batch_size=32)
    data_iter = iter(data_loader)
    x1,x2,y = next(data_iter)
    print(x1.shape)
    print(x2.shape)
    print(y.shape)


