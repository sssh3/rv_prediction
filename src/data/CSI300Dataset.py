from torch.utils.data import Dataset, DataLoader
from src.data.get_rv import get_rv
import polars as pl
import os
import torch
import numpy as np
from src.utils.tools import MinMaxScaler

class CSI300Dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.scaler = MinMaxScaler()
        self.__read_data__()
    
    def __read_data__(self):
        processed_data_path = 'data/processed/CSI300_' + self.args.freq + '.csv'
        if not os.path.exists(processed_data_path):
            get_rv()
        
        df = pl.read_csv(processed_data_path)

        df = df.with_columns(df.sql('''
                SELECT 
                SUBSTR(timestamp, 1, 4) AS year,
                SUBSTR(timestamp, 6, 2) AS month,
                SUBSTR(timestamp, 9, 2) AS day
                FROM self;
                                    '''))
        df = df.drop('timestamp')

        x_arr = df.drop('rv').to_numpy().astype(np.float32)
        y_arr = df.select('rv').to_numpy().astype(np.float32)

        data_arr = np.concatenate([x_arr, y_arr], axis=1)
        data_arr = self.scaler.fit_transform(data_arr)

        self.data_y = data_arr[:, -1:]
        self.data_x = data_arr[:, :-1]



    def __getitem__(self, index):
        seq_x = self.data_x[index : index + self.args.seq_len]
        seq_y = self.data_y[index : index + self.args.seq_len]

        if self.args.model == 'FNN':
            input = np.concatenate([seq_x, seq_y], axis=1).flatten()
            target = self.data_y[index + self.args.seq_len + 1].flatten()
            return input, target
        
        if self.args.model == 'LSTM':
            input = np.concatenate([seq_x, seq_y], axis=1)
            target = self.data_y[index + self.args.seq_len + 1].flatten()
            return input, target
    
    def __len__(self):
        return len(self.data_x) - self.args.seq_len
    
    def inverse_transform(self, data):
        return self.scaler.inv_transform(data)
