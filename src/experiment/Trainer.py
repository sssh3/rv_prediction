import torch
import torch.nn as nn
import torch.optim as optim
from src.data.CSI300Dataset import CSI300Dataset
from src.models.FNN import FNN
from src.models.LSTM import LSTM
from torch.utils.data import DataLoader, Subset
import time
import torch.nn.functional as F

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self._build_model()
        self.silence = False


    def _read_data(self):
        dataset = CSI300Dataset(self.args)

        total_len = len(dataset)
        train_size = int(0.7 * total_len)
        val_size = int(0.15 * total_len)
        test_size = total_len - train_size - val_size
        
        self.train_set = Subset(dataset, range(0, train_size))
        self.val_set = Subset(dataset, range(train_size, train_size + val_size))
        self.test_set = Subset(dataset, range(train_size + val_size, total_len))

        self.train_loader = DataLoader(self.train_set, drop_last=self.args.drop_last, batch_size=self.args.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, drop_last=self.args.drop_last, batch_size=self.args.batch_size)
        self.test_loader = DataLoader(self.test_set, drop_last=self.args.drop_last, batch_size=self.args.batch_size)


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def _build_model(self):
        self._read_data()

        if self.args.model == 'FNN':
            input_len = len(self.train_set[0][0])
            self.model = FNN(input_len, self.args.hidden_size, self.args.hidden_num, 1, self.args.dropout).to(self.device)

        if self.args.model == 'LSTM':
            input_len = len(self.train_set[0][0][0])
            self.model = LSTM(input_len, self.args.hidden_size, self.args.num_layers, 1, self.args.dropout).to(self.device)

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()

    def train(self):
        start_time = time.time()
        for epoch in range(self.args.train_epochs):
            self.model.train()
            total_loss = 0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.train_loader)

            if epoch % 10 == 0 or epoch == self.args.train_epochs-1:
                if not self.silence:
                    print(f'Epoch [{epoch+1}/{self.args.train_epochs}], Loss: {avg_loss:.8f}')
                    print(f'Used time: {time.time()-start_time}')

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            total_mse = 0.0
            total_samples = 0

            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                mse = F.mse_loss(outputs, targets, reduction='sum').item()
                total_mse += mse
                total_samples += targets.size(0)

            average_mse = total_mse / total_samples
            if not self.silence:
                print(f'Mean Squared Error: {average_mse:.8f}')

        return average_mse