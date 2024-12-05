import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_num=1, output_size=1, dropout=0):
        super(FNN, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_num)]
        )
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.fc_out(x)
        return x