import torch
from torch import nn
import math

class ITransformer(nn.Module):
    def __init__(self, seq_len, input_dim, output_dim, d_model, nhead, num_encoder_layers, 
                 dim_feedforward, dropout, batch_first=True):
        super(ITransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(seq_len, d_model)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=batch_first
            ),
            num_layers=num_encoder_layers
        )
        
        self.output_layer = nn.Linear(input_dim * d_model, output_dim)

    def forward(self, src):
        # src is of shape (batch_size, seq_length, input_size)
        # Transpose into (batch_size, input_size, seq_length)
        src = torch.transpose(src, 1, 2)

        src = self.embedding(src) * math.sqrt(self.d_model)  # Shape: (batch_size, input_size, d_model)

        output = self.encoder(src)

        # (batch_size, input_size, d_model) -> (batch_size, input_size*d_model)
        output = output.view(-1, self.input_dim * self.d_model)
        
        output = self.output_layer(output)
        
        return output