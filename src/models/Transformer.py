import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        '''
        
        # Add positional encoding to the input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward, dropout, batch_first=True):
        super(Transformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=batch_first
        )
        
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        # src and tgt are of shape (batch_size, seq_length, input_size)
        src = self.embedding(src) * math.sqrt(self.d_model)  # Shape: (batch_size, seq_length, d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)  # Shape: (batch_size, seq_length, d_model)
        
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        output = self.transformer(src, tgt)
        output = self.output_layer(output)
        
        return output
