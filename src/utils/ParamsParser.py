import argparse

class ParamsParser(argparse.ArgumentParser):
    def __init__(self, model, device):
        super().__init__(description=model)
        self.add_argument('--device', type=str, default=device)
        
        self.add_argument('--learning_rate', type=int, default=0.01)
        self.add_argument('--batch_size', type=int, default=16)
        self.add_argument('--data', type=str, default='CSI300')
        self.add_argument('--freq', type=str, choices=['1d', '1w', '5m', '30m'], default='1d')
        self.add_argument('--seq_len', type=int, default=20)
        self.add_argument('--drop_last', type=bool, default=True)

        if model == 'FNN':
            self.add_argument('--train_epochs', type=int, default=45)
            self.add_argument('--hidden_size', type=int, default=44)
            self.add_argument('--hidden_num', type=int, default=1)
            self.add_argument('--dropout', type=float, default=0.354369)

        if model == 'LSTM':
            self.add_argument('--train_epochs', type=int, default=40)
            self.add_argument('--hidden_size', type=int, default=40)
            self.add_argument('--num_layers', type=int, default=2)
            self.add_argument('--dropout', type=float, default=0.501112)

        if model == 'Transformer':
            self.add_argument('--train_epochs', type=int, default=20)
            self.add_argument('--d_model', type=int, default=64)
            self.add_argument('--n_head', type=int, choices=[2, 4, 8], default=4)
            self.add_argument('--num_encoder_layers', type=int, default=2)
            self.add_argument('--num_decoder_layers', type=int, default=2)
            self.add_argument('--dim_feedforward', type=int, default=128)
            self.add_argument('--dropout', type=float, default=0.3)


    