import argparse

class ParamsParser(argparse.ArgumentParser):
    def __init__(self, model, device):
        super().__init__(description=model)
        self.add_argument('--device', type=str, default=device)
        
        
        self.add_argument('--data', type=str, choices=['CSI300', 'SHCOMP', 'SZCOMP', 'ChiNextComp'], default='CSI300')
        self.add_argument('--freq', type=str, choices=['1d', '1w', '5m', '30m'], default='1d')
        self.add_argument('--seq_len', type=int, default=100)
        self.add_argument('--drop_last', type=bool, default=True)

        if model == 'FNN':
            self.add_argument('--learning_rate', type=float, default=0.0243)
            self.add_argument('--batch_size', type=int, default=16)
            self.add_argument('--train_epochs', type=int, default=82)
            self.add_argument('--hidden_size', type=int, default=31)
            self.add_argument('--hidden_num', type=int, default=3)
            self.add_argument('--dropout', type=float, default=0.336)

        if model == 'LSTM':
            self.add_argument('--learning_rate', type=float, default=0.0284)
            self.add_argument('--batch_size', type=int, default=16)
            self.add_argument('--train_epochs', type=int, default=49)
            self.add_argument('--hidden_size', type=int, default=98)
            self.add_argument('--num_layers', type=int, default=3)
            self.add_argument('--dropout', type=float, default=0.445)

        if model == 'Transformer':
            self.add_argument('--learning_rate', type=float, default=0.0175)
            self.add_argument('--batch_size', type=int, default=16)
            self.add_argument('--train_epochs', type=int, default=22)
            possible_d_models = [d for d in range(8, 513) if d % 8 == 0]
            self.add_argument('--d_model', type=int, choices=possible_d_models, default=344)
            self.add_argument('--n_head', type=int, choices=[2, 4, 8], default=4)
            self.add_argument('--num_encoder_layers', type=int, default=5)
            self.add_argument('--num_decoder_layers', type=int, default=2)
            self.add_argument('--dim_feedforward', type=int, default=381)
            self.add_argument('--dropout', type=float, default=0.0951)

        if model == 'iTransformer':
            self.add_argument('--learning_rate', type=float, default=0.00115)
            self.add_argument('--batch_size', type=int, default=16)
            self.add_argument('--train_epochs', type=int, default=23)
            possible_d_models = [d for d in range(8, 513) if d % 8 == 0]
            self.add_argument('--d_model', type=int, choices=possible_d_models, default=160)
            self.add_argument('--n_head', type=int, choices=[2, 4, 8], default=2)
            self.add_argument('--num_encoder_layers', type=int, default=4)
            self.add_argument('--dim_feedforward', type=int, default=528)
            self.add_argument('--dropout', type=float, default=0.290)


    