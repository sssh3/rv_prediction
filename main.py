from src.experiment.Trainer import Trainer
from src.utils.ParamsParser import ParamsParser
from src.utils.tools import get_device
import torch


model = 'FNN'
# ['FNN', 'LSTM', 'Transformer', 'iTransformer']

if __name__ == '__main__':
    device = get_device()
    parser = ParamsParser(model, device)
    args = parser.parse_args()
    args.model = model
    print(args)

    trainer = Trainer(args)
    trainer.train()
    trainer.evaluate()
    trainer.test()
    
