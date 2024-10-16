from src.experiment.Trainer import Trainer
from src.utils.ParamsParser import ParamsParser
from src.utils.tools import get_device
import torch

import optuna
import subprocess

model = 'LSTM'
# ['FNN', 'LSTM']

def objective(trial):
    device = get_device()
    parser = ParamsParser(model, device)
    args = parser.parse_args()
    args.model = model

    args.train_epochs = trial.suggest_int('train_epochs', 10, 30)
    args.hidden_size = trial.suggest_int('hidden_size', 10, 100)
    args.dropout = trial.suggest_float('dropout', 0, 0.8)

    trainer = Trainer(args)
    trainer.silence=True
    trainer.train()
    
    if trial.should_prune():
        raise optuna.TrialPruned()

    return trainer.evaluate()



if __name__ == '__main__':
    log_file_name = 'LSTM'
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend('./src/tuning_logs/' + log_file_name + '.log'))
    study = optuna.create_study(
        direction='minimize', 
        storage=storage, 
        study_name='LSTM_1',
        load_if_exists=True)
    study.optimize(objective, n_trials=50, timeout=1200)
    