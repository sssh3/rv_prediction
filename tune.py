from src.experiment.Trainer import Trainer
from src.utils.ParamsParser import ParamsParser
from src.utils.tools import get_device
import torch

import optuna
import subprocess

model = 'Transformer'
# ['FNN', 'LSTM', 'Transformer']

def objective(trial):
    device = get_device()
    parser = ParamsParser(model, device)
    args = parser.parse_args()
    args.model = model

    if model == 'FNN':
        args.train_epochs = trial.suggest_int('train_epochs', 10, 100)
        args.hidden_size = trial.suggest_int('hidden_size', 10, 100)
        args.dropout = trial.suggest_float('dropout', 0, 0.8)
        args.hidden_num = trial.suggest_int('hidden_num', 1, 5)

    if model == 'LSTM':
        args.train_epochs = trial.suggest_int('train_epochs', 10, 50)
        args.hidden_size = trial.suggest_int('hidden_size', 10, 100)
        args.dropout = trial.suggest_float('dropout', 0, 0.8)
        args.hidden_num = trial.suggest_int('num_layers', 1, 5)

    if model == 'Transformer':
        args.train_epochs = trial.suggest_int('train_epochs', 10, 50)
        args.dropout = trial.suggest_float('dropout', 0, 0.8)
        args.d_model = trial.suggest_int('d_model', 4, 128)
        # args.n_head = trial.suggest_int('n_head', 2, 8)
        args.num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 3)
        args.num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 3)
        args.dim_feedforward = trial.suggest_int('dim_feedforward', 32, 128)


    trainer = Trainer(args)
    trainer.silence=True
    trainer.train()
    
    if trial.should_prune():
        raise optuna.TrialPruned()

    return trainer.evaluate()



if __name__ == '__main__':
    log_file_name = model
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend('./src/tuning_logs/' + log_file_name + '.log'))
    study = optuna.create_study(
        direction='minimize', 
        storage=storage, 
        study_name=f'{model}_1',
        load_if_exists=True)
    study.optimize(objective, n_trials=50, timeout=1200)
    