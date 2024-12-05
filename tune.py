from src.experiment.Trainer import Trainer
from src.utils.ParamsParser import ParamsParser
from src.utils.tools import get_device
import torch

import optuna
import subprocess

model = 'iTransformer'
# ['FNN', 'LSTM', 'Transformer', 'iTransformer']
n_trails = 100
study_name_postfix = '1'

def objective(trial):
    device = get_device()
    parser = ParamsParser(model, device)
    args = parser.parse_args()
    args.model = model

    if model == 'FNN':
        args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        args.train_epochs = trial.suggest_int('train_epochs', 10, 100)
        args.hidden_size = trial.suggest_int('hidden_size', 10, 100)
        args.dropout = trial.suggest_float('dropout', 0, 0.5)
        args.hidden_num = trial.suggest_int('hidden_num', 1, 5)

    if model == 'LSTM':
        args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        args.train_epochs = trial.suggest_int('train_epochs', 10, 50)
        args.hidden_size = trial.suggest_int('hidden_size', 10, 100)
        args.dropout = trial.suggest_float('dropout', 0, 0.5)
        args.hidden_num = trial.suggest_int('num_layers', 1, 5)

    if model == 'Transformer':
        args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        args.train_epochs = trial.suggest_int('train_epochs', 10, 30)
        args.dropout = trial.suggest_float('dropout', 0, 0.5)
        
        # d_model must be divisible by n_heads
        args.n_head = trial.suggest_categorical('n_head', [2, 4, 8])
        possible_d_models = [d for d in range(8, 513) if d % 8 == 0]
        args.d_model = trial.suggest_categorical('d_model', possible_d_models)

        args.num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 5)
        args.num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 3)
        args.dim_feedforward = trial.suggest_int('dim_feedforward', 64, 1024)

    if model == 'iTransformer':
        args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        args.train_epochs = trial.suggest_int('train_epochs', 10, 30)
        args.dropout = trial.suggest_float('dropout', 0, 0.5)
        
        # d_model must be divisible by n_heads
        args.n_head = trial.suggest_categorical('n_head', [2, 4, 8])
        possible_d_models = [d for d in range(8, 513) if d % 8 == 0]
        args.d_model = trial.suggest_categorical('d_model', possible_d_models)

        args.num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 5)
        args.dim_feedforward = trial.suggest_int('dim_feedforward', 64, 1024)
        


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
        study_name=f'{model}_{study_name_postfix}',
        load_if_exists=True)
    study.optimize(objective, n_trials=n_trails, timeout=36000)
    print(study.best_params)
    