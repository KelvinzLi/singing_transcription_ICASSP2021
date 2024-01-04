import torch
import torch.nn as nn
import argparse
from predictor import Predictor
import yaml, toml
import os

def extract(config, *args):
    assert isinstance(config, dict)
    
    out = config
    for arg in args:
        if arg in out:
            out = out[arg]
        else:
            out = None
            break
    return out

def main(args):
    device= 'cpu'
    if torch.cuda.is_available():
        device = args.device
    print ("use", device)

    config_path = os.path.join(args.model_dir, 'config.toml')

    if not os.path.exists(config_path):
        raise FileExistsError('config.toml doesn\'t exist')

    config = toml.load(config_path)

    print(yaml.dump(config, allow_unicode=True, default_flow_style=False))
    
    predictor_kwargs_pairs = [
        ('model_path', ('model', 'path')),
        ('model_import_path', ('model', 'model_import_path')),
        ('model_kwargs', ('model', 'model_kwargs')),
    ]
    
    fit_kwargs_pairs = [
        ('batch_size', ('training', 'batch_size')),
        ('valid_batch_size', ('training', 'valid_batch_size')),
        ('epoch', ('training', 'epoch')),
        ('lr', ('training', 'lr')),
        ('weight_decay', ('training', 'weight_decay')),
        ('onset_pos_weight', ('training', 'onset_pos_weight')),
        ('save_prefix', ('training', 'save_prefix')),
        
        ('dataset_import_path', ('dataset', 'dataset_import_path')),
        ('dataset_kwargs', ('dataset', 'dataset_kwargs')),
        ('train_dataset_path', ('dataset', 'train_dataset_path')),
        ('valid_dataset_path', ('dataset', 'valid_dataset_path')),
        ('label_smoother_import_path', ('dataset', 'label_smoother_import_path')),
        ('label_smoother_kwargs', ('dataset', 'label_smoother_kwargs')),
    ]
    
    predictor_kwargs = {}
    for kwarg_name, keys in predictor_kwargs_pairs:
        kwarg = extract(config, *keys)
        if kwarg is not None:
            predictor_kwargs[kwarg_name] = kwarg
            
    fit_kwargs = {}
    for kwarg_name, keys in fit_kwargs_pairs:
        kwarg = extract(config, *keys)
        if kwarg is not None:
            fit_kwargs[kwarg_name] = kwarg
            
    print(fit_kwargs)
    
    predictor = Predictor(device=device, 
                          **predictor_kwargs,
                         )
    # predictor.fit(
    #     train_dataset_path=args.training_dataset,
    #     valid_dataset_path=args.validation_dataset,
    #     model_dir=args.model_dir,
    #     batch_size=50,
    #     valid_batch_size=200,
    #     epoch=10,
    #     lr=1e-4,
    #     save_every_epoch=1,
    #     save_prefix=args.save_prefix
    # )
    
    predictor.fit(
        model_dir=args.model_dir,
        save_every_epoch=1,
        
        **fit_kwargs,
    )


if __name__ == '__main__':
    """
    This script performs training and validation of the singing transcription model.
    training_dataset: The pkl file used as training data.
    validation_dataset: The pkl file used as validation data.
    model_dir: The directory that stores models.
    save_prefix: The prefix of the models.
    device: The device (e.g. cuda:0) to use if cuda is available.
    model-path: Pre-trained model (optional). If provided, the weights of the pre-trained model will be loaded, and used as the initial weights.

    Sample usage:
    python train.py train.pkl test.pkl models/ effnet cuda:0
    The script will use train.pkl as training set, test.pkl as validation set (it will predict the whole validation set every epoch, and print the validation loss).
    Each epoch, a model file called called "effnet_{epoch}" will be saved at "models/".
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('device')
    
    args = parser.parse_args()

    main(args)
