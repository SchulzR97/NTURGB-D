import data
from model import ActionPredictionModel
from torch.utils.data import DataLoader
from args import get_args_demo_train
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import shutil

AP_MODEL = 'model.pt'
RUN_DATA_FILE = 'run_data.json'
START_TIME = datetime.now()

def accuracy(Y, T):
    Y = torch.argmax(Y, dim=1)
    T = torch.argmax(T, dim=1)

    acc = ((Y == T).sum() / T.numel()).item()

    return acc

def time(Y = None, T = None):
    return (datetime.now() - START_TIME).total_seconds() / (60 * 60)

def plot_run_data(run_id, run_data):
    if not os.path.isdir(f'runs/{run_id}/plot'):
        os.mkdir(f'runs/{run_id}/plot')
    for key in run_data:
        key_str = key.replace('_', ' ')
        if key == time.__name__:
            plt.plot(run_data[key]['val'], color='orange')
        else:
            plt.plot(run_data[key]['train'], color='cornflowerblue', alpha=0.5)
            plt.plot(run_data[key]['val'], color='orange', alpha=0.5)
            plt.plot(run_data[key]['train_avg'], color='cornflowerblue', label='train')
            plt.plot(run_data[key]['val_avg'], color='orange', label='val')
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-')
        plt.grid(which='minor', linestyle='dotted')
        _, _, ymin, ymax = plt.axis()
        if ymin < 0:
            plt.ylim(0, ymax)
        xmin, xmax = 0, len(run_data[key]['train'])-1
        if xmax == xmin:
            xmax += 1
        plt.xlim(xmin, xmax)
        plt.xlabel('epoch')
        plt.ylabel(key_str)
        plt.title(key_str)
        if not key == time.__name__:
            plt.legend()
        plt.savefig(f'runs/{run_id}/plot/{key}.png')
        plt.close()

def initialize(args, compute_metrics):
    run_data = {}
    for m in compute_metrics:
        run_data[m.__name__] = {
            'train': [],
            'val': [],
            'train_avg': [],
            'val_avg': []
        }

    if not os.path.isdir('runs'):
        os.mkdir('runs')

    if args.load_run is not None and os.path.isdir(f'runs/{args.load_run}'):
        run_id = args.load_run

        if os.path.isfile(f'runs/{run_id}/{AP_MODEL}'):
            with open(f'runs/{run_id}/{AP_MODEL}', 'rb') as f:
                model.load_state_dict(torch.load(f))
        if os.path.isfile(f'runs/{run_id}/{RUN_DATA_FILE}'):
            with open(f'runs/{run_id}/{RUN_DATA_FILE}', 'r') as f:
                run_data = json.load(f)
    else:
        run_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        os.mkdir(f'runs/{run_id}')
        shutil.copyfile('model.py', f'runs/{run_id}/model.py')
    
    with open(f'runs/{run_id}/args.json', 'w') as f:
        json.dump(args.__dict__, f)

    return run_id, run_data

def compute(model:ActionPredictionModel, criterion, optimizer, dataloader, args, compute_metrics, train:bool):
    if train:
        model.train()
    else:
        model.eval()
    
    optimizer.zero_grad()
    progress = tqdm(range(args.samples_per_epoch), desc='train' if train else 'val', leave=False)

    run_data = {}
    for m in compute_metrics:
        run_data[m.__name__] = []

    iterator = iter(dataloader)
    for i in progress:
        X, T = next(iterator)
        X, T = X.to(args.device), T.to(args.device)
        if train:
            Y = model(X)
            loss = criterion(Y, T)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                Y = model(X)
                loss = criterion(Y, T)
        
        for m in compute_metrics:
            val = m(Y, T)
            if isinstance(val, torch.Tensor):
                val = val.item()
            run_data[m.__name__].append(val)

    for key in run_data:
        run_data[key] = np.average(run_data[key])

    return run_data

if __name__ == '__main__':
    args = get_args_demo_train()

    if args.split == 'full':
        ds_train, ds_val = data.get_datasets_depth_masked_full(args.in_dir, split_prop=0.7, shuffle=True, sequence_length=args.sequence_length)
    elif args.split == 'CS':
        ds_train, ds_val = data.get_datasets_depth_masked_CS(args.in_dir, sequence_length=args.sequence_length)
    elif args.split == 'CV':
        ds_train, ds_val = data.get_datasets_depth_masked_CV(args.in_dir, sequence_length=args.sequence_length)
    else:
        raise Exception(f'Split {args.split} is not supported. Please privide one of the following values: [full, CV, CS]. See the documentation for more information.')
    
    dl_train = DataLoader(ds_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_val = DataLoader(ds_val, args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ActionPredictionModel(args).to(args.device)
    criterion = F.binary_cross_entropy
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    compute_metrics = [
        accuracy,
        criterion,
        time
    ]

    run_id, run_data = initialize(args, compute_metrics)

    epoch = len(run_data[criterion.__name__]['train'])
    while epoch < args.num_epochs:
        run_data_train = compute(model, criterion, optim, dl_train, args, compute_metrics, train=True)
        run_data_val = compute(model, criterion, optim, dl_val, args, compute_metrics, train=False)

        for key in run_data:
            run_data[key]['train'].append(run_data_train[key])
            run_data[key]['val'].append(run_data_val[key])
            mavg_epochs = epoch if epoch < args.mavg_epochs else args.mavg_epochs
            run_data[key]['train_avg'].append(np.average(run_data[key]['train'][-mavg_epochs:]))
            run_data[key]['val_avg'].append(np.average(run_data[key]['val'][-mavg_epochs:]))

        plot_run_data(run_id, run_data)
        with open(f'runs/{run_id}/{AP_MODEL}', 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(f'runs/{run_id}/{RUN_DATA_FILE}', 'w') as f:
            json.dump(run_data, f)

        acc_train = run_data[accuracy.__name__]['train_avg'][-1]
        acc_val = run_data[accuracy.__name__]['val_avg'][-1]
        print(f'epoch {epoch} train {acc_train*100:0.4f}% val {acc_val*100:0.4f}% time {datetime.now()-START_TIME}')
        epoch += 1