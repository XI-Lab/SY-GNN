import os

import numpy as np
import torch
import torch.nn as nn

import pack as spk
from pack.utils import read_from_json
from pack.datasets.qm_sym import properties
from pack.optimizer import Adam_multitask

__all__ = ['train', 'evaluate', 'evaluate_dataset', 'export_model']


def train(args, model, train_loader, val_loader, device, mean=None, stddev=None):
    # setup hook and logging

    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam_multitask(trainable_params, lr=args.lr)
    hooks = [
        spk.train.MaxEpochHook(args.max_epochs),
        spk.train.ExponentialDecayHook(optimizer, gamma=args.lr_decay, step_size=args.lr_patience)
    ]
    metrics = []
    i = 0
    for p in properties:
        metrics.extend([spk.metrics.MeanAbsoluteError(p, p, mean=mean[i], stddev=stddev[i]),
                        spk.metrics.RootMeanSquaredError(p, p, mean=mean[i], stddev=stddev[i])])
        i += 1
    if args.logger == 'csv':
        logger = spk.train.CSVHook(os.path.join(args.modelpath, 'log'),
                                   metrics,
                                   every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)
    elif args.logger == 'tensorboard':
        logger = spk.train.TensorboardHook(os.path.join(args.modelpath, 'log'),
                                           metrics,
                                           every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)

    def loss(batch, result):
        func = nn.MSELoss(reduction='none')
        pred = torch.cat([result[i].unsqueeze(1) for i in properties], 1)

        truth = torch.cat([batch[i] for i in properties], 1)
        individual_loss = torch.mean(func(pred, truth), dim=0)
        return individual_loss, torch.mean(individual_loss).detach()

    trainer = spk.train.Trainer(args.modelpath, model, loss, optimizer,
                                train_loader, val_loader, hooks=hooks, mean=mean, stddev=stddev)
    trainer.train(device)


def evaluate(args, model, property, train_loader, val_loader, test_loader, device, mean=None, stddev=None):
    header = ['Subset']
    metrics = []
    i = 0
    for p in properties:
        header.extend([p + ' MAE', p + ' RMSE'])
        metrics.extend([spk.metrics.MeanAbsoluteError(p, p, mean=mean[i], stddev=stddev[i]),
                        spk.metrics.RootMeanSquaredError(p, p, mean=mean[i], stddev=stddev[i])])
        i += 1
    results = []
    if 'train' in args.split:
        results.append(['training'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, train_loader, device)])

    if 'validation' in args.split:
        results.append(['validation'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, val_loader, device)])

    if 'test' in args.split:
        detail, result = evaluate_dataset_detail(metrics, model, test_loader, device, property)
        results.append(['test'] + ['%.5f' % i for i in result])

        np.savetxt(os.path.join(args.modelpath, f'eval_{property}_detail.csv'), detail, fmt='%s', delimiter=',')
        # remove [ ]
        f = open(os.path.join(args.modelpath, f'eval_{property}_detail.csv'), "r")
        lines = [l for l in f.readlines()]
        f.close()
        for i in range(len(lines)):
            lines[i] = lines[i].replace('[', '').replace(']', '')
        f = open(os.path.join(args.modelpath, f'eval_{property}_detail.csv'), "w")
        f.writelines(lines)
        f.close()

    header = ','.join(header)
    results = np.array(results)

    np.savetxt(os.path.join(args.modelpath, 'evaluation.csv'), results, header=header, fmt='%s', delimiter=',')


def evaluate_dataset_detail(metrics, model, loader, device, property):
    for metric in metrics:
        metric.reset()

    detail = []

    for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        result = model(batch)
        for i in range(result['y'].shape[0]):
            detail.append([
                float(batch[property][i]),
                float(result['y'][i]),
                abs(float(batch[property][i]) - float(result['y'][i])),
                int(batch['_atomic_numbers'][i].cpu().clamp(0, 1).sum())
            ])

        for metric in metrics:
            metric.add_batch(batch, result)

    results = [metric.aggregate() for metric in metrics]
    return detail, results


def evaluate_dataset(metrics, model, loader, device):
    for metric in metrics:
        metric.reset()

    for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)

        for metric in metrics:
            metric.add_batch(batch, result)

    results = [metric.aggregate() for metric in metrics]
    return results


def export_model(args):
    jsonpath = os.path.join(args.modelpath, 'args.json')
    train_args = read_from_json(jsonpath)
    model = get_model(train_args, atomref=np.zeros((100, 1)))
    model.load_state_dict(
        torch.load(os.path.join(args.modelpath, 'best_model')))

    torch.save(model, args.destpath)
