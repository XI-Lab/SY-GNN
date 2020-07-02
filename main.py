import argparse
import logging
import os
import sys
from shutil import rmtree

import numpy as np
import torch
import torch.nn as nn
from pack.train_func import *
from torch.utils.data.sampler import RandomSampler

import pack as spk
from pack.datasets import QM_sym
from pack.utils import compute_params, to_json, read_from_json
from pack.datasets.qm_sym import properties, atomref

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def export_model(args):
    jsonpath = os.path.join(args.modelpath, 'args.json')
    train_args = read_from_json(jsonpath)
    model = get_model(train_args, atomref=np.zeros((100, 1)))  # delete atomref when no use
    model.load_state_dict(
        torch.load(os.path.join(args.modelpath, 'best_model')))

    torch.save(model, args.destpath)


def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser()

    ## command-specific
    cmd_parser = argparse.ArgumentParser(add_help=False)
    cmd_parser.add_argument('--cuda', help='Set flag to use GPU(s)', action='store_true')
    cmd_parser.add_argument('--parallel',
                            help='Run data-parallel on all available GPUs (specify with environment variable'
                                 + ' CUDA_VISIBLE_DEVICES)',
                            action='store_true')
    cmd_parser.add_argument('--batch_size', type=int,
                            help='Mini-batch size for training and prediction (default: %(default)s)',
                            default=64)

    ## training
    train_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    train_parser.add_argument('datapath',
                              help='Path / destination of XYZ dataset directory')
    train_parser.add_argument('modelpath',
                              help='Destination for models and logs')

    train_parser.add_argument('--seed', type=int, default=1,
                              help='Set random seed for torch and numpy.')
    train_parser.add_argument('--overwrite',
                              help='Remove previous model directory.',
                              action='store_true')
    train_parser.add_argument('--data_size', type=int,
                              help='Training set size (default: %(default) means for all)',
                              default=0)

    train_parser.add_argument('--split_path',
                              help='Path / destination of npz with data splits',
                              default=None)
    train_parser.add_argument('--split',
                              help='Split into [train] [validation] and use remaining for testing',
                              type=float, nargs=2, default=[0.8, 0.1])
    train_parser.add_argument('--max_epochs', type=int,
                              help='Maximum number of training epochs (default: %(default)s)',
                              default=300)
    train_parser.add_argument('--lr', type=float,
                              help='Initial learning rate (default: %(default)s)',
                              default=1e-4)
    train_parser.add_argument('--lr_patience', type=int,
                              help='Steps to reduce (default: %(default)s)',
                              default=5000)
    train_parser.add_argument('--lr_decay', type=float,
                              help='Learning rate decay (default: %(default)s)',
                              default=0.97)

    train_parser.add_argument('--logger',
                              help='Choose logger for training process (default: %(default)s)',
                              choices=['csv', 'tensorboard'], default='tensorboard')
    train_parser.add_argument('--log_every_n_epochs', type=int,
                              help='Log metrics every given number of epochs (default: %(default)s)',
                              default=1)

    ## evaluation
    eval_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    eval_parser.add_argument('datapath', help='Path of XYZ dataset')
    eval_parser.add_argument('modelpath', help='Path of stored model')
    eval_parser.add_argument('--split',
                             help='Evaluate trained model on given split',
                             choices=['train', 'validation', 'test'],
                             default=['test'], nargs='+')

    # model-specific parsers
    model_parser = argparse.ArgumentParser(add_help=False)

    #######  SY-GNN model  #######
    sygnn_parser = argparse.ArgumentParser(add_help=False, parents=[model_parser])
    sygnn_parser.add_argument('--features', type=int,
                              help='Size of atom-wise representation (default: %(default)s)',
                              default=192)
    sygnn_parser.add_argument('--interactions', type=int,
                              help='Number of interaction blocks (default: %(default)s)',
                              default=6)
    sygnn_parser.add_argument('--cutoff', type=float, default=5.,
                              help='Cutoff radius of local environment (default: %(default)s)')
    sygnn_parser.add_argument('--cutoff_network', type=str, default='none',
                              choices=['none', 'hard'],
                              help='wCutoff network to use (default: %(default)s)')
    sygnn_parser.add_argument('--num_gaussians', type=int, default=25,
                              help='Number of Gaussians to expand distances (default: %(default)s)')

    ## setup subparser structure
    cmd_subparsers = main_parser.add_subparsers(dest='mode', help='Command-specific arguments')
    cmd_subparsers.required = True
    subparser_train = cmd_subparsers.add_parser('train', help='Training help')
    subparser_eval = cmd_subparsers.add_parser('eval', help='Eval help')

    subparser_export = cmd_subparsers.add_parser('export', help='Export help')
    subparser_export.add_argument('modelpath', help='Path of stored model')
    subparser_export.add_argument('destpath', help='Destination path for exported model')

    train_subparsers = subparser_train.add_subparsers(dest='model', help='Model-specific arguments')
    train_subparsers.required = True
    train_subparsers.add_parser('sygnn', help='SYGNN help', parents=[train_parser, sygnn_parser])

    eval_subparsers = subparser_eval.add_subparsers(dest='model', help='Model-specific arguments')
    eval_subparsers.required = True
    eval_subparsers.add_parser('sygnn', help='SYGNN help', parents=[eval_parser, sygnn_parser])

    return main_parser


def get_model(args, atomref=None, mean=None, stddev=None, parallelize=False):
    representation = spk.representation.SYGNN(args.features,
                                              args.features,
                                              args.interactions,
                                              args.cutoff,
                                              args.num_gaussians,
                                              cutoff_network=args.cutoff_network)

    atomwise_output = spk.atomistic.Atomwise(n_in=args.features * (args.interactions + 1),
                                             mean=mean,
                                             stddev=stddev,
                                             atomref=atomref)
    model = spk.atomistic.AtomisticModel(representation, atomwise_output)

    if parallelize:
        model = nn.DataParallel(model)

    logging.info("The model you built has: %d parameters" % compute_params(model))

    return model


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == 'export':
        export_model(args)
        sys.exit(0)

    device = torch.device("cuda" if args.cuda else "cpu")
    argparse_dict = vars(args)
    jsonpath = os.path.join(args.modelpath, 'args.json')

    if args.mode == 'train':
        if args.overwrite and os.path.exists(args.modelpath):
            rmtree(args.modelpath)
            logging.info('existing model will be overwritten...')

        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)

        to_json(jsonpath, argparse_dict)

        spk.utils.set_random_seed(args.seed)
        train_args = args
    else:
        train_args = read_from_json(jsonpath)

    logging.info('XYZ will be loaded...')
    XYZ = QM_sym(args.datapath, load_from_file=False, properties=properties,
                 collect_triples=False, sym_tags=True)

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, 'split.npz')

    logging.info('create splits...')
    data_train, data_val, data_test = XYZ.create_splits(*train_args.split,
                                                        split_file=split_path)

    print('training size: ' + str(args.data_size))
    if args.data_size != 0:
        data_train.subset = data_train.subset[:args.data_size]
    logging.info('load data...')
    train_loader = spk.data.AtomsLoader(data_train, batch_size=args.batch_size,
                                        sampler=RandomSampler(data_train),
                                        num_workers=4, pin_memory=True)
    val_loader = spk.data.AtomsLoader(data_val, batch_size=args.batch_size,
                                      num_workers=2, pin_memory=True)

    if args.mode == 'train':
        logging.info('calculate statistics...')
        split_data = np.load(split_path)
        mean, stddev = train_loader.get_statistics(properties,
                                                   per_atom=False,
                                                   atomrefs=atomref)
        mean = torch.Tensor(mean)
        stddev = torch.Tensor(stddev)
        logging.info('mean: ' + str(mean))
        logging.info('stddev: ' + str(stddev))
        np.savez(split_path, train_idx=split_data['train_idx'],
                 val_idx=split_data['val_idx'],
                 test_idx=split_data['test_idx'], mean=mean, stddev=stddev)
    else:
        mean, stddev = None, None

    # construct the model
    model = get_model(train_args, atomref=atomref, mean=mean, stddev=stddev,
                      parallelize=args.parallel).to(device)

    if args.mode == 'eval':
        if args.parallel:
            model.module.load_state_dict(torch.load(os.path.join(args.modelpath, 'best_model')))
        else:
            model.load_state_dict(torch.load(os.path.join(args.modelpath, 'best_model')))
    if args.mode == 'train':
        logging.info("training...")
        train(args, model, train_loader, val_loader, device, mean=mean.cuda(), stddev=stddev.cuda())
        logging.info("...training done!")
    elif args.mode == 'eval':
        logging.info("evaluating...")
        test_loader = spk.data.AtomsLoader(data_test,
                                           batch_size=args.batch_size,
                                           num_workers=2, pin_memory=True)
        with torch.no_grad():
            evaluate(args, model, properties, train_loader,
                     val_loader, test_loader, device, mean=mean.cuda(), stddev=stddev.cuda())
        logging.info("... done!")
    else:
        print('Unknown mode:', args.mode)
