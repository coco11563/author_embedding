
import argparse
import json
import logging
import os

import numpy as np
import torch

sys_path = "/home/xiaomeng/jupyter_base/KGE_EXPERIMENT"  # the model check point and log will save in this path

import sys
sys.path.append(sys_path)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config_170000.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, epoch ,args, best = False):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    argparse_dict = vars(args)
    if best :
        path = os.path.join(args.save_path, "best", args.model)
        with open(os.path.join(path, 'config'), 'w') as fjson:
            json.dump(argparse_dict, fjson)
        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(path, 'checkpoint')
        )

        entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(path, 'entity_embedding'),
            entity_embedding
        )

        relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(path, 'relation_embedding'),
            relation_embedding
        )
    else :
        path = os.path.join(args.save_path, args.model)

        with open(os.path.join(path,'config_{}.json'.format(epoch)), 'w') as fjson:
            json.dump(argparse_dict, fjson)
        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(path, 'checkpoint_{}'.format(epoch))
        )

        entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(path, 'entity_embedding_{}'.format(epoch)),
            entity_embedding
        )

        relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(path, 'relation_embedding_{}'.format(epoch)),
            relation_embedding
        )


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    if not os.path.exists(filename):
        os.system(r"touch {}".format(filename))
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def parse_args(args=None):

    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train_use_transE.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)

    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_valid', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=True)
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data', default=True)

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets', default=False)
    parser.add_argument('--regions', type=int, nargs='+', default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    parser.add_argument('--model', default='RoGCN', type=str)

    parser.add_argument('-de', '--double_entity_embedding', action='store_true', default=True)
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true', default=False)

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=1000, type=int)
    parser.add_argument('-g', '--gamma', default=24.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true',default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=32, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=sys_path + "/checkpoint/", type=str)
    parser.add_argument('--max_steps', default=300000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=500, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=100, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    # from old dog new trick
    parser.add_argument('--ent-regularization', type=float, default=1.55e-10, help='DO NOT MANUALLY SET')
    parser.add_argument('--rel-regularization', type=float, default=3.93e-15, help='DO NOT MANUALLY SET')
    parser.add_argument('--inverse-relation', type=bool, default=False, help='DO NOT MANUALLY SET')
    parser.add_argument('--ent-dropout',type=float, default=0)
    parser.add_argument('--rel-dropout',type=float, default=0)

    parser.add_argument('--rel-ini',type=float, default=None)
    parser.add_argument('--ent-ini',type=float, default=None)

    parser.add_argument('--graph_activation', type=str, default='gelu')
    parser.add_argument('--self_loop', type=bool, default=True)
    parser.add_argument('--graph_model_layer', type = int, default=2)

    return parser.parse_args(args)


