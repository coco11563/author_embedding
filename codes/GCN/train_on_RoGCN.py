from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dgl

from utils import train_utils

sys_path = "/home/xiaomeng/jupyter_base/KGE_EXPERIMENT"  # the model check point and log will save in this path

import sys

sys.path.append(sys_path)

from utils.sample.EdgeDataLoader import EdgeDataLoader
import json
import logging
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils.sample as samplers
import torch.nn.functional as Fn
import torch.functional as F

from dataloader import BidirectionalOneShotIterator
from dataloader import TrainDataset
from utils.dglds import load_data,build_sub_graph,one_shot_iterator
from GCN import GCN
from utils.train_utils import *

log_name = 'log/NSFC_DISTMULT_nofreeze_NSFC-F01{}.log'.format(time.time())  # log name


# logging = get_logger(sys_path + log_name, name = log_name)

def main(args):
    # cuda.occumpy_mem(0)
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    # elif args.data_path is None:
    #     raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    data = load_data('NSFC')
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels
    args.nentity = num_nodes
    args.nrelation = num_rels

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % num_nodes)
    logging.info('#relation: %d' % num_rels)
    train_triples = []
    for i, j, k in train_data:
        train_triples.append((i, j, k))
    test_triples = []
    for i, j, k in test_data:
        test_triples.append((i, j, k))
    valid_triples = []
    for i, j, k in valid_data:
        valid_triples.append((i, j, k))
    # All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    act_fun = None
    if args.graph_activation == 'relu':
        act_fun = Fn.relu
    elif args.graph_activation == 'gelu':
        act_fun = Fn.gelu
    elif args.graph_activation == 'sigmoid':
        act_fun = Fn.sigmoid
    elif args.graph_activation == 'elu':
        act_fun = Fn.elu
    elif args.graph_activation == 'glu':
        act_fun = Fn.glu
    elif args.graph_activation == 'none':
        act_fun = None
    else:
        raise ValueError('the graph activation function should be [relu gelu sigmoid elu glu or none] but instead of {}'
                         .format(args.graph_activation))

    whole_graph = data._g

    kge_model = RoGCN(
        whole_graph=whole_graph,
        nentity=num_nodes,
        nrelation=num_rels,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        graph_layer_num=args.graph_model_layer,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        ent_drop=args.ent_dropout,
        rel_drop=args.rel_dropout,
        ent_ini=args.ent_ini,
        self_loop=args.self_loop,
        graph_activation=act_fun
    )

    train_graph = build_sub_graph(whole_graph, whole_graph.edata['train_mask'], False)
    test_graph = build_sub_graph(whole_graph, whole_graph.edata['test_mask'], False)
    valid_graph = build_sub_graph(whole_graph, whole_graph.edata['val_mask'], False)

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
        kge_model.whole_graph = kge_model.whole_graph.to('cuda:0')
    if args.do_train:
        # Set training dataloader iterator

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.graph_model_layer)
        e_id = train_graph.edges(form='eid')
        etype_dict = train_graph.edata['etype']
        type_dict = data.type_dict
        type_set = data.type_constrain_dict
        train_data_set = EdgeDataLoader(
                train_graph, e_id, sampler,
                negative_sampler=samplers.Uniform(args.negative_sample_size, type_dict, type_set, etype_dict),
                # negative_sampler=dgl.dataloading.negative_sampler.Uniform(args.negative_sample_size),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4
        )
        train_data_iterator = one_shot_iterator(train_data_set)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)
        best_mrr = -1.0
        training_logs = []
        v_step = args.valid_steps
        best_time = 0
        # Training Loop
        for step in range(init_step, args.max_steps):

            log = kge_model.train_step(kge_model, optimizer, train_data_iterator, args)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }

                save_model(kge_model, optimizer, save_variable_list, step, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % v_step == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                mrr = metrics['MRR']
                if mrr > best_mrr:
                    if mrr > 0.18:
                        v_step = 300
                    if mrr > 0.19:
                        v_step = 200
                    best_time += 1
                    best_mrr = mrr
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, step, args, best=True)
                    logging.info('new best mrr explore in step {}, value is {}'.format(step, mrr))
                log_metrics('Valid', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, step, args)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    args = train_utils.parse_args(None)
    main(args)