from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import sklearn
import pickle
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
from config import config, default, generate_config
from metric import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import flops_counter
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
import verification
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))
import fresnet
import fmobilefacenet
import fmobilenet
import fmnasnet
import fdensenet


logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--dataset', default=default.dataset, help='dataset config')
    parser.add_argument('--network', default=default.network, help='network config')
    parser.add_argument('--loss', default=default.loss, help='loss config')
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset, args.loss)
    parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
    parser.add_argument('--pretrained', default=default.pretrained, help='pretrained model to load')
    parser.add_argument('--pretrained-epoch', type=int, default=default.pretrained_epoch, help='pretrained epoch to load')
    parser.add_argument('--ckpt', type=int, default=default.ckpt, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--verbose', type=int, default=default.verbose, help='do verification testing and model saving every verbose batches')
    parser.add_argument('--lr', type=float, default=default.lr, help='start learning rate')
    parser.add_argument('--lr-steps', type=str, default=default.lr_steps, help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=default.wd, help='weight decay')
    parser.add_argument('--mom', type=float, default=default.mom, help='momentum')
    parser.add_argument('--frequent', type=int, default=default.frequent, help='')
    parser.add_argument('--per-batch-size', type=int, default=default.per_batch_size, help='batch size in each context')
    parser.add_argument('--kvstore', type=str, default=default.kvstore, help='kvstore setting')
    args = parser.parse_args()
    return args


def eval_net(args):
    prefix = os.path.join(args.models_root, '%s-%s-%s'%(args.network, args.loss, args.dataset), 'model')
    prefix_dir = os.path.dirname(prefix)
    print('prefix', prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    args.batch_size = args.per_batch_size
    args.rescale_threshold = 0
    args.image_channel = config.image_shape[2]
    config.batch_size = args.batch_size
    config.per_batch_size = args.per_batch_size

    data_dir = config.dataset_path
    image_size = config.image_shape[0:2]
    assert len(image_size)==2
    assert image_size[0]==image_size[1]

    print('loading', args.pretrained, args.pretrained_epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, args.pretrained_epoch)
    arg_num = 0
    for k in arg_params:
        param = arg_params[k]
        print(k, np.prod(param.shape))
        arg_num += np.prod(param.shape)
    print('arg num:', arg_num)
    aux_num = 0
    for k in aux_params:
        param = aux_params[k]
        aux_num += np.prod(param.shape)
    print('aux num:', aux_num)

    if config.count_flops:
        all_layers = sym.get_internals()
        _sym = all_layers['fc1_output']
        FLOPs = flops_counter.count_flops(_sym, data=(1,3,image_size[0],image_size[1]))
        _str = flops_counter.flops_str(FLOPs)
        print('Network FLOPs: %s'%_str)

    model = mx.mod.Module(
        context=mx.gpu(),
        symbol=sym,
    )
    model.bind(data_shapes=[('data', (args.batch_size, 3,112,112))])
    model.set_params(arg_params, aux_params)

    ver_list = []
    ver_name_list = []
    for name in config.test_sets:
        path = os.path.join(data_dir,name+".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            print('ver', name)

    def ver_test(nbatch):
        results = []
        for i in range(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10, None, None)
            print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
            results.append(acc2)
        return results

    ver_test(0)


def main():
    global args
    args = parse_args()
    eval_net(args)


if __name__ == '__main__':
    main()

