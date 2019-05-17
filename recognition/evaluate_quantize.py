from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import datetime
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
    sym_512, arg_params_512, aux_params_512 = mx.model.load_checkpoint('./models/model-y1/model-y1-512d',
                                                                       args.pretrained_epoch)
    model_512 = mx.mod.Module(
        context=mx.gpu(),
        symbol=sym_512,
    )
    model_512.bind(data_shapes=[('data', (args.batch_size, 3, 112, 112))])
    model_512.set_params(arg_params_512, aux_params_512)

    sym_map, arg_params_map, aux_params_map = mx.model.load_checkpoint('./models/model-y1/model-y1-map',
                                                                       args.pretrained_epoch)
    model_map = mx.mod.Module(
        context=mx.gpu(),
        symbol=sym_map,
    )
    model_map.bind(data_shapes=[('data', (args.batch_size, 512))])
    model_map.set_params(arg_params_map, aux_params_map)

    ver_list = []
    ver_name_list = []
    for name in config.test_sets:
        path = os.path.join(data_dir,name+".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            print('ver', name)

    for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = test(ver_list[i], model_512, model_map, args.batch_size, 10)
        print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))


def test(data_set, mx_model1, mx_model2, batch_size, nfolds=10, quantize_bits=8):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    _label = nd.ones( (batch_size,) )
    for i in xrange( len(data_list) ):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba<data.shape[0]:
            bb = min(ba+batch_size, data.shape[0])
            count = bb-ba
            _data = nd.slice_axis(data, axis=0, begin=bb-batch_size, end=bb)
            db = mx.io.DataBatch(data=(_data,), label=(_label,))
            mx_model1.forward(db, is_train=False)
            net_out = mx_model1.get_outputs()
            _embeddings = net_out[0].asnumpy()
            quantized_embeddings, maximums, minimums = quantize_embeddings(_embeddings, bits=quantize_bits)
            _embeddings = dequantize_embeddings(quantized_embeddings, maximums, minimums, quantize_bits)
            if embeddings is None:
                embeddings = np.zeros( (data.shape[0], _embeddings.shape[1]) )
            embeddings[ba:bb,:] = _embeddings[(batch_size-count):,:]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in xrange(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    acc1 = 0.0
    std1 = 0.0

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    _, _, accuracy, val, val_std, far = verification.evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def quantize(embedding, maximum=None, minimum=None, bits=8):
    if maximum==None:
        maximum = np.max(embedding, -1)
    if minimum==None:
        minimum = np.min(embedding, -1)
    quantum = (maximum-minimum)/(np.power(2, bits))
    quantized_embedding = []
    for element in embedding:
        quantized_embedding.append(int((element-minimum) // quantum))
    return quantized_embedding, maximum, minimum, quantum


def dequantize(quantized_embedding, maximum, minimum, bits):
    quantum = (maximum-minimum)/(np.power(2, bits))
    embedding = []
    for element in quantized_embedding:
        embedding.append(quantum*element+minimum)
    return embedding


def quantize_embeddings(embeddings, maximum=None, minimum=None, bits=8):
    quantized_embeddings = []
    maximums = []
    minimums = []
    for i in range(len(embeddings)):
        quantized_embedding, maximum, minimum, quantum = quantize(embeddings[i], maximum, minimum, bits)
        quantized_embeddings.append(quantized_embedding)
        maximums.append(maximum)
        minimums.append(minimum)
    return quantized_embeddings, maximums, minimums


def dequantize_embeddings(quantized_embeddings, maximums, minimums, bits):
    assert len(quantized_embeddings) == len(maximums) and len(maximums) == len(minimums)
    embeddings=[]
    for i in range(len(maximums)):
        embeddings.append(dequantize(quantized_embeddings[i], maximums[i], minimums[i], bits))
    return embeddings


def main():
    global args
    args = parse_args()
    eval_net(args)


if __name__ == '__main__':
    main()

