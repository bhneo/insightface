from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys

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
import fmobilefacenet_decoder
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


def get_symbol(encoder, train_idx):
    de_linear_1, deconv_6sep, deres_5, dedconv_45, deres_4, dedconv_34, deres_3, dedconv_23, deconv_2_dw, deconv_1, img = eval(config.net_name + '_decoder').get_symbol(encoder)
    # presentation = mx.symbol.Variable('presentation')

    syms = [de_linear_1, deconv_6sep, deres_5, dedconv_45, deres_4, dedconv_34, deres_3, dedconv_23, deconv_2_dw, deconv_1, img]
    train_layer = syms[train_idx]

    loss = mx.symbol.square(train_layer - encoder)
    loss = mx.symbol.MakeLoss(loss)
    out_list = [loss]
    out_list.append(mx.symbol.BlockGrad(img))
    out = mx.symbol.Group(out_list)

    # fixed_params = []
    # for sym in syms:
    #     for arg in sym.list_arguments():
    #         if arg.endswith('_weight') or arg.endswith('_bias'):
    #             fixed_params.append(arg)
    return out


def train_net(args):
    ctx = []
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    else:
        cvd = []
    if len(cvd)>0:
      for i in range(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
    prefix = os.path.join(args.models_root, '%s-%s'%(args.network+'_decoder', args.dataset), 'model')
    prefix_dir = os.path.dirname(prefix)
    print('prefix', prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    args.ctx_num = len(ctx)
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = config.image_shape[2]
    config.batch_size = args.batch_size
    config.per_batch_size = args.per_batch_size

    data_dir = config.dataset_path
    path_imgrec = None
    path_imglist = None
    image_size = config.image_shape[0:2]
    assert len(image_size)==2
    assert image_size[0]==image_size[1]
    print('image_size', image_size)
    print('num_classes', config.num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")

    print('Called with argument:', args, config)
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None

    begin_epoch = 0

    print('loading', args.pretrained, args.pretrained_epoch)
    sym_encoder, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, args.pretrained_epoch)
    sym_encoder = sym_encoder.get_internals()['conv_6dw7_7_batchnorm_output']
    #[de_linear_1, deconv_6sep, deres_5, dedconv_45, deres_4, dedconv_34, deres_3, dedconv_23, deconv_2_dw, deconv_1, img]
    idx = 0
    sym_encoder_decoder = get_symbol(sym_encoder, idx)


    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym_encoder_decoder,
        fixed_param_names=sym_encoder.list_arguments()
    )
    val_dataiter = None

    from image_iter import FaceImageIter
    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        rand_mirror          = False,
        mean                 = mean,
        cutoff               = config.data_cutoff,
        color_jittering      = config.data_color,
        images_filter        = config.data_images_filter,
    )
    metric1 = LossValueMetric()
    eval_metrics = [mx.metric.create(metric1)]

    if config.net_name=='fresnet' or config.net_name=='fmobilefacenet':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    #initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)
    _cb = mx.callback.Speedometer(args.batch_size, args.frequent)

    ver_list = []
    ver_name_list = []
    for name in config.val_targets:
      path = os.path.join(data_dir,name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)


    highest_acc = [0.0, 0.0]  #lfw and target
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for step in lr_steps:
        if mbatch==step:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

      if mbatch>=0 and mbatch%args.verbose==0:
          # data = data_list[i]
          # embeddings = None
          # ba = 0
          # while ba < data.shape[0]:
          #     _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
          #     # print(_data.shape, _label.shape)
          #     time0 = datetime.datetime.now()
          #     if data_extra is None:
          #         db = mx.io.DataBatch(data=(_data,), label=(_label,))
          #     else:
          #         db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label,))
          #     model.forward(db, is_train=False)
          #     net_out = model.get_outputs()
          #     # _arg, _aux = model.get_params()
          #     # __arg = {}
          #     # for k,v in _arg.iteritems():
          #     #  __arg[k] = v.as_in_context(_ctx)
          #     # _arg = __arg
          #     # _arg["data"] = _data.as_in_context(_ctx)
          #     # _arg["softmax_label"] = _label.as_in_context(_ctx)
          #     # for k,v in _arg.iteritems():
          #     #  print(k,v.context)
          #     # exe = sym.bind(_ctx, _arg ,args_grad=None, grad_req="null", aux_states=_aux)
          #     # exe.forward(is_train=False)
          #     # net_out = exe.outputs
          #     _embeddings = net_out[0].asnumpy()

        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        is_highest = False

        if is_highest:
          do_save = True
        if args.ckpt==0:
          do_save = False
        elif args.ckpt==2:
          do_save = True
        elif args.ckpt==3:
          msave = 1

        if do_save:
            print('saving', msave)
            arg, aux = model.get_params()
            mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if config.max_steps>0 and mbatch>config.max_steps:
        sys.exit(0)

    epoch_cb = None
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = 999999,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = args.kvstore,
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )


class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
      pred = preds[-1].asnumpy()
      loss = pred[0]
      self.sum_metric += loss
      self.num_inst += 1.0


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()

