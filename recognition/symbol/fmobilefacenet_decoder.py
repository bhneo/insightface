
import sys
import os
import mxnet as mx
import symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config


def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)
    act = Act(data=bn, act_type=config.net_act, name='%s%s_relu' %(name, suffix))
    return act

def DeConv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), target_shape=None, num_group=1, name=None, suffix='', activate=True):
    conv = mx.sym.Deconvolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, target_shape=target_shape, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)
    if activate:
        act = Act(data=bn, act_type=config.net_act, name='%s%s_relu' %(name, suffix))
    else:
        act = bn
    return act
    
def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)    
    return bn

def DeLinear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Deconvolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)
    act = Act(data=bn, act_type=config.net_act, name='%s%s_relu' % (name, suffix))
    return act

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    return conv    

    
def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return proj

def DeDResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), target_shape=None, num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = DeConv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, target_shape=target_shape, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return proj

def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
    	shortcut=identity
    	conv=DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
    	identity=conv+shortcut
    return identity
        

# def get_symbol():
#     num_classes = config.emb_size
#     print('in_network', config)
#     fc_type = config.net_output
#     data = mx.symbol.Variable(name="data")
#     data = data-127.5
#     data = data*0.0078125
#     blocks = config.net_blocks
#     conv_1 = Conv(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1")
#     if blocks[0]==1:
#       conv_2_dw = Conv(conv_1, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw")
#     else:
#       conv_2_dw = Residual(conv_1, num_block=blocks[0], num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=64, name="res_2")
#     conv_23 = DResidual(conv_2_dw, num_out=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=128, name="dconv_23")
#     conv_3 = Residual(conv_23, num_block=blocks[1], num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128, name="res_3")
#     conv_34 = DResidual(conv_3, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, name="dconv_34")
#     conv_4 = Residual(conv_34, num_block=blocks[2], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, name="res_4")
#     conv_45 = DResidual(conv_4, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, name="dconv_45")
#     conv_5 = Residual(conv_45, num_block=blocks[3], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, name="res_5")
#     conv_6_sep = Conv(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep")
#
#     fc1 = symbol_utils.get_fc1(conv_6_sep, num_classes, fc_type)
#     return fc1

SHAPE = True
def infer_shape(sym):
    if SHAPE:
        _, sym_shape, _ = sym.infer_shape(data=(1, 3, 112, 112))
    else:
        sym_shape = None
    return sym_shape

def get_symbol(inputs):
    print('decoder')
    blocks = config.net_blocks
    all_shape,inputs_shape,_ = inputs.infer_shape(data=(1, 3, 112, 112))
    de_linear_1 = DeLinear(data=inputs, num_filter=512, kernel=(7,7),num_group=512,stride=(1,1),target_shape=(7,7),name='de_linear_1')
    de_linear_1_shape = infer_shape(de_linear_1)

    deconv_6sep = Conv(de_linear_1, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="deconv_6sep")
    deconv_6sep_shape = infer_shape(deconv_6sep)
    deres_5 = Residual(deconv_6sep, num_block=blocks[3], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      num_group=256, name="deres_5")
    deres_5_shape = infer_shape(deres_5)
    dedconv_45 = DeDResidual(deres_5, num_out=128, kernel=(3, 3), stride=(2, 2), target_shape=(14, 14), num_group=512, name="dedconv_45")
    dedconv_45_shape = infer_shape(dedconv_45)
    deres_4 = Residual(dedconv_45, num_block=blocks[2], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(0, 0),
                      num_group=256, name="deres_4")
    deres_4_shape = infer_shape(deres_4)
    dedconv_34 = DeDResidual(deres_4, num_out=64, kernel=(3, 3), stride=(2, 2), target_shape=(28, 28), num_group=256, name="dedconv_34")
    dedconv_34_shape = infer_shape(dedconv_34)
    deres_3 = Residual(dedconv_34, num_block=blocks[1], num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128,
                      name="deres_3")
    deres_3_shape = infer_shape(deres_3)
    dedconv_23 = DeDResidual(deres_3, num_out=64, kernel=(3, 3), stride=(2, 2), target_shape=(56, 56), num_group=128, name="dedconv_23")
    dedconv_23_shape = infer_shape(dedconv_23)

    if blocks[0]==1:
        deconv_2_dw = Conv(dedconv_23, num_group=64, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="deconv_2_dw")
        deconv_2_dw_shape = infer_shape(deconv_2_dw)
    else:
        deconv_2_dw = Residual(dedconv_23, num_block=blocks[0], num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=64, name="deres_2")

    deconv_1 = mx.sym.Deconvolution(data=deconv_2_dw, num_filter=3, kernel=(3,3), num_group=1, stride=(2,2),
                                target_shape=(112,112), no_bias=True, name='%s%s_conv2d' % ('deconv_1', ''))
    deconv_1 = mx.sym.sigmoid(data=deconv_1, name='deconv_1_sigmoid')
    deconv_1_shape = infer_shape(deconv_1)

    img = deconv_1*255
    return de_linear_1, deconv_6sep, deres_5, dedconv_45, deres_4, dedconv_34, deres_3, dedconv_23, deconv_2_dw, deconv_1, img
