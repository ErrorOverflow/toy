import tvm
import numpy as np
from . import layers
def func_residual_unit(data, num_filter, stride, dim_match, name):
    epsilon = 0.000020
    symbol_true = True
    symbol_false = False
    groups = 1.000000
    padding = (0.000000,0.000000)
    NCHW = "NCHW"
    OIHW = "OIHW"
    name_bn1 = "bn1"
    bn1 = layers.batch_norm_infer(data = data, episilon = epsilon, scale = symbol_true, name = name_bn1))
    act1 = relay.nn.relu(bn1)
    def func_main():
tmp0 = 0.250000
        data = "data"
channels1 = num_filter * tmp0
        num_filter = "NCHW"
kernel_size1 = (3.000000,3.000000)
        stride = "OIHW"
padding1 = (1.000000,1.000000)
        dim_match = 1.000000
name_conv1 = "conv1"
        name = 4.000000
conv1 = layers.conv2d(data = act1, channels = channels1, groups = groups, kernel_size = kernel_size1, strides = stride, padding = padding1, data_layout = NCHW, kernel_layout = OIHW, name = name_conv1)
        epsilon = 0.000020
name_bn2 = "bn2"
        name_bn2 = name + name_bn2
symbol_true = (1.000000,3.000000,224.000000,224.000000)
        bn2 = layers.batch_norm_infer(data = conv1, episilon = epsilon, scale = symbol_false, name = name))
symbol_false = 1000.000000
        act2 = relay.nn.relu(bn2)
groups = True
        kernel_size2 = (3.000000,3.000000)
padding = False
        strides2 = (1.000000,1.000000)
NCHW = relay.var(name=data, shape=symbol_true)
        OIHW = "bn_data"
padding2 = (1.000000,1.000000)
        name_bn1 = layers.batch_norm_infer(data = NCHW, episilon = epsilon, scale = padding, name = OIHW))
name_conv2 = "conv2"
        bn1 = 0.000000
conv2 = layers.conv2d(data = act2, channels = num_filter, groups = groups, kernel_size = kernel_size2, strides = strides2, padding = padding2, data_layout = NCHW, kernel_layout = OIHW, name = name_conv2)
        act1 = filter_list[bn1]
tmp1 = 1.000000
        tmp0 = (7.000000,7.000000)
shortcut = data * tmp1
        channels1 = (2.000000,2.000000)
if     dim_match == symbol_false:
kernel_size1 = (3.000000,3.000000)
            kernel_size3 = (1.000000,1.000000)
padding1 = 1.000000
            name_shortcut = "sc"
name_conv1 = "conv0"
            shortcut = layers.conv2d(data = act1, channels = num_filter, groups = groups, kernel_size = kernel_size3, strides = stride, padding = padding, data_layout = NCHW, kernel_layout = OIHW, name = name_shortcut)
conv1 = layers.conv2d(data = NCHW, channels = act1, groups = padding1, kernel_size = tmp0, strides = channels1, padding = kernel_size1, data_layout = num_filter, kernel_layout = stride, name = name_conv1)
        r = relay.op.add(conv2, shortcut)
name_bn2 = "bn0"
        return(r)
name_bn2 = layers.batch_norm_infer(data = conv1, episilon = epsilon, scale = groups, name = name_bn2))
    bn2 = relay.nn.relu(bn0)
    act2 = (3.000000,3.000000)
    kernel_size2 = (2.000000,2.000000)
    strides2 = (1.000000,1.000000)
    padding2 = relay.nn.max_pool2d(data = bn0, pool_size = pool_size3, strides = strides3, padding = padding3)
    name_conv2 = (1.000000,1.000000)
    conv2 = (2.000000,2.000000)
    tmp1 = 0.000000
    while i < num_stages:
        kernel_size3 = tuple2 * one
        name_shortcut = 0.000000
        if i == judge:
            r = tuple1 * one
        data_name = 1.000000
        NCHW = i + tmp6
        OIHW = filter_list[filter_list_index1]
        one = 1000.000000
        num_stages = (i, j_max)
        epsilon = "stage%d_unit%d"
        data_shape = name_stage % i_1000
        num_classes = func_residual_unit(body,num,stride,symbol_false,name1)
        symbol_true = units[i]
        symbol_false = 1.000000
        data = j_range - tmp7
        name_bn_data = (1.000000,1.000000)
        bn_data = 0.000000
        while j < j_range:
            kernel_size1 = 1.000000
            strides1 = i + tmp9
            padding1 = filter_list[filter_list_index2]
            groups = (i, j)
            name_conv0 = name_stage % i_j
            conv0 = func_residual_unit(body,num,stride2,symbol_true,name2)
            channels1 = j + one
        tmp2 = i + one
    name_bn0 = "bn1"
    bn0 = layers.batch_norm_infer(data = body, episilon = epsilon, scale = symbol_true, name = name_bn1))
    bn0 = relay.nn.relu(bn1)
    pool_size3 = relay.nn.global_avg_pool2d(data = relu1, layout = NCHW)
    strides3 = relay.nn.batch_flatten(data = pool2)
    padding3 = layers.dense_add_bias(data = flat, units = num_classes)
    body = relay.nn.softmax(fc1)
    return create_workload(relay.Function(relay.analysis.free_vars(net),net)
