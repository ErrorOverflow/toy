from tvm import relay
from .init import create_workload
from . import layers

def func_residual_unit(data, num_filter, stride, dim_match, name):
    epsilon = 2e-05
    symbol_true = True
    symbol_false = False
    groups = 1
    padding = (0,0)
    NCHW = "NCHW"
    OIHW = "OIHW"
    name_bn1 = "_bn1"
    name_bn1 = name + name_bn1
    bn1 = layers.batch_norm_infer(data = data, epsilon = epsilon, scale = symbol_true, name = name_bn1)
    act1 = relay.nn.relu(bn1)
    tmp0 = 0.25
    channels1 = num_filter * tmp0
    kernel_size1 = (3,3)
    padding1 = (1,1)
    name_conv1 = "_conv1"
    name_conv1 = name + name_conv1
    conv1 = layers.conv2d(data = act1, channels = int(channels1), groups = groups, kernel_size = kernel_size1, strides = stride, padding = padding1, data_layout = NCHW, kernel_layout = OIHW, name = name_conv1)
    name_bn2 = "_bn2"
    name_bn2 = name + name_bn2
    bn2 = layers.batch_norm_infer(data = conv1, epsilon = epsilon, scale = symbol_false, name = name)
    act2 = relay.nn.relu(bn2)
    kernel_size2 = (3,3)
    strides2 = (1,1)
    padding2 = (1,1)
    name_conv2 = "_conv2"
    name_conv2 = name + name_conv2
    conv2 = layers.conv2d(data = act2, channels = int(num_filter), groups = groups, kernel_size = kernel_size2, strides = strides2, padding = padding2, data_layout = NCHW, kernel_layout = OIHW, name = name_conv2)
    r = relay.add(conv2, data)
    if dim_match == symbol_false:
        kernel_size3 = (1,1)
        name_shortcut = "_sc"
        name_shortcut = name + name_shortcut
        shortcut = layers.conv2d(data = act1, channels = int(num_filter), groups = groups, kernel_size = kernel_size3, strides = stride, padding = padding, data_layout = NCHW, kernel_layout = OIHW, name = name_shortcut)
        r = relay.add(conv2, shortcut)
    return r

def func_main():
    data_name = "data"
    NCHW = "NCHW"
    OIHW = "OIHW"
    one = 1
    num_stages = 4
    epsilon = 2e-05
    data_shape = (1,3,224,224)
    units = [2,2,2,2]
    filter_list = [64,64,128,256,512]
    num_classes = 1000
    symbol_true = True
    symbol_false = False
    data = relay.var(data_name, shape=data_shape)
    name_bn_data = "bn_data"
    bn_data = layers.batch_norm_infer(data = data, epsilon = epsilon, scale = symbol_false, name = name_bn_data)
    tmp2 = 0
    channels1 = filter_list[tmp2]
    kernel_size1 = (7,7)
    strides1 = (2,2)
    padding1 = (3,3)
    groups = 1
    name_conv0 = "conv0"
    conv0 = layers.conv2d(data = bn_data, channels = int(channels1), groups = groups, kernel_size = kernel_size1, strides = strides1, padding = padding1, data_layout = NCHW, kernel_layout = OIHW, name = name_conv0)
    name_bn0 = "bn0"
    bn0 = layers.batch_norm_infer(data = conv0, epsilon = epsilon, scale = symbol_true, name = name_bn0)
    bn0 = relay.nn.relu(bn0)
    pool_size3 = (3,3)
    strides3 = (2,2)
    padding3 = (1,1)
    body = relay.nn.max_pool2d(data = bn0, pool_size = pool_size3, strides = strides3, padding = padding3)
    tuple1 = (1,1)
    tuple2 = (2,2)
    i = 0
    while i < num_stages:
        stride = tuple2 * one
        judge = 0
        if i == judge:
            stride = tuple1 * one
        tmp5 = 1
        filter_list_index1 = i + tmp5
        num = filter_list[filter_list_index1]
        j_max = 1000
        i_1000 = (i, j_max)
        name_stage = "stage%d_unit%d"
        name1 = name_stage % i_1000
        body = func_residual_unit(body,num,stride,symbol_false,name1)
        j_range = units[i]
        tmp6 = 1
        j_range = j_range - tmp6
        stride2 = (1,1)
        j = 0
        while j < j_range:
            tmp8 = 1
            filter_list_index2 = i + tmp8
            num = filter_list[filter_list_index2]
            i_j = (i, j)
            name2 = name_stage % i_j
            body = func_residual_unit(body,num,stride2,symbol_true,name2)
            j = j + one
        i = i + one
    name_bn1 = "bn1"
    bn1 = layers.batch_norm_infer(data = body, epsilon = epsilon, scale = symbol_true, name = name_bn1)
    relu1 = relay.nn.relu(bn1)
    pool2 = relay.nn.global_avg_pool2d(data = relu1, layout = NCHW)
    flat = relay.nn.batch_flatten(data = pool2)
    fc1 = layers.dense_add_bias(data = flat, units = num_classes)
    net = relay.nn.softmax(fc1)
    return create_workload(relay.Function(relay.analysis.free_vars(net),net))
