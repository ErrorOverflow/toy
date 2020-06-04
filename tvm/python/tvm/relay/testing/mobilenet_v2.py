from tvm import relay
from .init import create_workload
from . import layers

def func_conv_block(data, name, channels, strides):
    kernel_size = (3,3)
    padding = (1,1)
    epsilon = 1e-05
    groups = 1
    symbol_true = True
    symbol_false = False
    NCHW = "NCHW"
    kernel_layout = layers.conv_kernel_layout(data_layout = NCHW, is_depthwise = symbol_false)
    name_conv = "_conv"
    name1 = name + name_conv
    conv = layers.conv2d(data = data, channels = int(channels), groups = groups, kernel_size = kernel_size, strides = strides, padding = padding, data_layout = NCHW, kernel_layout = kernel_layout, name = name1)
    name_bn = "_bn"
    name2 = name + name_bn
    bn = layers.batch_norm_infer(data = conv, epsilon = epsilon, scale = symbol_true, name = name2)
    act = relay.nn.relu(bn)
    return act

def func_separable_conv_block(data, name, depthwise_channels, pointwise_channels, downsample, layout):
    kernel_size = (3,3)
    padding = (1,1)
    epsilon = 1e-05
    symbol_true = True
    symbol_false = False
    strides = (2,2)
    strides_2 = (1,1)
    if downsample == symbol_false:
        tmp1 = 1
        strides = strides_2 * tmp1
    kernel_layout1 = layers.conv_kernel_layout(data_layout = layout, is_depthwise = symbol_true)
    name_conv1 = "_depthwise_conv1"
    name1 = name + name_conv1
    conv1 = layers.conv2d(data = data, channels = int(depthwise_channels), groups = depthwise_channels, kernel_size = kernel_size, strides = strides, padding = padding, data_layout = layout, kernel_layout = kernel_layout1, name = name1)
    name_bn1 = "_bn1"
    name2 = name + name_bn1
    bn1 = layers.batch_norm_infer(data = conv1, epsilon = epsilon, scale = symbol_true, name = name2)
    act1 = relay.nn.relu(bn1)
    groups = 1
    kernel_size2 = (1,1)
    strides2 = (1,1)
    padding2 = (0,0)
    kernel_layout2 = layers.conv_kernel_layout(data_layout = layout, is_depthwise = symbol_false)
    name_conv2 = "_conv2"
    name3 = name + name_conv2
    conv2 = layers.conv2d(data = act1, channels = int(pointwise_channels), groups = groups, kernel_size = kernel_size2, strides = strides2, padding = padding2, data_layout = layout, kernel_layout = kernel_layout2, name = name3)
    name_bn2 = "_bn2"
    name4 = name + name_bn2
    bn2 = layers.batch_norm_infer(data = conv2, epsilon = epsilon, scale = symbol_true, name = name4)
    act2 = relay.nn.relu(bn2)
    return act2

def func_main():
    num_classes = 1000
    data_shape = (1,3,224,224)
    alpha = 1
    symbol_true = True
    symbol_false = False
    layout = "NCHW"
    data_name = "data"
    data = relay.var(data_name, shape=data_shape)
    name_conv_block_1 = "conv_block_1"
    tmp2 = 32
    channels1 = tmp2 * alpha
    strides1 = (2,2)
    body = func_conv_block(data,name_conv_block_1,channels1,strides1)
    name_separable_conv_block_1 = "separable_conv_block_1"
    tmp3 = 32
    depthwise_channels1 = tmp3 * alpha
    tmp4 = 64
    pointwise_channels1 = tmp4 * alpha
    body = func_separable_conv_block(body,name_separable_conv_block_1,depthwise_channels1,pointwise_channels1,symbol_false,layout)
    name_separable_conv_block_2 = "separable_conv_block_2"
    tmp5 = 64
    depthwise_channels2 = tmp5 * alpha
    tmp6 = 128
    pointwise_channels2 = tmp6 * alpha
    body = func_separable_conv_block(body,name_separable_conv_block_2,depthwise_channels2,pointwise_channels2,symbol_true,layout)
    name_separable_conv_block_3 = "separable_conv_block_3"
    tmp7 = 128
    depthwise_channels3 = tmp7 * alpha
    tmp8 = 128
    pointwise_channels3 = tmp8 * alpha
    body = func_separable_conv_block(body,name_separable_conv_block_3,depthwise_channels3,pointwise_channels3,symbol_false,layout)
    name_separable_conv_block_4 = "separable_conv_block_4"
    tmp9 = 128
    depthwise_channels4 = tmp9 * alpha
    tmp10 = 256
    pointwise_channels4 = tmp10 * alpha
    body = func_separable_conv_block(body,name_separable_conv_block_4,depthwise_channels4,pointwise_channels4,symbol_true,layout)
    name_separable_conv_block_5 = "separable_conv_block_5"
    tmp11 = 256
    depthwise_channels5 = tmp11 * alpha
    tmp12 = 256
    pointwise_channels5 = tmp12 * alpha
    body = func_separable_conv_block(body,name_separable_conv_block_5,depthwise_channels5,pointwise_channels5,symbol_false,layout)
    name_separable_conv_block_6 = "separable_conv_block_6"
    tmp13 = 256
    depthwise_channels6 = tmp13 * alpha
    tmp14 = 512
    pointwise_channels6 = tmp14 * alpha
    body = func_separable_conv_block(body,name_separable_conv_block_6,depthwise_channels6,pointwise_channels6,symbol_true,layout)
    i_end = 12
    i_step = 1
    i = 7
    while i < i_end:
        name_separable_conv_block = "separable_conv_block_%d"
        name_separable_conv_block_i = name_separable_conv_block % i
        tmp16 = 512
        depthwise_channelsi = tmp16 * alpha
        tmp17 = 512
        pointwise_channelsi = tmp17 * alpha
        body = func_separable_conv_block(body,name_separable_conv_block_i,depthwise_channelsi,pointwise_channelsi,symbol_false,layout)
        i = i + i_step
    name_separable_conv_block_12 = "separable_conv_block_12"
    tmp18 = 512
    depthwise_channels12 = tmp18 * alpha
    tmp19 = 1024
    pointwise_channels12 = tmp19 * alpha
    body = func_separable_conv_block(body,name_separable_conv_block_12,depthwise_channels12,pointwise_channels12,symbol_true,layout)
    name_separable_conv_block_13 = "separable_conv_block_13"
    tmp20 = 1024
    depthwise_channels13 = tmp20 * alpha
    tmp21 = 1024
    pointwise_channels13 = tmp21 * alpha
    body = func_separable_conv_block(body,name_separable_conv_block_13,depthwise_channels13,pointwise_channels13,symbol_false,layout)
    pool = relay.nn.global_avg_pool2d(data = body, layout = layout)
    flatten = relay.nn.batch_flatten(data = pool)
    name_weight = "fc_weight"
    weight = relay.var(name_weight, shape=None)
    fc = relay.nn.dense(flatten, weight, num_classes)
    sm = relay.nn.softmax(fc)
    return create_workload(relay.Function(relay.analysis.free_vars(sm),sm))
