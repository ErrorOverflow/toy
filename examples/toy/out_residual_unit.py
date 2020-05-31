
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
    tmp0 = 0.250000
    channels1 = num_filter * tmp0
    kernel_size1 = (3.000000,3.000000)
    padding1 = (1.000000,1.000000)
    name_conv1 = "conv1"
    conv1 = layers.conv2d(data = act1, channels = channels1, groups = groups, kernel_size = kernel_size1, strides = stride, padding = padding1, data_layout = NCHW, kernel_layout = OIHW, name = name_conv1)
    name_bn2 = "bn2"
    name_bn2 = name + name_bn2
    bn2 = layers.batch_norm_infer(data = conv1, episilon = epsilon, scale = symbol_false, name = name))
    act2 = relay.nn.relu(bn2)
    kernel_size2 = (3.000000,3.000000)
    strides2 = (1.000000,1.000000)
    padding2 = (1.000000,1.000000)
    name_conv2 = "conv2"
    conv2 = layers.conv2d(data = act2, channels = num_filter, groups = groups, kernel_size = kernel_size2, strides = strides2, padding = padding2, data_layout = NCHW, kernel_layout = OIHW, name = name_conv2)
    tmp1 = 1.000000
    shortcut = data * tmp1
    if dim_match == symbol_false:
        kernel_size3 = (1.000000,1.000000)
        name_shortcut = "sc"
        shortcut = layers.conv2d(data = act1, channels = num_filter, groups = groups, kernel_size = kernel_size3, strides = stride, padding = padding, data_layout = NCHW, kernel_layout = OIHW, name = name_shortcut)
    r = relay.op.add(conv2, shortcut)
    return(r)
