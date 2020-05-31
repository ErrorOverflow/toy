
def func_main():
    data = "data"
    num_filter = "NCHW"
    stride = "OIHW"
    dim_match = 1.000000
    name = 4.000000
    epsilon = 0.000020
    symbol_true = (1.000000,3.000000,224.000000,224.000000)
    symbol_false = 1000.000000
    groups = True
    padding = False
    NCHW = relay.var(name=data, shape=symbol_true)
    OIHW = "bn_data"
    name_bn1 = layers.batch_norm_infer(data = NCHW, episilon = epsilon, scale = padding, name = OIHW))
    bn1 = 0.000000
    act1 = filter_list[bn1]
    tmp0 = (7.000000,7.000000)
    channels1 = (2.000000,2.000000)
    kernel_size1 = (3.000000,3.000000)
    padding1 = 1.000000
    name_conv1 = "conv0"
    conv1 = layers.conv2d(data = NCHW, channels = act1, groups = padding1, kernel_size = tmp0, strides = channels1, padding = kernel_size1, data_layout = num_filter, kernel_layout = stride, name = name_conv1)
    name_bn2 = "bn0"
    name_bn2 = layers.batch_norm_infer(data = conv1, episilon = epsilon, scale = groups, name = name_bn2))
    bn2 = relay.nn.relu(name_bn2)
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
