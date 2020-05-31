from tvm import relay
from .init import create_workload
from . import layers

def func_Conv(data, num_filter, kernel, stride, pad, name, suffix):
    ConvName = (name, suffix)
    epsilon = 0.000020
    symbol_false = False
    groups = 1.000000
    NCHW = "NCHW"
    OIHW = "OIHW"
    conv_name = "%s%s_conv1"
    name_conv = conv_name % ConvName
    conv = layers.conv2d(data = data, channels = num_filter, groups = groups, kernel_size = kernel, strides = stride, padding = pad, data_layout = NCHW, kernel_layout = OIHW, name = name_conv)
    bn_name = "%s%s_bn"
    name_bn = bn_name % ConvName
    bn = layers.batch_norm_infer(data = conv, episilon = epsilon, scale = symbol_false, name = name_bn)
    act = relay.nn.relu(bn)
    return(act)

def func_Pooling(data, kernel, stride, pad, pool_type, name):
    AVG = "avg"
    symbol_true = True
    r = relay.nn.max_pool2d(data = data, pool_size = kernel, strides = stride, padding = pad)
    if pool_type == AVG:
        r = func_avg_pool2d(data,kernel,stride,pad,symbol_true)
    return(r)

def func_Inception7A(data, num_1x1, num_3x3_red, num_3x3_1, num_3x3_2, num_5x5_red, num_5x5, pool, proj, name):
    kernel = (1.000000,1.000000)
    stride = (1.000000,1.000000)
    pad = (0.000000,0.000000)
    suffix = " "
    s_conv = "%s_conv"
    name_tower_1x1 = s_conv % name
    tower_1x1 = func_Conv(data,num_1x1,kernel,stride,pad,name_tower_1x1,suffix)
    s_tower = "%s_tower"
    name_tower = s_tower % name
    suffix_tower_5x5 = "_conv"
    tower_5x5 = func_Conv(data,num_5x5_red,kernel,stride,pad,name_tower,suffix_tower_5x5)
    kernel_tower_5x5 = (5.000000,5.000000)
    pad_tower_5x5 = (2.000000,2.000000)
    suffix_tower_5x5_2 = "_conv_1"
    tower_5x5 = func_Conv(tower_5x5,num_5x5,kernel_tower_5x5,stride,pad_tower_5x5,name_tower,suffix_tower_5x5_2)
    s_tower_1 = "s_tower_1"
    name_tower_1 = s_tower_1 % name
    suffix_tower_3x3 = "_conv"
    tower_3x3 = func_Conv(data,num_3x3_red,kernel,stride,pad,name_tower_1,suffix_tower_3x3)
    suffix_tower_3x3_2 = "_conv_1"
    kernel_5 = (3.000000,3.000000)
    pad_5 = (1.000000,1.000000)
    tower_3x3 = func_Conv(tower_3x3,num_3x3_1,kernel_5,stride,pad_5,name_tower_1,suffix_tower_3x3_2)
    suffix_tower_3x3_3 = "_conv_2"
    tower_3x3 = func_Conv(tower_3x3,num_3x3_2,kernel_5,stride,pad_5,name_tower_1,suffix_tower_3x3_3)
    pool_name = (pool, name)
    s_pool = "%s_pool_%s_pool"
    name_pooling = s_pool % pool_name
    pooling = func_Pooling(data,kernel_5,stride,pad_5,pool,name_pooling)
    s_tower_2 = "s_tower_2"
    name_tower_2 = s_tower_2 % name
    suffix_cproj = "_conv"
    cproj = func_Conv(pooling,proj,kernel,stride,pad,name_tower_2,suffix_cproj)
    axis = 1.000000
    return(concat)

def func_Inception7B(data, num_3x3, num_d3x3_red, num_d3x3_1, num_d3x3_2, pool, name):
    kernel_1 = (3.000000,3.000000)
    stride_1 = (2.000000,2.000000)
    pad_1 = (0.000000,0.000000)
    s_conv = "%s_conv"
    name_1 = s_conv % name
    suffix = " "
    tower_3x3 = func_Conv(data,num_3x3,kernel_1,stride_1,pad_1,name_1,suffix)
    kernel = (1.000000,1.000000)
    stride = (1.000000,1.000000)
    pad = (0.000000,0.000000)
    s_tower = "%s_tower"
    name_2 = s_tower % name
    suffix_2 = "_conv"
    tower_d3x3 = func_Conv(data,num_d3x3_red,kernel,stride,pad,name_2,suffix_2)
    kernel_3 = (3.000000,3.000000)
    stride_3 = (1.000000,1.000000)
    pad_3 = (1.000000,1.000000)
    suffix_3 = "_conv_1"
    tower_d3x3 = func_Conv(tower_d3x3,num_d3x3_1,kernel_3,stride_3,pad_3,name_2,suffix_3)
    kernel_4 = (3.000000,3.000000)
    stride_4 = (2.000000,2.000000)
    pad_4 = (0.000000,0.000000)
    suffix_4 = "_conv_2"
    tower_d3x3 = func_Conv(tower_d3x3,num_d3x3_2,kernel_4,stride_4,pad_4,name_2,suffix_4)
    kernel_5 = (3.000000,3.000000)
    stride_5 = (2.000000,2.000000)
    pad_5 = (0.000000,0.000000)
    pool_type = "max"
    s_pool = "max_pool_%s_pool"
    name5 = s_pool % name
    pooling = func_Pooling(data,kernel_5,stride_5,pad_5,pool_type,name5)
    axis = 1.000000
    return(concat)

def func_Inception7C(data, num_1x1, num_d7_red, num_d7_1, num_d7_2, num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4, pool, proj, name):
    stride = (1.000000,1.000000)
    kernel = (1.000000,1.000000)
    pad = (0.000000,0.000000)
    s_conv = "%s_conv"
    name1 = s_conv % name
    suffix = " "
    tower_1x1 = func_Conv(data,num_1x1,kernel,stride,pad,name1,suffix)
    s_tower = "%s_tower"
    name2 = s_tower % name
    suffix_2 = "_conv"
    tower_d7 = func_Conv(data,num_d7_red,kernel,stride,pad,name2,suffix_2)
    kernel_3 = (1.000000,7.000000)
    pad_3 = (0.000000,3.000000)
    suffix_3 = "_conv_1"
    tower_d7 = func_Conv(tower_d7,num_d7_1,kernel_3,stride,pad_3,name2,suffix_3)
    kernel_4 = (7.000000,1.000000)
    pad_4 = (3.000000,0.000000)
    suffix_4 = "_conv_2"
    tower_d7 = func_Conv(tower_d7,num_d7_2,kernel_4,stride,pad_4,name2,suffix_4)
    s_tower_1 = "%s_tower_1"
    name5 = s_tower_1 % name
    tower_q7 = func_Conv(data,num_q7_red,kernel,stride,pad,name5,suffix_2)
    tower_q7 = func_Conv(tower_q7,num_q7_1,kernel_4,stride,pad_4,name5,suffix_3)
    tower_q7 = func_Conv(tower_q7,num_q7_2,kernel_3,stride,pad_3,name5,suffix_4)
    suffix_8 = "_conv_3"
    tower_q7 = func_Conv(tower_q7,num_q7_3,kernel_4,stride,pad_4,name5,suffix_8)
    suffix_9 = "_conv_4"
    tower_q7 = func_Conv(tower_q7,num_q7_4,kernel_3,stride,pad_3,name5,suffix_9)
    kernel_10 = (3.000000,3.000000)
    stride_10 = (1.000000,1.000000)
    pad_10 = (1.000000,1.000000)
    s_pool = "%s_pool_%s_pool"
    pool_name = (pool, name)
    name_10 = s_pool % pool_name
    pooling = func_Pooling(data,kernel_10,stride_10,pad_10,pool,name_10)
    kernel_11 = (1.000000,1.000000)
    s_tower_2 = "%s_tower_2"
    name_11 = s_tower_2 % name
    suffix_11 = "_conv"
    cproj = func_Conv(pooling,proj,kernel_11,stride,pad,name_11,suffix_11)
    axis = 1.000000
    return(concat)

def func_Inception7D(data, num_3x3_red, num_3x3, num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3, pool, name):
    stride = (1.000000,1.000000)
    kernel = (1.000000,1.000000)
    pad = (0.000000,0.000000)
    s_tower = "%s_tower"
    name_1 = s_tower % name
    suffix_1 = "_conv"
    tower_3x3 = func_Conv(data,num_3x3_red,kernel,stride,pad,name_1,suffix_1)
    kernel_2 = (3.000000,3.000000)
    stride_2 = (2.000000,2.000000)
    pad_2 = (0.000000,0.000000)
    suffix_2 = "_conv_1"
    tower_3x3 = func_Conv(tower_3x3,num_3x3,kernel_2,stride_2,pad_2,name_1,suffix_2)
    s_tower_1 = "%s_tower_1"
    name_3 = s_tower % name
    suffix_3 = "_conv"
    tower_d7_3x3 = func_Conv(data,num_d7_3x3_red,kernel,stride,pad,name_3,suffix_3)
    kernel_4 = (1.000000,7.000000)
    pad_4 = (0.000000,3.000000)
    suffix_4 = "_conv_1"
    tower_d7_3x3 = func_Conv(tower_d7_3x3,num_d7_1,kernel_4,stride,pad_4,name_3,suffix_4)
    kernel_5 = (7.000000,1.000000)
    pad_5 = (3.000000,0.000000)
    suffix_5 = "_conv_2"
    tower_d7_3x3 = func_Conv(tower_d7_3x3,num_d7_2,kernel_5,stride,pad_5,name_3,suffix_5)
    kernel_6 = (3.000000,3.000000)
    stride_6 = (2.000000,2.000000)
    suffix_6 = "_conv_3"
    tower_d7_3x3 = func_Conv(tower_d7_3x3,num_d7_3x3,kernel_6,stride_6,pad,name_3,suffix_6)
    kernel_7 = (3.000000,3.000000)
    stride_7 = (2.000000,2.000000)
    pad_7 = (0.000000,0.000000)
    s_pool = "%s_pool_%s_pool"
    pool_name = (pool, name)
    name_7 = s_pool % pool_name
    pooling = func_Pooling(data,kernel_7,stride_7,pool,pad_7,name_7)
    axis = 1.000000
    return(concat)

def func_Inception7E(data, num_1x1, num_d3_red, num_d3_1, num_d3_2, num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2, pool, proj, name):
    stride = (1.000000,1.000000)
    kernel = (1.000000,1.000000)
    pad = (0.000000,0.000000)
    suffix = " "
    kernel_1 = (1.000000,1.000000)
    s_conv = "%s_conv"
    name_1 = s_conv % name
    tower_1x1 = func_Conv(data,num_1x1,kernel_1,stride,pad,name_1,suffix)
    s_tower = "%s_tower"
    name_2 = s_tower % name
    suffix_2 = "_conv"
    tower_d3 = func_Conv(data,num_d3_red,kernel,stride,pad,name_2,suffix_2)
    kernel_3 = (1.000000,3.000000)
    pad_3 = (0.000000,1.000000)
    suffix_3 = "_mixed_conv"
    tower_d3_a = func_Conv(tower_d3,num_d3_1,kernel_3,stride,pad_3,name_2,suffix_3)
    kernel_4 = (3.000000,1.000000)
    pad_4 = (1.000000,0.000000)
    suffix_4 = "_mixed_conv_1"
    tower_d3_b = func_Conv(tower_d3,num_d3_2,kernel_4,stride,pad_4,name_2,suffix_4)
    s_tower_1 = "%s_tower_1"
    name_5 = s_tower_1 % name
    suffix_5 = "_conv"
    tower_3x3_d3 = func_Conv(data,num_3x3_d3_red,kernel,stride,pad,name_5,suffix_5)
    kernel_6 = (3.000000,3.000000)
    pad_6 = (1.000000,1.000000)
    suffix_6 = "_conv_1"
    tower_3x3_d3 = func_Conv(tower_3x3_d3,num_3x3,kernel_6,stride,pad_6,name_5,suffix_6)
    kernel_7 = (1.000000,3.000000)
    pad_7 = (0.000000,1.000000)
    suffix_7 = "_mixed_conv"
    tower_3x3_d3_a = func_Conv(tower_3x3_d3,num_3x3_d3_1,kernel_7,stride,pad_7,name_5,suffix_7)
    kernel_8 = (3.000000,1.000000)
    pad_8 = (1.000000,0.000000)
    suffix_8 = "_mixed_conv_1"
    tower_3x3_d3_b = func_Conv(tower_3x3_d3,num_3x3_d3_2,kernel_8,stride,pad_8,name_5,suffix_8)
    kernel_9 = (3.000000,3.000000)
    stride_9 = (1.000000,1.000000)
    pad_9 = (1.000000,1.000000)
    s_pool = "%s_pool_%s_pool"
    pool_name = (pool, name)
    name_9 = s_pool % pool_name
    pooling = func_Pooling(data,kernel_9,stride_9,pad_9,pool,name_9)
    kernel_10 = (1.000000,1.000000)
    s_tower_2 = "%s_tower_2"
    name_10 = s_tower_2 % name
    suffix_10 = "_conv"
    cproj = func_Conv(pooling,proj,kernel_10,stride,pad,name_10,suffix_10)
    axis = 1.000000
    return(concat)

def func_main():
    data_shape = (1.000000,3.000000,299.000000,299.000000)
    data_name = "data"
    data = relay.var(name=data_name, shape=data_shape)
    num_filter_1 = 32.000000
    kernel_1 = (3.000000,3.000000)
    stride_1 = (2.000000,2.000000)
    pad = (0.000000,0.000000)
    name_1 = "conv"
    suffix = " "
    conv = func_Conv(data,num_filter_1,kernel_1,stride_1,pad,name_1,suffix)
    num_filter_2 = 32.000000
    kernel_2 = (3.000000,3.000000)
    stride = (1.000000,1.000000)
    name_2 = "conv_1"
    conv_1 = func_Conv(conv,num_filter_2,kernel_2,stride,pad,name_2,suffix)
    num_filter_3 = 64.000000
    kernel_3 = (3.000000,3.000000)
    pad_3 = (1.000000,1.000000)
    name_3 = "conv_2"
    conv_2 = func_Conv(conv_1,num_filter_3,kernel_3,stride,pad_3,name_3,suffix)
    kernel_4 = (3.000000,3.000000)
    stride_4 = (2.000000,2.000000)
    pool_type = "max"
    name_4 = "pool"
    pool = func_Pooling(conv_2,kernel_4,stride_4,pool_type,pad,name_4)
    num_filter_5 = 80.000000
    kernel_5 = (1.000000,1.000000)
    name_5 = "conv_3"
    conv_3 = func_Conv(pool,num_filter_5,kernel_5,stride,pad,name_5,suffix)
    num_filter_6 = 80.000000
    kernel_6 = (3.000000,3.000000)
    name_6 = "conv_4"
    conv_4 = func_Conv(conv_3,num_filter_6,kernel_6,stride,pad,name_6,suffix)
    name_7 = "pool1"
    pool1 = func_Pooling(conv_4,kernel_4,stride_4,pool_type,pad,name_7)
    avg = "avg"
    mixed = "mixed"
    tmp1 = 64.000000
    tmp2 = 64.000000
    tmp3 = 96.000000
    tmp4 = 96.000000
    tmp5 = 48.000000
    tmp6 = 64.000000
    tmp7 = 32.000000
    in3a = func_Inception7A(pool1,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,avg,tmp7,mixed)
    mixed_1 = "mixed_1"
    tmp8 = 64.000000
    tmp9 = 64.000000
    tmp10 = 96.000000
    tmp11 = 96.000000
    tmp12 = 48.000000
    tmp13 = 64.000000
    tmp14 = 64.000000
    in3b = func_Inception7A(in3a,tmp8,tmp9,tmp10,tmp11,tmp12,tmp13,avg,tmp14,mixed_1)
    mixed_2 = "mixed_2"
    tmp15 = 64.000000
    tmp16 = 64.000000
    tmp17 = 96.000000
    tmp18 = 96.000000
    tmp19 = 48.000000
    tmp20 = 64.000000
    tmp21 = 64.000000
    in3c = func_Inception7A(in3b,tmp15,tmp16,tmp17,tmp18,tmp19,tmp20,avg,tmp21,mixed_2)
    mixed_3 = "mixed_3"
    max = "max"
    tmp22 = 384.000000
    tmp23 = 64.000000
    tmp24 = 96.000000
    tmp25 = 96.000000
    in3d = func_Inception7B(in3c,tmp22,tmp23,tmp24,tmp25,max,mixed_3)
    mixed_4 = "mixed_4"
    tmp26 = 192.000000
    tmp27 = 128.000000
    tmp28 = 128.000000
    tmp29 = 192.000000
    tmp30 = 128.000000
    tmp31 = 128.000000
    tmp32 = 128.000000
    tmp33 = 128.000000
    tmp34 = 192.000000
    tmp35 = 192.000000
    in4a = func_Inception7C(in3d,tmp26,tmp27,tmp28,tmp29,tmp30,tmp31,tmp32,tmp33,tmp34,avg,tmp35,mixed_4)
    mixed_5 = "mixed_5"
    tmp36 = 192.000000
    tmp37 = 160.000000
    tmp38 = 160.000000
    tmp39 = 192.000000
    tmp40 = 160.000000
    tmp41 = 160.000000
    tmp42 = 160.000000
    tmp43 = 160.000000
    tmp44 = 192.000000
    tmp45 = 192.000000
    in4b = func_Inception7C(in4a,tmp36,tmp37,tmp38,tmp39,tmp40,tmp41,tmp42,tmp43,tmp44,avg,tmp45,mixed_5)
    mixed_6 = "mixed_6"
    tmp46 = 192.000000
    tmp47 = 160.000000
    tmp48 = 160.000000
    tmp49 = 192.000000
    tmp50 = 160.000000
    tmp51 = 160.000000
    tmp52 = 160.000000
    tmp53 = 160.000000
    tmp54 = 192.000000
    tmp55 = 192.000000
    in4c = func_Inception7C(in4b,tmp46,tmp47,tmp48,tmp49,tmp50,tmp51,tmp52,tmp53,tmp54,avg,tmp55,mixed_6)
    mixed_7 = "mixed_7"
    tmp56 = 192.000000
    tmp57 = 192.000000
    tmp58 = 192.000000
    tmp59 = 192.000000
    tmp60 = 192.000000
    tmp61 = 192.000000
    tmp62 = 192.000000
    tmp63 = 192.000000
    tmp64 = 192.000000
    tmp65 = 192.000000
    in4d = func_Inception7C(in4c,tmp56,tmp57,tmp58,tmp59,tmp60,tmp61,tmp62,tmp63,tmp64,avg,tmp65,mixed_7)
    mixed_8 = "mixed_8"
    tmp66 = 192.000000
    tmp67 = 320.000000
    tmp68 = 192.000000
    tmp69 = 192.000000
    tmp70 = 192.000000
    tmp71 = 192.000000
    in4e = func_Inception7D(in4d,tmp66,tmp67,tmp68,tmp69,tmp70,tmp71,max,mixed_8)
    mixed_9 = "mixed_9"
    tmp72 = 320.000000
    tmp73 = 384.000000
    tmp74 = 384.000000
    tmp75 = 384.000000
    tmp76 = 448.000000
    tmp77 = 384.000000
    tmp78 = 384.000000
    tmp79 = 384.000000
    tmp80 = 192.000000
    in5a = func_Inception7E(in4e,tmp72,tmp73,tmp74,tmp75,tmp76,tmp77,tmp78,tmp79,avg,tmp80,mixed_9)
    mixed_10 = "mixed_10"
    tmp81 = 320.000000
    tmp82 = 384.000000
    tmp83 = 384.000000
    tmp84 = 384.000000
    tmp85 = 448.000000
    tmp86 = 384.000000
    tmp87 = 384.000000
    tmp88 = 384.000000
    tmp89 = 192.000000
    in5b = func_Inception7E(in5a,tmp81,tmp82,tmp83,tmp84,tmp85,tmp86,tmp87,tmp88,max,tmp89,mixed_10)
    kernel_8 = (8.000000,8.000000)
    stride_8 = (1.000000,1.000000)
    name_8 = "global_pool"
    pool = func_Pooling(in5b,kernel_8,stride_8,avg,pad,name_8)
    flatten = relay.nn.batch_flatten(data = pool)
    name_fc1_weight = "fc1_weight"
    None = "None"
    fc1_weight = relay.var(name=name_fc1_weight, shape=None)
    tmp90 = 1000.000000
    fc1 = relay.nn.dense(flatten, fc1_weight, tmp90)
    name_fc2_bias = "fc2_bias"
    fc2_bias = relay.var(name=name_fc2_bias, shape=None)
    fc1 = relay.nn.bias_add(fc1, fc2_bias)
    inception_v3 = relay.nn.softmax(fc1)
    return create_workload(relay.Function(relay.analysis.free_vars(inception_v3),inception_v3)
