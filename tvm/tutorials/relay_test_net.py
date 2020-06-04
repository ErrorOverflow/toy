import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm

import pickle
from tvm.contrib import graph_runtime
from tvm.relay.testing import resnet
from tvm.relay.testing import mobilenet
from tvm.relay.testing import inception_v3
from tvm.relay.testing import mobilenet_v2
from tvm.relay.testing import inception_v3_2
from tvm.relay.testing import resnet_v2

# return resnet(units= [3, 4, 6, 3],
#               num_stages=4,
#               filter_list=[64, 256, 512, 1024, 2048],
#               num_classes=1000,
#               data_shape= (1,3, 224, 224),
#               bottle_neck=True,
#               dtype="float32")


def test_resnet():
    batch_size = 1
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    mod, params = resnet.get_workload()
    target = tvm.target.cuda()
    ctx = tvm.gpu()
    opt_level = 0
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)

    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**params)
    module.run()
    out = module.get_output(0).asnumpy()
    print(out.flatten()[0:10])

def test_mobilenet():
    batch_size = 1
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    mod, params = mobilenet.get_workload()
    target = tvm.target.cuda()
    ctx = tvm.gpu()
    opt_level = 0
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)

    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**params)
    module.run()
    out = module.get_output(0).asnumpy()
    print(out.flatten()[0:10])


def test_inception():
    batch_size = 1
    image_shape = (3, 299, 299)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    mod, params = inception_v3.get_workload()
    target = tvm.target.cuda()
    ctx = tvm.gpu()
    opt_level = 0
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)

    module = graph_runtime.create(graph, lib, ctx)
    module.set_input("data", data)
    module.set_input(**params)
    module.run()
    out = module.get_output(0).asnumpy()
    print(out.flatten()[0:10])

def test_resnet_v2():
    batch_size = 1
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    
    mod2, params2 = resnet_v2.func_main()

    # mod, params = resnet.get_workload()

    target = tvm.target.cuda()
    ctx = tvm.gpu()
    opt_level = 0
    with relay.build_config(opt_level=opt_level):
        graph2, lib2, params2 = relay.build_module.build(
            mod2, target, params=params2)

    # with relay.build_config(opt_level=opt_level):
    #     graph, lib, params = relay.build_module.build(
    #         mod, target, params=params)

    # module = graph_runtime.create(graph, lib, ctx)
    # module.set_input("data", data)
    # print(params.__len__())
    # module.set_input(**params)
    # module.run()
    # out = module.get_output(0).asnumpy()
    # print(out.flatten()[0:10])

    module = graph_runtime.create(graph2, lib2, ctx)
    module.set_input("data", data)
    print(params2.__len__())
    module.set_input(**params2)
    module.run()
    out = module.get_output(0).asnumpy()
    print(out.flatten()[0:10])

def test_mobilenet_v2():
    batch_size = 1
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    mod2, params2 = mobilenet_v2.func_main()
    # mod, params = mobilenet.get_workload()

    target = tvm.target.cuda()
    ctx = tvm.gpu()
    opt_level = 0
    with relay.build_config(opt_level=opt_level):
        graph2, lib2, params2 = relay.build_module.build(
            mod2, target, params=params2)

    # with relay.build_config(opt_level=opt_level):
    #     graph, lib, params = relay.build_module.build(
    #         mod, target, params=params)
    # module = graph_runtime.create(graph, lib, ctx)
    # module.set_input("data", data)
    # module.set_input(**params)
    # module.run()
    # out = module.get_output(0).asnumpy()
    # print(out.flatten()[0:10])   

    module = graph_runtime.create(graph2, lib2, ctx)
    module.set_input("data", data)
    module.set_input(**params2)
    module.run()
    out = module.get_output(0).asnumpy()
    print(out.flatten()[0:10])

def test_inception_2():
    batch_size = 1
    image_shape = (3, 299, 299)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    mod2, params2 = inception_v3_2.func_main()
    # mod, params = inception_v3.get_workload()
    
    target = tvm.target.cuda()
    ctx = tvm.gpu()
    opt_level = 0
    with relay.build_config(opt_level=opt_level):
        graph2, lib2, params2 = relay.build_module.build(
            mod2, target, params=params2)

    # with relay.build_config(opt_level=opt_level):
    #     graph, lib, params = relay.build_module.build(
    #         mod, target, params=params)
    # print(len(params))
    # print(len(params2))
    # module = graph_runtime.create(graph, lib, ctx)
    # module.set_input("data", data)
    # module.set_input(**params)
    # module.run()
    # out = module.get_output(0).asnumpy()
    # print(out.flatten()[0:10])

    module = graph_runtime.create(graph2, lib2, ctx)
    module.set_input("data", data)
    module.set_input(**params2)
    module.run()
    out = module.get_output(0).asnumpy()
    print(out.flatten()[0:10])

if __name__ == "__main__":
    test_inception()
