import os
import time
def start(i):
    command = "nvprof --unified-memory-profiling off --log-file /home/wml/tvm/tutorials/nvprof_log/inception_toy/output%i.log python3 /home/wml/tvm/tutorials/relay_test_net.py" % i
    os.system(command)

for i in range(12):
    start(i)
    time.sleep(20)