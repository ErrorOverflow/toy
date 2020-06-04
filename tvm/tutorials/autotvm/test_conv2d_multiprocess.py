import multiprocessing
import os

def start():
    #os.system("/usr/bin/python3 /home/wml/tvm/tutorials/autotvm/test_conv2d.py")
    #换成pyhton3路径和test_conv2d路径
    os.system("/path/to/python3 /path/to/tvm/tutorials/autotvm/test_conv2d.py")
for i in range(multiprocessing.cpu_count()):
    #不能直接target tune_and_evaluate() 会阻塞
    #可以并行编译，等到后期真正tune优化的时候还需要解决进程通信，可以考虑文件通信或者socket端口通信。
    multiprocessing.Process(name='Tuner No.'+str(i), target=start, args=()).start()