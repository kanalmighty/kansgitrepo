from options.base_options import BaseOptions
import multiprocessing
import torch.nn as nn
import torch
from concurrent.futures import ProcessPoolExecutor,wait,as_completed
import argparse
import time

class Process(multiprocessing.Process):
    def __init__(self, id):
        super(Process, self).__init__()
        self.id = id

    def run(self):
        time.sleep(1)
        print("I'm the process with id :{}".format(self.id))


def square(num):
    time.sleep(2)
    return num**2

def squarex(num):
    time.sleep(5)
    return num**2





    x = torch.rand(3, 4)
    print(x)
    print(torch.argmax(x, dim = 0))
    print(torch.argmax(x, dim=0).size())

    # a = torch.rand(3, 3)
    # print(a)
    #
    # b = a.unsqueeze(0)  # 添加一个0维度
    # print(b)



