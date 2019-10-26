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



if __name__ == '__main__':
    # executor = ProcessPoolExecutor(max_workers=4)
    # fut1 = executor.submit(square, 2)
    # fut2 = executor.submit(squarex, 3)
    # # wait([fut1, fut2])
    # print(fut1.result(),fut2.result())
    #
    criteria = nn.CrossEntropyLoss()
    y = torch.randn(2,)
    y_hat = torch.randn(2, 2)
    loss = criteria(y_hat, y.long())
    print(loss.item())

    # a = torch.rand(3, 3)
    # print(a)
    #
    # b = a.unsqueeze(0)  # 添加一个0维度
    # print(b)



