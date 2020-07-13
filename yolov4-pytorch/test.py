import torch
from torch import nn
from torchsummary import summary
from nets.CSPdarknet import darknet53, CSPDarkNet
from nets.yolo4 import YoloBody

if __name__ == "__main__":
 # 类定义
 class People:
  # 定义基本属性
  name = ''
  age = 0
  # 定义私有属性,私有属性在类外部无法直接进行访问
  __weight = 0

  # 定义构造方法
  def __init__(self, n, a, w):
   self.name = n
   self.age = a
   self.__weight = w

  def getWeight(self):
   return self.__weight

  def speak(self):
   # print("%s 说: 我 %d 岁。" %(self.name,self.age))
   print("{0:s} 说: 我 {1:d} 岁。体重 {2:d}kg".format(self.name, self.age, self.__weight))


 # 单继承示例
 class Student(People):
  grade = ''

  def __init__(self, n, a, w, g):
   # 调用父类的构函
   People.__init__(self, n, a, w)
   # super().__init__(self, n, a, w)
   self.grade = g

  # 覆写父类的方法
  def speak(self):
   # People.speak(self)
   print("%s 说: 我 %d 岁了，我在读 %d 年级，体重%dkg" % (self.name, self.age, self.grade, self.getWeight()))


 s = Student('ken', 10, 60, 3)
 s.speak()  # 调用子类方法
 super(Student, s).speak()  # 调用父类方法
