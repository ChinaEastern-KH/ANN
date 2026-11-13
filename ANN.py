#导包
import torch                                            #PyTorch框架，封装了张量的各种操作
from torch.utils.data import TensorDataset              #数据集对象.
from torch.utils.data import DataLoader                 #数据加载器.
import torch.nn as nn                                   #neural network，封装了神经网络的各种操作
import torch.optim as optim                             #优化器
from sklearn.model_selection import train_test_split    #训练集和测试集的划分
import matplotlib.pyplot as plt                         #绘图
import numpy as np                                      #数组(矩阵)操作
import pandas as pd                                     #数据处理
import time                                             #时间模块
from torchsummary import summary                        #模型结构可视化
#构建函数和数据集
def create_dataset():
    data=pd.read_csv('./data/手机价格预测.csv')           #加载csv
    x,y=data.iloc[:,:-1],data.iloc[:,-1]                #获取x特征列y标签列
    x=x.astype(np.float32)                              #把特征列转成浮点型
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3,stratify=y)  #切分训练集和测试集
    train_dataset=TensorDataset(torch.tensor(x_train.values),torch.tensor(y_train.values))
    test_dataset=TensorDataset(torch.tensor(x_test.values),torch.tensor(y_test.values))         #把数据集封装成张量数据集
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))
#搭建神经网络
class PhonePriceModel(nn.Module):
    def __init__(self,input_dim,output_dim):        #初始化父类成员，搭建神经网络
        super().__init__()                          #初始化父类成员
        self.linear1=nn.Linear(input_dim,128)
        self.linear2=nn.Linear(128, 256)
        self.output=nn.Linear(256,output_dim)        #搭建神经网络
    def forward(self,x):                                       #定义前向传播方法
        x=torch.relu(self.linear1(x))                          #隐藏层1：加权求和+relu
        x=torch.relu(self.linear2(x))                          #隐藏层2：加权求和+relu
        x=self.output(x)                                       #隐藏层3：加权求和
        return x
#测试
if __name__=='__main__':
    train_dataset,test_dataset,input_dim,output_dim=create_dataset()
    model=PhonePriceModel(input_dim,output_dim)
    summary(model,input_size=(20,input_dim))