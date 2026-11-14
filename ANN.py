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
def create_dataset():                                   #构建函数和数据集
    data=pd.read_csv('./data/手机价格预测.csv')           #加载csv
    x,y=data.iloc[:,:-1],data.iloc[:,-1]                #获取x特征列y标签列
    x=x.astype(np.float32)                              #把特征列转成浮点型
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3,stratify=y)  #切分训练集和测试集
    train_dataset=TensorDataset(torch.tensor(x_train.values),torch.tensor(y_train.values))
    test_dataset=TensorDataset(torch.tensor(x_test.values),torch.tensor(y_test.values))         #把数据集封装成张量数据集
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))
class PhonePriceModel(nn.Module):                   #搭建神经网络
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
def train(train_dataset,input_dim,output_dim):                              #模型训练
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)       #创建数据加载器
    model = PhonePriceModel(input_dim, output_dim)                          #创建神经网络模型
    criterion=nn.CrossEntropyLoss()                                         #定义损失函数
    optimizer=optim.SGD(model.parameters(),lr=0.001)                         #创建优化器对象
    epochs=50                                                               #定义变量，记录训练总轮数
    for epoch in range(epochs):                                             #开始每轮训练
        total_loss,batch_num=0.0,0                                          #定义变量，记录每次的损失值、训练批次数
        start=time.time()                                                   #定义变量，表示训练开始时间
        for x,y in train_loader:                                            #开始本轮各批次训练
            model.train()                                                   #训练模式
            y_pred=model(x)                                                 #模型预测
            loss=criterion(y_pred,y)                                        #计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                                                #梯度清零，反向传播，优化参数
            total_loss+=loss.item()                                         #累加每批次平均损失值
            batch_num+=1
        print(f'epoch:{epoch+1},loss:{total_loss/batch_num:.4f},time:{time.time()-start:.2f}s')
    torch.save(model.state_dict(),'./model/phone.pth')                    #保存模型
def evaluate(test_dataset,input_dim,output_dim):                            #模型测试
    model=PhonePriceModel(input_dim,output_dim)                             #创建神经网络分类对象
    model.load_state_dict(torch.load('./model/phone.pth'))                  #加载模型参数
    test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False)       #创建测试集的数据加载器对象
    correct=0                                                               #定义预测正确样本个数的变量
    for x,y in test_loader:                                                 #从数据加载器中获取每批次数据
        model.eval()                                                        #切换到测试模式
        y_pred=model(x)                                                     #模型预测
        y_pred=torch.argmax(y_pred,dim=1)                                   #获取最大值对应的下标和类别
        correct+=(y_pred==y).sum()                                          #统计预测正确样本个数
    print(f'准确率：{correct/len(test_dataset):.4f}')
if __name__=='__main__':                                                #测试
    train_dataset,test_dataset,input_dim,output_dim=create_dataset()
    #model=PhonePriceModel(input_dim,output_dim)
    #ummary(model,input_size=(20,input_dim))
    #train(train_dataset,input_dim,output_dim)
    evaluate(test_dataset, input_dim, output_dim)