import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim




class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.

    arg：
        in_channels（int）：输入序列数据中的通道数
        out_channels（int）：卷积产生的通道数
        kernel_size（int）：图卷积内核的大小
        t_kernel_size（int）：时间卷积核的大小
        t_stride（整数，可选）：时间卷积的跨度。默认值：1
        t_padding（int，可选）：在控件的两边都添加了时间零填充 输入。默认值：0
        t_dilation（整数，可选）：时间内核元素之间的间距。   默认值：1
        偏见（布尔型，可选）：如果为``True''，则向输出添加可学习的偏见。 默认值：``True``
    形状：
        -Input [0]：以（N,in_channels,T_ {in},V）格式输入图形序列
        -Input [1]：以（K,V,V）格式输入图邻接矩阵
        -output[0]：Outpu图形序列,格式为（N,out_channels,T_ {out},V）`
        -Output [1]：输出数据的图形邻接矩阵，格式为（K,V,V）`
        哪里
            ：ma：`N`是批处理大小，
            ：math：`K`是空间内核大小，如：math：`K == kernel_size [1]`，
            ：math：`T_ {in} / T_ {out}`是输入/输出序列的长度,
            V是图形节点的数量。

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    # 输入是2，输出是5 ,kernel_size=8
    def __init__(self,
                 in_channels,     #2
                 out_channels,     #5
                 kernel_size,      #8
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size   #8
        self.conv = nn.Conv2d(
            in_channels,    #2
            out_channels,   #5  5个通道 5个卷积核
            kernel_size=(t_kernel_size, 1),       #(1,1)---
            padding=(t_padding, 0),               #(0,0)相当于没有padding
            stride=(t_stride, 1),                 #(1,1)
            dilation=(t_dilation, 1),             #(1,1)
            bias=bias)           #到这里 卷积核为1，维度没有改变，所以卷积之后的x只有第二个维度变了，2--》5
#x:[ batch_sise,2 , 8,57]
    def forward(self, x, A):    #A是[8,57,57]存的所有人的距离
        assert A.size(0) == self.kernel_size  #成立，往下运行，不成立抛出异常
        x = self.conv(x)   ##[1,5,8,57],[8,57,57]一个有5个通道，每个通道包含2种人[5*2]，每种人有8[5*2*8]个时刻，一种人有57个[5*2*8*57]！！
        # 一共有4个维度：57个人的某一时刻共57*8个数据，与他人57个的某一时刻57*8个数据，！！
        # 第一矩阵是57个人在所有时刻的与57个人在第一个时刻的距离，因此是8*57*57相乘降维到8*57！！
        # 在第二个大的维度上包含某一个通道57个人在8个时刻与57个人在8个时刻的距离！！！
        # 在第三个大的维度上包含所有通道的所有距离
        x = torch.einsum('nctv,tvw->nctw', (x, A))#对应XA的公式，傅里叶变换频域的卷积等于时域的相乘[1,5,8,57],[8,57,57]————>[1,5,8,57]
                                                   # 只关注后两个维度[8,57]和[57,57]
        #A是utils.py中的seq_to_graph函数，将A转化为（A+I）乘以D逆     A是邻接矩阵

        return x.contiguous(), A
    

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data    输入序列数据中的通道数
        out_channels (int): Number of channels produced by the convolution  卷积产生的通道数
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel    时间卷积核和图卷积核的大小
        stride (int, optional): Stride of the temporal convolution. Default: 1              时间卷积的步幅。默认值：1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``   如果为``True''，则应用残差机制。默认值：``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format    输入图形序列
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format              输入图邻接矩阵
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,                                               N是batch_size
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,    `K`是空间内核大小，
            :math:`T_{in}/T_{out}` is a length of input/output sequence,              是输入/输出序列的长度，
            :math:`V` is the number of graph nodes.                                V'是图形节点的数量
    """

    # 输入是2，输出是5，kernel_size=[3,8]
    #st-gcn 中kernel_size卷积核是时间卷积核和图卷积核的大小
    #输入的图序列 是（batch_size,2，8，V）V是图节点的数量（相当于行人数量）
    #输入的图的邻接矩阵是 （K,V,V）K是空间内核的大小,也是图卷积核的大小（参数设定就是obs_len） (8,V,V)
    #输出
    def __init__(self,
                 in_channels,      #2
                 out_channels,      #5
                 kernel_size,          # [3,8]
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1  #3%2==1
        padding = ((kernel_size[0] - 1) // 2, 0)     #(1,0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])       #输入是2，输出是5 ,kernel_size=8  图卷积核为kernel_size[1]=8
        #开始的in_channel是2，经过卷积得到 128,5,8,57  *  8,57,57  =————> 128,5,8,57  求了个距离和缩了一维

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,      #5
                out_channels,      #5
                (kernel_size[0], 1),  #（3，1）  n=(w-f+2p)+1 = (8-1(3-1)+2 -1)+1 = 8  (57-1(1-1)+2*0 -1)+1 = 57不变---所以经过tcn输出（128，5，8，57）
                (stride, 1),          #（1，1）                                #维度没有改变
                padding,                #(1,0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:   #默认是true,应用残差机制  第一层st-gcn时候，就走第三个条件，如果再多几个gcn,就执行第二个条件
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,     #2          输入为：[128,2,8,57]--经过（1，1）kernel,得到 [128,5, 8-1+1=8，57-1+1=57]--维度中只有输出通道改变了
                    out_channels,    #5
                    kernel_size=1,
                    stride=(stride, 1)),  #[1,1]
                nn.BatchNorm2d(out_channels),
            )     #[8,57]————>  n=8-1+1=8 [8,57]

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        # [128,2,8,57],[8,57,57]
        res = self.residual(x)      #第一个tcn的时候不经过残差，第二个经过才是这个[128,2,8,57]————>[128,5,8,57]，因为输出通道数不一样了
        x, A = self.gcn(x, A)      # 得到的结果 128,5,8,57  *  8,57,57  =————> 128,5,8,57  A没有变化
                                    #第一层tcn没有和残差相加，之后的每一层tcn后面都加上残差连接了
        x = self.tcn(x) + res      #tcn[128,5,8,57]相加  （这两个维度一样，tcn+残差机制）维度还是 128,5,8,57
        
        if not self.use_mdn:
            x = self.prelu(x)  #激活层

        return x, A

class social_stgcnn(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(social_stgcnn,self).__init__()
        self.n_stgcnn= n_stgcnn    #1
        self.n_txpcnn = n_txpcnn   #5
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))     #输入是2，输出是5，kernel_size=(3,8)----128,5,8,57
        for j in range(1,self.n_stgcnn):  #0
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))     #n=5-3+2 +1=5  || ---128，8，5，57-------->128,12,5,57
        for j in range(1,self.n_txpcnn):   #1,2,3,4
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))  #n=w-3+2+1不变
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)       # 128*12*5*57 ----> 128,12,5,57
            
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):  #0,1,2,3,4
            self.prelus.append(nn.PReLU())   #默认0.25


        
    def forward(self,v,a):

        for k in range(self.n_stgcnn):  #1 从0到1不包含1   k=0
            v,a = self.st_gcns[k](v,a)  # 128,5,8,57    A：8,57,57
            
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])     #128,5,8,57————>128,8,5,57
        
        v = self.prelus[0](self.tpcnns[0](v))      #128,12,5,57  ||  ---128，8，5，57-------->128,12,5,57

        for k in range(1,self.n_txpcnn-1):      #1,2,3,4
            v =  self.prelus[k](self.tpcnns[k](v)) + v     #128*12*5*57 4个残差tcn
            
        v = self.tpcnn_ouput(v)                            #128*12*5*57 || 128*12*5*57 ----> 128,12,5,57
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])  #128*5*12*57
        
        
        return v,a     ##128*5*12*57   A：8,57,57
