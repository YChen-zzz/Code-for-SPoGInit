import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import math
from torch_geometric.utils import to_networkx
from torch_geometric.nn import JumpingKnowledge, APPNP,GATConv
import pandas as pd
import torch_geometric
from torch_geometric.nn.inits import zeros,ones
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul
import scipy.sparse
import numpy as np
import math

from models.GCN_layer import GCNConv


    
class RezeroGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,bn = False,initialization = "glorot",activation = "Tanh"):
        super(RezeroGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.alpha = torch.nn.Parameter(torch.zeros(num_layers))
        self.beta = torch.nn.Parameter(torch.ones(num_layers))
        self.inProj = torch.nn.Linear(in_channels, hidden_channels)
        self.bn = bn
        if self.bn==True:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels,affine=False))
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,depth = num_layers,initialization = initialization,cached=True))
            if self.bn==True:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels,affine=False))
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.activation = activation
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.bn==True:
            for bn in self.bns:
                bn.reset_parameters()
        zeros(self.alpha)
        self.linear.reset_parameters()
        self.inProj.reset_parameters()
        ones(self.beta)

    
    def print_x(self, x, edge_index):
        x = self.inProj(x)
        inp = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.bn == True:
                x = self.bns[i](x)
            if self.activation == "Tanh":
                x = torch.tanh(x)
            elif self.activation == "ReLU":
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.beta[i]*inp + self.alpha[i]*x
            inp=x
        x = self.linear(x)
        return x
    
    def forward(self, x, edge_index):
        x = self.print_x(x, edge_index)
        return F.log_softmax(x, dim=1)

    def print_all_x(self,x,edge_index):
        out = []
        x = self.inProj(x)
        inp = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.bn == True:
                x = self.bns[i](x)
            if self.activation == "Tanh":
                x = torch.tanh(x)
            elif self.activation == "ReLU":
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.beta[i]*inp + self.alpha[i]*x
            out+=[x]
            inp=x
        x = self.linear(x)
        return out

class MyResGCN(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels,out_channels, num_layers, dropout,bn = False,initialization = "conventional",activation = "Tanh"):
        super(MyResGCN,self).__init__()
        self.inProj = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.outProj = torch.nn.Linear(hidden_channels, out_channels)
        self.activation = activation
        self.num_layers=num_layers
        self.bn = bn
        if self.bn==True:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels,affine=False))
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,depth = num_layers,initialization = initialization,cached=True))
            if self.bn==True:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels,affine=False))
        #self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))
        self.dropout = dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.bn==True:
            for bn in self.bns:
                bn.reset_parameters()
        self.outProj.reset_parameters()
        self.inProj.reset_parameters()
    
    def print_x(self,x, edge_index):
        x = self.inProj(x)
        last = x
        for i in range(self.num_layers):
            x = self.convs[i](x,edge_index)
            ## bn
            if self.activation == "ReLU":
                x = F.relu(x)
            elif self.activation == "Tanh":
                x = torch.tanh(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + last
            if self.bn == True:
                x = self.bns[i](x)
            last = x
        x = self.outProj(x)
        return x
    
    def forward(self,x, edge_index):
        x = self.print_x(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def print_all_x(self,x, edge_index):
        out = []
        x = self.inProj(x)
        last = x
        for i in range(self.num_layers):
            x = self.convs[i](x,edge_index)
            if self.activation == "ReLU":
                x = F.relu(x)
            elif self.activation == "Tanh":
                x = torch.tanh(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + last
            if self.bn == True:
                x = self.bns[i](x)
            last = x
            out.append(x)
        return out


class gatResGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,bn = False,initialization = "glorot",activation = "Tanh"):
        super(gatResGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.alpha = torch.nn.Parameter(torch.ones(num_layers))
        self.beta = torch.nn.Parameter(torch.ones(num_layers))
        self.inProj = torch.nn.Linear(in_channels, hidden_channels)
        self.bn = bn
        if self.bn==True:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels,affine=False))
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,depth = num_layers,initialization = initialization,cached=True))
            if self.bn==True:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels,affine=False))
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.activation = activation
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.bn==True:
            for bn in self.bns:
                bn.reset_parameters()
        ones(self.alpha)
        self.linear.reset_parameters()
        self.inProj.reset_parameters()
        ones(self.beta)

    
    def print_x(self, x, edge_index):
        x = self.inProj(x)
        inp = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.bn == True:
                x = self.bns[i](x)
            if self.activation == "Tanh":
                x = torch.tanh(x)
            elif self.activation == "ReLU":
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.beta[i]*inp + self.alpha[i]*x
            inp=x
        x = self.linear(x)
        return x
    
    def forward(self, x, edge_index):
        x = self.print_x(x, edge_index)
        return F.log_softmax(x, dim=1)

    def print_all_x(self,x,edge_index):
        out = []
        x = self.inProj(x)
        inp = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.bn == True:
                x = self.bns[i](x)
            if self.activation == "Tanh":
                x = torch.tanh(x)
            elif self.activation == "ReLU":
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.beta[i]*inp + self.alpha[i]*x
            out+=[x]
            inp=x
        x = self.linear(x)
        return out
    
class MyGCN(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels,out_channels, num_layers, dropout,bn = False,initialization = "conventional",activation = "Tanh"):
        super(MyGCN,self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.activation = activation
        self.num_layers=num_layers
        self.bn = bn
        self.convs.append(GCNConv(in_channels, hidden_channels,depth = num_layers,initialization = initialization,cached=False))
        if self.bn==True:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels,affine=False))
        for i in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,depth = num_layers,initialization = initialization,cached=False))
            if self.bn==True:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels,affine=False))
        self.convs.append(GCNConv( hidden_channels,out_channels, depth = num_layers,initialization = initialization,cached=False))
        #self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))
        self.dropout = dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.bn==True:
            for bn in self.bns:
                bn.reset_parameters()
    
    def print_x(self,x, edge_index):
        for i in range(self.num_layers-1):
            x = self.convs[i](x,edge_index)
            #if self.bn == True:
            #    x = self.bns[i](x)
            if self.activation == "ReLU":
                x = F.relu(x)
            elif self.activation == "Tanh":
                x = torch.tanh(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.bn == True:
                x = self.bns[i](x)
        x = self.convs[-1](x,edge_index)
        return x
    
    def forward(self,x, edge_index):
        x = self.print_x(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def print_all_x(self,x, edge_index):
        out = []
        for i in range(self.num_layers-1):
            x = self.convs[i](x,edge_index)
            #if self.bn == True:
            #    x = self.bns[i](x)
            if self.activation == "ReLU":
                x = F.relu(x)
            elif self.activation == "Tanh":
                x = torch.tanh(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.bn == True:
                x = self.bns[i](x)
            out.append(x) # 发现cpu问题是显著的增加spog的显存
        x = self.convs[-1](x,edge_index)
        return out
    
    def print_all_x_hybrid(self, x, edge_index):
        out = []
        for i in range(self.num_layers-1):
            x = self.convs[i](x,edge_index)
            #if self.bn == True:
            #    x = self.bns[i](x)
            if self.activation == "ReLU":
                x = F.relu(x)
            elif self.activation == "Tanh":
                x = torch.tanh(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.bn == True:
                x = self.bns[i](x)
            if i in [0, self.num_layers-2]:  # 只保存首尾层的完整tensor
                out.append(x)
            else:
                with torch.no_grad():
                    norm = torch.norm(x).item()
                    out.append(norm)
        x = self.convs[-1](x,edge_index)
        return out


    

