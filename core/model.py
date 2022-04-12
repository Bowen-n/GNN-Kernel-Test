# @Time: 2022.4.9 15:55
# @Author: Bolun Wu

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MLP

from kernel import GCNKernel, GCNKernel_noNorm, GINKernel, GraphSAGEKernel


class BasicGNN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels,
                 num_layers,
                 conv_layers,
                 out_channels,
                 dropout=0.5,
                 norm=None,
                 aggr=None):
        super(BasicGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.conv_layers = conv_layers
        self.dropout = dropout
        self.aggr = aggr
        
        assert 1 <= conv_layers <= num_layers

        # norm layers
        self.norms = None
        if norm is not None:
            self.norms = nn.ModuleList()
            for _ in range(num_layers-1):
                self.norms.append(copy.deepcopy(norm))
                
        # gcn conv layers
        self.convs = nn.ModuleList()

        for i in range(conv_layers):
            if i == 0:
                self.convs.append(self.init_conv(in_channels, hidden_channels))
            elif i == num_layers-1:
                self.convs.append(self.init_conv(hidden_channels, out_channels))
            else:
                self.convs.append(self.init_conv(hidden_channels, hidden_channels))

        # linear layers
        if conv_layers < num_layers:
            self.lins = nn.ModuleList()
            for i in range(conv_layers, num_layers):
                if i == num_layers-1:
                    self.lins.append(nn.Linear(hidden_channels, out_channels))
                else:
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.conv_layers):
            x = self.convs[i](x, edge_index)
            if i == self.conv_layers - 1:
                break

            if self.norms is not None:
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.conv_layers < self.num_layers:
            lin_layers = self.num_layers - self.conv_layers
            for i in range(lin_layers):
                x = self.lins[i](x)
                if i == lin_layers - 1:
                    break
                
                if self.norms is not None:
                    x = self.norms[i+self.conv_layers](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
    
    def init_conv(self):
        raise NotImplementedError

    
class GCN(BasicGNN):        
    def init_conv(self, in_channels, out_channels):
        if self.aggr is None: self.aggr = 'add'
        return GCNKernel(in_channels, out_channels, aggr=self.aggr)


class GCN_noNorm(BasicGNN):
    def init_conv(self, in_channels, out_channels):
        if self.aggr is None: self.aggr = 'add'
        return GCNKernel_noNorm(in_channels, out_channels, aggr=self.aggr)


class GIN(BasicGNN):
    def init_conv(self, in_channels, out_channels):
        if self.aggr is None: self.aggr = 'add'
        # if self.norms is not None: self.norms = None
        mlp = MLP([in_channels, out_channels, out_channels], batch_norm=True)
        return GINKernel(mlp, aggr=self.aggr)


class GraphSAGE(BasicGNN):        
    def init_conv(self, in_channels, out_channels):
        if self.aggr is None: self.aggr = 'mean'
        return GraphSAGEKernel(in_channels, out_channels, aggr=self.aggr)


if __name__ == '__main__':
    model = GCN(in_channels=500, 
                hidden_channels=256, 
                num_layers=3, 
                conv_layers=3, 
                out_channels=3, 
                norm=nn.BatchNorm1d(256))
    
    print(model.__repr__())
