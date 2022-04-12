# @Time: 2022.4.7 19:31
# @Author: Bolun Wu

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNKernel(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(GCNKernel, self).__init__(aggr=aggr)
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # step1: add self-loops to A
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # step2: linear transformation
        x = self.lin(x)
        # step3: compute normalization
        ## row in source node (x_j), col is target node (x_i: center node when aggregating)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype) # in degree
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] # norm term for each edge
        
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def __repr__(self):
        return f'GCNKernel_{self.aggr}: ({self.lin.in_features}, {self.lin.out_features})'


class GCNKernel_noNorm(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(GCNKernel_noNorm, self).__init__(aggr=aggr)
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # add self-loops to A
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return x_j
    
    def __repr__(self):
        return f'GCNKernel_noNorm_{self.aggr}: ({self.lin.in_features}, {self.lin.out_features})'


class GraphSAGEKernel(MessagePassing):
    """Ref: https://arxiv.org/pdf/1706.02216.pdf
    """
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super(GraphSAGEKernel, self).__init__(aggr=aggr)
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # add self-loops to A
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)
        return self.lin(out)
    
    def message(self, x_j):
        return x_j

    def __repr__(self):
        return f'GraphSAGEKernel_{self.aggr}: ({self.lin.in_features}, {self.lin.out_features})'
        

class GINKernel(MessagePassing):
    def __init__(self, mlp: nn.Module, eps=0., train_eps=False, aggr='add'):
        super(GINKernel, self).__init__(aggr=aggr)
        self.mlp = mlp
        if train_eps: self.eps = nn.Parameter(torch.Tensor([eps]))
        else: self.register_buffer('eps', torch.Tensor([eps]))
        
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        out = (1 + self.eps) * x + out
        return self.mlp(out)
    
    def message(self, x_j):
        # this is the same as default `message`
        return x_j

    def __repr__(self):
        return f'GINKernel: {self.mlp.__repr__()}'



if __name__ == '__main__':
    model = GCNKernel(in_channels=4, out_channels=4)
    x = torch.randn([3, 8])
    x = torch.tensor([[1,1,1,1],
                      [2,2,2,2],
                      [3,3,3,3]], dtype=torch.float)
    edge_index = torch.tensor([[0, 0],
                               [1, 2]], dtype=torch.long)
    
    model(x, edge_index)
    