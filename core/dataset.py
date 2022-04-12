# @Time: 2022.4.6 22:14
# @Author: Bolun Wu

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from utils import *


def get_planetoid_dataset(name):
    transform = T.Compose([T.NormalizeFeatures()])
    dataset = Planetoid(root=data_path, split='full', name=name, transform=transform)
    return dataset


if __name__ == '__main__':
    
    # Cora, CiteSeer, PubMed
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = get_planetoid_dataset(name=name)
        train_size = sum(dataset.data.train_mask)
        val_size = sum(dataset.data.val_mask)
        test_size = sum(dataset.data.test_mask)
        
        print(f'{dataset.name}: {train_size}, {val_size}, {test_size}.')
        print(dataset)
