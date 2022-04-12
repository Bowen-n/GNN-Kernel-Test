# @Time: 2022.4.6 22:14
# @Author: Bolun Wu

import argparse
import json
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torchmetrics.functional import accuracy, f1_score

import wandb
from dataset import get_planetoid_dataset
from model import GCN, GIN, GCN_noNorm, GraphSAGE
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BaseTrainer(object):
    def __init__(self, dataset, model, name='model', log_interval=50):
        self.dataset = dataset
        self.name = name
        self.model = model.to(device)
        self.log_interval = log_interval
        
    def train(self, n_epochs, lr, save_dir):
        self.n_epochs, self.lr, self.save_dir = n_epochs, lr, save_dir

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'{self.name}.pt'))


class NodeClassificationTrainer(BaseTrainer):
    def __init__(self, dataset, model, name='node_model'):
        super(NodeClassificationTrainer, self).__init__(dataset, model, name)
        
    def train(self, n_epochs=100, lr=1e-3, save_dir='result'):
        # meta
        self.n_epochs, self.lr, self.save_dir = n_epochs, lr, save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1e6])
        criterion = nn.CrossEntropyLoss()
        
        # best logger
        self.best_eval_acc = 0.0
        self.best_perform = None
        # writer = SummaryWriter(log_dir=self.save_dir)
        
        # training loop
        self.epoch = 1
        tik = time.time()
        for epoch in range(1, self.n_epochs+1):
            self.epoch = epoch
            data = self.dataset[0].to(device)
            
            # train
            self.model.train()
            ## forward
            out = self.model(data)
            ## backward
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
        
            # eval
            self.model.eval()
            with torch.no_grad():
                logits = self.model(data)
            
            perform = {'epoch': f'{self.epoch}(start from 1)','train': {}, 'val': {}, 'test': {}}
            for key in ['train', 'val', 'test']:
                mask = data[f'{key}_mask']
                loss = criterion(out[mask], data.y[mask]).item()
                
                acc = accuracy(logits[mask], data.y[mask]).item()
                macro_f1 = f1_score(logits[mask], data.y[mask], average='macro', num_classes=self.dataset.num_classes).item()
                
                perform[key]['loss'] = loss
                perform[key]['acc'] = acc
                perform[key]['macro_f1'] = macro_f1
                
                # tensorboard log
                # writer.add_scalar(f'Loss/{key}', loss, self.epoch-1)
                # writer.add_scalar(f'Accuracy/{key}', acc, self.epoch-1)
                # writer.add_scalar(f'F1/{key}', macro_f1, self.epoch-1)
                
                # wandb log
                wandb.log({f'Loss/{key}': loss}, commit=False)
                wandb.log({f'Accuracy/{key}': acc}, commit=False)
                wandb.log({f'Macro F1/{key}': macro_f1}, commit=False)

            wandb.log({'lr': scheduler.get_last_lr()[0]}, commit=True)
            # writer.add_scalar('lr', scheduler.get_last_lr()[0], self.epoch-1)

            # save
            if perform['val']['acc'] > self.best_eval_acc:
                self.best_eval_acc = perform['val']['acc']
                self.best_perform = perform
                wandb.run.summary['best_val_acc'] = perform['val']['acc']
                wandb.run.summary['best_test_acc'] = perform['test']['acc']
                wandb.run.summary['best_val_f1'] = perform['val']['macro_f1']
                wandb.run.summary['best_test_f1'] = perform['test']['macro_f1']
                self.save_model()

            # verbose during training
            if self.epoch % self.log_interval == 0:
                duration = time.time() - tik
                time_per_epoch = duration / self.epoch
                print('Epoch {}, time {:.4f}/ep, train loss {:.4f}, train acc {:.4f}, val acc {:.4f}, test acc {:.4f}.'.format(
                    self.epoch, time_per_epoch, perform['train']['loss'], perform['train']['acc'],
                    perform['val']['acc'], perform['test']['acc']))
                
            scheduler.step()
        
        # writer.close()
        
        # verbose the training result
        print('Best epoch {}, eval acc {:.4f}, test acc {:.4f}.'.format(
            self.best_perform['epoch'], self.best_eval_acc, self.best_perform['test']['acc']))
    
        # save best performance
        with open(os.path.join(self.save_dir, 'measure.json'), 'w') as f:
            json.dump(self.best_perform, f, indent=1)
            
        # save model structures
        with open(os.path.join(self.save_dir, 'model.txt'), 'w') as f:
            f.write(f'{self.model.__repr__()}')



if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='PubMed', choices=('Cora', 'CiteSeer', 'PubMed'),
                        help='dataset name')
    parser.add_argument('-m', '--model', type=str, default='gcn', choices=('gin', 'gcn', 'gcn_nonorm', 'graphsage'),
                        help='gnn model name')
    parser.add_argument('--hidden', type=int, default=256,
                        help='hidden dimension of the model')
    parser.add_argument('-l', '--num-layers', type=int, default=3,
                        help='number of layers in model')
    parser.add_argument('-c', '--conv-layers', type=int, default=3,
                        help='number of graph convolutional layers')
    parser.add_argument('--bn', action='store_true', default=False,
                        help='whether use BN in gcn model')
    parser.add_argument('--aggr', type=str, default=None, choices=('mean', 'add', 'max'), 
                        help='aggregation method that gnn kernel to be used.')
    parser.add_argument('-s', '--save-name', type=str, default=None,
                        help='model and tensorboard saving directory name')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for the whole program')
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)

    # result dir
    os.makedirs(result_path, exist_ok=True)
    if args.save_name is None:
        args.save_name = f'{args.dataset}_{args.model}_{args.hidden}_{args.num_layers}_{args.conv_layers}_{args.aggr}'
        if args.bn: args.save_name += '_bn'

    wandb.init(project='gnn_kernels',
               entity='bowen-wu',
               name=args.save_name)
    wandb.config.update(args)
    
    # dataset
    dataset = get_planetoid_dataset(args.dataset)
    
    # model
    if args.bn: norm = nn.BatchNorm1d(args.hidden)
    else: norm = None
        
    if args.model == 'gcn':
        model = GCN(in_channels=dataset[0].x.shape[1], 
                    hidden_channels=args.hidden, 
                    num_layers=args.num_layers,
                    conv_layers=args.conv_layers,
                    out_channels=dataset.num_classes, 
                    norm=norm,
                    aggr=args.aggr)
    elif args.model == 'gin':
        model = GIN(in_channels=dataset[0].x.shape[1], 
                    hidden_channels=args.hidden, 
                    num_layers=args.num_layers,
                    conv_layers=args.conv_layers,
                    out_channels=dataset.num_classes, 
                    norm=norm,
                    aggr=args.aggr)
    elif args.model == 'graphsage':
        model = GraphSAGE(in_channels=dataset[0].x.shape[1],
                          hidden_channels=args.hidden,
                          num_layers=args.num_layers,
                          conv_layers=args.conv_layers,
                          out_channels=dataset.num_classes, 
                          norm=norm,
                          aggr=args.aggr)
    elif args.model == 'gcn_nonorm':
        model = GCN_noNorm(in_channels=dataset[0].x.shape[1],
                           hidden_channels=args.hidden,
                           num_layers=args.num_layers,
                           conv_layers=args.conv_layers,
                           out_channels=dataset.num_classes,
                           norm=norm,
                           aggr=args.aggr)
    
    # trainer
    trainer = NodeClassificationTrainer(dataset, model, name=args.model)
    # train
    trainer.train(n_epochs=600,
                  lr=1e-3,
                  save_dir=os.path.join(result_path, args.save_name))
    
    # save args
    with open(os.path.join(result_path, args.save_name, 'args.txt'), 'w') as f:
        f.write(f'{args}')

