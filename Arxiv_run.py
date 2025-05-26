import argparse

import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
#from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import time
from logger import Logger
from torch_geometric import seed_everything

from typing import Optional, Tuple

from torch_geometric.nn import inits
import math
import numpy as np
from arxiv_year import get_idx_split
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from models.model import *
from spoginit import SpogInit

# torch.set_default_dtype(torch.float64)

def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model.forward(data.x, data.adj_t)[train_idx]
    #print(f"out : {out[:20]}")
    #print(f"predict: {data.y.squeeze(1)[train_idx][:20]}")
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

def train_year(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model.forward(data.x, data.adj_t)[train_idx]
    #print(f"out : {out[:20]}")
    #print(f"predict: {data.y[train_idx][:20]}")
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()



@torch.no_grad()
def test(model, data, split_idx, evaluator,train_idx):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[train_idx],
        'y_pred': y_pred[train_idx],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def test_year(model, data):
    model.eval()

    out = model(data.x, data.adj_t)
    #y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = int((torch.argmax(out[data.train_mask], dim=1) == data.y[data.train_mask]).float().sum()) / len(data.train_mask)
    valid_acc = int((torch.argmax(out[data.val_mask], dim=1) == data.y[data.val_mask]).float().sum()) / len(data.val_mask)
    test_acc = int((torch.argmax(out[data.test_mask], dim=1) == data.y[data.test_mask]).float().sum()) / len(data.test_mask)
    return train_acc, valid_acc, test_acc

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument("--model", default="GCN", help="model.")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_spog', default = False,action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument("--activation", default="Tanh", help="{ReLU and Tanh }.")
    parser.add_argument("--initialization", default="glorot", help="{glorot and conventional}.")
    parser.add_argument('--bn', action='store_true',
                    default=False, help='use bn in model')
    parser.add_argument("--data", default="ogbn-arxiv", help="dataset")
    parser.add_argument('--missing', type=float, default=0,
                    help='missing feature probability')
    parser.add_argument('--log', action='store_true', default=False, 
                    help='save training logs every 10 steps')
    parser.add_argument('--save_path', type=str, default="", 
                    help='files_to save results')
    parser.add_argument('--data_path', type=str, default="", 
                    help='files_to save results')
    
    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.data == "ogbn-arxiv":
        data_path = args.data_path

        dataset = PygNodePropPredDataset(name=f'{args.data}',root=data_path,transform=T.ToSparseTensor())
        
    
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        data = data.to(device)

        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        data.train_mask = train_idx.to(device)
        output_dim = dataset.num_classes

    elif args.data == "arxiv-year":
        data_path = args.data_path
        ## save from https://github.com/CUAI/Non-Homophily-Large-Scale/tree/master
        data = torch.load(f"{data_path}/arxiv_year.pt").to(device)
    
        split = np.load(f"./data/arxiv-year-splits.npy",allow_pickle=True)

        transform=T.ToSparseTensor()
        data = transform(data)
        data.train_mask = torch.tensor(split[0]["train"]).to(device)
        data.val_mask = torch.tensor(split[0]["valid"]).to(device)
        data.test_mask = torch.tensor(split[0]["test"]).to(device)
        
        train_mask = data.train_mask.to(device)
        test_mask = data.test_mask.to(device)
        val_mask = data.val_mask.to(device)
        input_dim = data.x.shape[1]
        train_idx = train_mask
        output_dim = 5

    
    train_idx = train_idx.to(device)
    seed_everything(args.seed)
    if args.model == "ResGCN":
        print("ResGCN")
        model = MyResGCN(data.num_features, args.hidden_channels,
                    output_dim, args.num_layers,
                    args.dropout,bn=args.bn,initialization = args.initialization, activation = args.activation).to(device)
    elif args.model == "gatResGCN":
        print("gatResGCN")
        model = gatResGCN(data.num_features, args.hidden_channels,
                    output_dim, args.num_layers,
                    args.dropout,bn=args.bn,initialization = args.initialization, activation = args.activation).to(device)
    elif args.model == "GCN": 
        print("GCN")
        model = MyGCN(in_channels = data.num_features,hidden_channels = args.hidden_channels,out_channels =output_dim,num_layers = args.num_layers, dropout = args.dropout, bn=args.bn,initialization = args.initialization,activation = args.activation).to(device)
    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))
    #evaluator = Evaluator(name="ogbn-arxiv")
    if args.data == "minesweeper":
        from evalutor_roc import Evaluator
        evaluator = Evaluator(name="minesweeper")
    else:
        from ogb.nodeproppred import Evaluator
        evaluator = Evaluator(name="ogbn-arxiv")
    logger = Logger(args.runs, args)
    
    acc_list = []
    
    # Initialize log storage if logging is enabled
    if args.log:
        training_logs = []
        log_columns = ['run', 'epoch', 'loss', 'train_acc', 'valid_acc', 'test_acc']
    
    for run in range(args.runs):
        # Add timer start
        start_time = time.time()
        
        if args.data == "arxiv-year":
            print(f"load {run}-th split")
            data.train_mask = torch.tensor(split[run]["train"]).to(device)
            data.val_mask = torch.tensor(split[run]["valid"]).to(device)
            data.test_mask = torch.tensor(split[run]["test"]).to(device)
            train_mask = data.train_mask.to(device)
            test_mask = data.test_mask.to(device)
            val_mask = data.val_mask.to(device)
            train_idx = train_mask



        model.reset_parameters()
        
        if args.use_spog==True:
            ## with bn but spog with bn
            #model.bn=False
            Spog = SpogInit(model,data.adj_t,train_idx,data.x.shape[0],data.x.shape[1],output_dim, device, "divide_stable")
            if "gatRes" in args.model:
                Spog.zerosingle_initialization_gate(data.x, data.y, lr=0.2,decay=1)
            elif "Res" in args.model:
                Spog.zerosingle_initialization(data.x, data.y,lr=0.2,decay=1)
            else:
                if args.data == "arxiv-year":
                    print("no generate")
                    Spog = SpogInit(model,data.adj_t,train_idx,data.x.shape[0],data.x.shape[1],output_dim, device, "divide_old")
                    Spog.zeroincrease_initialization(data.x, data.y,w2=10,max_pati=10,steps=500, lr = 0.07, generate_data = True)
                else:
                    Spog = SpogInit(model,data.adj_t,train_idx,data.x.shape[0],data.x.shape[1],output_dim, device, "divide_old")
                    Spog.zeroincrease_initialization(data.x, data.y,w2=10,max_pati=20,steps=40, lr = 0.05,generate_data = True)

        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9,0.9995),weight_decay = args.wd)

        #print(model.outProj.weight)
        
        bad_counter = 0
        best_val = 0
        final_test_acc = 0
        
        for epoch in range(1, 1 + args.epochs):
            if args.data == "ogbn-arxiv":
                loss = train(model, data, train_idx, optimizer)
                result = test(model, data, split_idx, evaluator,train_idx)
            else:
                loss = train_year(model, data, train_idx, optimizer)
                result = test_year(model, data)
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            
            # Store training logs if enabled and every 10 steps
            if args.log and epoch % 10 == 0:
                training_logs.append({
                    'run': run + 1,
                    'epoch': epoch,
                    'loss': loss,
                    'train_acc': train_acc,
                    'valid_acc': valid_acc,
                    'test_acc': test_acc
                })
                
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
                
            if valid_acc > best_val:
                best_val = valid_acc
                final_test_acc = test_acc
                bad_counter = 0
            else:
                bad_counter += 1

            #if bad_counter == 200:
            #    break
        acc_list.append(final_test_acc*100)
        
        print(f'Run {run+1}: Accuracy = {acc_list[-1]:.2f}%')
        logger.print_statistics(run)
    best_train,best_val, best_test =logger.print_statistics()
    print("best train ", best_train)

    ## get the save file name
    if args.bn==True:
        bn_str = "BN"
        if args.model == "mixhop":
            bn_str = "newBN"
    else:
        bn_str = "noBN"
        if args.model == "mixhop":
            bn_str = "BN"
    if args.use_spog == True:
        init_str = "spog"+args.initialization
    else:
        init_str = args.initialization


        
    acc_list=torch.tensor(acc_list)
    print(f'Avg Test: {acc_list.mean():.2f} ± {acc_list.std():.2f}')

    
    missing_str = ""

    if args.lr != 0.005:
        lr_str = "lr"+str(args.lr)
    else:
        lr_str = ""

    if args.data == "ogbn-arxiv":
        name_str =""
    else:
        name_str = "_"+args.data

    seed_str = "seed" + str(args.seed)
    
    filename = f'{args.save_path}{name_str}_{args.model}_{args.activation}_init{init_str}_layer{args.num_layers}_3run{bn_str}{lr_str}drop{str(int(args.dropout*10))}{missing_str}.csv'
    print(f"Saving results to {filename}")
    
    import pandas as pd
    df = pd.DataFrame({
        "best_train": np.array(best_train), 
        "best_val": np.array(best_val), 
        "best_test": np.array(best_test),
    })
    df.to_csv(filename)
    

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))


if __name__ == "__main__":
    main()




