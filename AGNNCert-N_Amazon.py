# -*- coding: utf-8 -*-

from gnn import NodeGCN,NodeGAT,NodeGSAGE

from node_hash import HashAgent, RobustAmazonNodeClassifier
import torch 
import numpy as np
import os 
import random

from ogb.nodeproppred import PygNodePropPredDataset

datasets = PygNodePropPredDataset(name = "ogbn-products") 

split_idx = datasets.get_idx_split()
graph = datasets[0] 


device =  "cuda" if torch.cuda.is_available() else "cpu"

paper = "GCN"
dataset = "Products"
train_args = {
        "dataset": dataset,
        "paper": paper,
        "lr" : 0.001,
        "epochs" : 2000,
        "clip_max" : 2.0,
        "batch_size": 64,
        "early_stopping": 100,
        "seed" : 42,
        "eval_enabled" : True
    }
inference_args = {
        "strategy": "adaptive",
        "route_confidence": 0.85,
        "early_stop_ratio": 0.5,
    }
print(train_args["dataset"])
print(train_args["seed"])
from utils import evaluate, store_checkpoint, load_best_model, train_model, resolve_checkpoint_path
x = torch.tensor(graph.x)
edge_index = torch.tensor(graph.edge_index)
labels = torch.tensor(graph.y)
num_x = x.shape[1]
num_labels = 47
model = NodeGCN(num_x, num_labels,hidden_size=64).to(device)


train_idx = []
valid_idx = []
test_idx=[]
for c in range(num_labels):
    idx = (torch.tensor(labels) == c).nonzero(as_tuple=False).view(-1)

    train_i = idx[:int(0.3*len(idx))]
    val_i = idx[int(0.3*len(idx)):int(0.5*len(idx))]
    test_i = idx[int(0.5*len(idx)):-1]
    train_idx.extend(train_i)
    valid_idx.extend(val_i)
    test_idx.extend(test_i)
        
T=100
hasher = HashAgent(h="md5",T=T)
r_model = RobustAmazonNodeClassifier(model,hasher,edge_index, x, labels, train_idx, valid_idx, test_idx,num_labels)

path = resolve_checkpoint_path("./checkpoints", "robust_n", paper, dataset, T)
if path is not None:
    r_model.load_model(path)
else:
    r_model.train(train_args)
    path = resolve_checkpoint_path("./checkpoints", "robust_n", paper, dataset, T)
    if path is not None:
        r_model.load_model(path)
     

test_labels = labels[test_idx]
out_test,M,inference_details = r_model.predict(test_idx, return_details=True, **inference_args)
test_preds = out_test.argmax(dim=1)
test_acc = evaluate(out_test, test_labels)
print(test_acc)
print({
    "strategy": inference_details["strategy"],
    "direct_route_ratio": inference_details.get("direct_route_ratio", 0.0),
    "early_stop_ratio_realized": inference_details.get("early_stop_ratio_realized", 0.0),
    "avg_subgraphs_used": inference_details.get("avg_subgraphs_used", 0.0),
})


count = {}
for i in range(len(M)):
    c = "{}".format(int(M[i])) 
    if test_preds[i]!=test_labels[i]:
        continue
    if not c in count:
        count[c]=1 
    else:
        count[c]+=1 
sorted_count = dict(sorted(count.items(),key=lambda item:int(item[0])))
print(sorted_count)
print(len(M))
