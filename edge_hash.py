# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import sys
sys.path.append("models/")
#from mlp import MLP
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import copy

from utils import evaluate, store_checkpoint, load_best_model, train_model
from sklearn.model_selection import train_test_split
from inference_utils import (
    build_adaptive_details,
    compute_vote_margin,
    confidence_from_logits,
    get_mask_indices,
    should_early_stop_by_remaining_votes,
)

device =  "cuda" if torch.cuda.is_available() else "cpu"



#The hash Agent is defined separately to operate the hash and division
class HashAgent():
    def __init__(self,h="md5",T=30):
        '''
            h: the hash function in "md5","sha1","sha256"
            T: the subset amount
        '''

        super(HashAgent, self).__init__()
        self.T = T
        self.h= h 
        
    #Given an edge with point node u and v, we hash it
    def hash_edge(self,V, u,v):
        hexstring = hex(V*u+v)
        hexstring= hexstring.encode()
        if self.h == "md5":
            hash_device = hashlib.md5()
        elif self.h == "sha1":
            hash_device = hashlib.sha1()
        elif self.h == "sha256":
            hash_device = hashlib.sha256()
        hash_device.update(hexstring)
        I = int(hash_device.hexdigest(),16)%self.T
        
        return I
    
    #Given a graph for node classification, we generate its subgraphs
    def generate_node_subgraphs(self, edge_index, x, y):
        
        subgraphs = []
        
        original = edge_index.detach().cpu() if torch.is_tensor(edge_index) else edge_index
        nodes = range(x.shape[0])
        tensor_device = x.device if torch.is_tensor(x) else None

        V= x.shape[0]
                    
        for i in range(self.T):
            subgraphs.append(Data(
                        x = x,
                        y = y,
                        edge_index = []
                    ))
        #print(V)
        for i in range(len(original[0])):
            
            u = int(original[0,i])
            v = int(original[1,i])
            if u>v:
                I = self.hash_edge(V,v,u)
            else:
                I = self.hash_edge(V,u,v)
            subgraphs[I].edge_index.append([u,v])
            
        new_subgraphs = []
        for i in range(self.T):
            if len(subgraphs[i].edge_index)==0:
                continue
            subgraphs[i].edge_index = torch.tensor(
                subgraphs[i].edge_index,
                dtype=torch.int64,
                device=tensor_device,
            ).transpose(1,0)
            new_subgraphs.append(subgraphs[i])
            
        return new_subgraphs


    #Given a graph for graph classification, we generate its subgraphs
    def generate_graph_subgraphs(self, edge_index, x, y):
        
        subgraphs = []
        
        original = edge_index.detach().cpu() if torch.is_tensor(edge_index) else edge_index
        nodes = range(x.shape[0])

        tensor_device = x.device if torch.is_tensor(x) else None
        zerox = torch.zeros(x[0].size(), device=tensor_device).reshape(1,-1)
        V= x.shape[0]
        mappings=-np.ones((self.T,x.shape[0]))
        cnt=torch.zeros(self.T)
        #Note in the subgraphs for node, isolated nodes should be deleted, so instead we only "keep" non-isolated nodes
        for i in range(self.T):
            subgraphs.append(Data(
                        x = zerox,
                        y = y,
                        edge_index = []
                    ))
            
        for i in range(len(original[0])):
            
            u = int(original[0,i])
            v = int(original[1,i])
            
            if u>v:
                I = self.hash_edge(V,v,u)
            else:
                I = self.hash_edge(V,u,v)
            if mappings[I,u]==-1:
                mappings[I,u]=subgraphs[I].x.shape[0]
                subgraphs[I].x = torch.cat((subgraphs[I].x,x[u].reshape(1,-1)),dim=0)
            if mappings[I,v]==-1:
                mappings[I,v]=subgraphs[I].x.shape[0]
                subgraphs[I].x = torch.cat((subgraphs[I].x,x[v].reshape(1,-1)),dim=0)
            subgraphs[I].edge_index.append([mappings[I,u],mappings[I,v]])
            
        for i in range(self.T):
            if len(subgraphs[i].edge_index)==0:
                subgraphs[i].edge_index=None
                continue
            subgraphs[i].edge_index = torch.tensor(
                subgraphs[i].edge_index,
                dtype=torch.int64,
                device=tensor_device,
            ).transpose(1,0)
            
        return subgraphs

    def generate_amazon_subgraphs(self, edge_index, x, y):
        
        subgraphs = []
        
        original = edge_index.detach().cpu() if torch.is_tensor(edge_index) else edge_index
        nodes = range(x.shape[0])
        tensor_device = x.device if torch.is_tensor(x) else None

        V= x.shape[0]
                    
        for i in range(self.T):
            subgraphs.append(Data(
                        x = x,
                        y = y,
                        edge_index = []
                    ))
            
        for i in range(len(original[0])):
            
            u = int(original[0,i])
            v = int(original[1,i])
            if u>v:
                I = self.hash_edge(V,v,u)
            else:
                I = self.hash_edge(V,u,v)
            subgraphs[I].edge_index.append([u,v])
            
        new_subgraphs = []
        for i in range(self.T):
            if len(subgraphs[i].edge_index)==0:
                continue
            subgraphs[i].edge_index = torch.tensor(
                subgraphs[i].edge_index,
                dtype=torch.int64,
                device=tensor_device,
            ).transpose(1,0)
            new_subgraphs.append(subgraphs[i])
            
        return new_subgraphs

class RobustNodeClassifier():
    def __init__(self,model,Hasher,edge_index, x, y, train_mask, val_mask, test_mask,num_labels):

        super(RobustNodeClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.Hasher = Hasher
        self.edge_index = edge_index.to(self.device)
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.num_labels= num_labels
    def load_model(self,path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def _get_subgraphs(self):
        return self.Hasher.generate_node_subgraphs(self.edge_index, self.x, self.y)

    def predict(
        self,
        mask,
        strategy="adaptive",
        route_confidence=0.85,
        early_stop_ratio=0.5,
        return_details=False,
    ):
        if strategy == "baseline":
            out, M = self.vote(mask)
            details = {
                "strategy": "baseline",
                "total_samples": int(out.shape[0]),
                "total_subgraphs_available": int(len(self._get_subgraphs())),
            }
            if return_details:
                return out, M, details
            return out, M
        if strategy == "adaptive":
            return self.adaptive_vote(
                mask,
                route_confidence=route_confidence,
                early_stop_ratio=early_stop_ratio,
                return_details=return_details,
            )
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    def train(self, train_args ):
        subgraphs = self._get_subgraphs()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_args["lr"])
        criterion = torch.nn.CrossEntropyLoss()
    
        best_val_acc = 0.0
        best_epoch = 0
    
        for epoch in range(0, train_args["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.x, self.edge_index)
            loss = criterion(out[self.train_mask], self.y[self.train_mask])
            for i in range(len(subgraphs)):
                out_sub = self.model(subgraphs[i].x,subgraphs[i].edge_index)
                loss+=criterion(out_sub[self.train_mask], self.y[self.train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=train_args["clip_max"])
            optimizer.step()
            
            train_acc = 0
            test_acc = 0
            val_acc = 0
            with torch.no_grad():
                out_train,_ = self.vote(self.train_mask)
                out_val,_ = self.vote(self.val_mask)
                out_test,_ = self.vote(self.test_mask)
                train_acc = evaluate(out_train, self.y[self.train_mask].detach().cpu())
                val_acc = evaluate(out_val, self.y[self.val_mask].detach().cpu())
                test_acc = evaluate(out_test, self.y[self.test_mask].detach().cpu())
                
            print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")
            if val_acc > best_val_acc: # New best results
                print("Val improved")
                best_val_acc = val_acc
                best_epoch = epoch
                store_checkpoint("robust_e/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.Hasher.T), self.model, train_acc, val_acc, test_acc)

            if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
                break
        
     
    def test(self ):   
        
        out_test,M = self.vote(self.test_mask)
        test_acc = evaluate(out_test, self.y[self.test_mask].detach().cpu())
        return test_acc, M
        
    def vote(self, mask):
        subgraphs = self._get_subgraphs()
        mask_idx_cpu = get_mask_indices(mask)
        mask_idx = mask_idx_cpu.to(self.device)
        V_test = int(mask_idx_cpu.shape[0])
        votes = torch.zeros((V_test,self.num_labels), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            for i in range(len(subgraphs)):
                out_sub = self.model(subgraphs[i].x,subgraphs[i].edge_index)
                preds = out_sub.index_select(0, mask_idx).argmax(dim=1).detach().cpu()
                votes[torch.arange(V_test), preds] += 1

        M = compute_vote_margin(votes)
        return votes, M

    def adaptive_vote(
        self,
        mask,
        route_confidence=0.85,
        early_stop_ratio=0.5,
        return_details=False,
    ):
        mask_idx_cpu = get_mask_indices(mask)
        mask_idx = mask_idx_cpu.to(self.device)
        sample_count = int(mask_idx_cpu.shape[0])
        outputs = torch.zeros((sample_count, self.num_labels), dtype=torch.float32)
        margins = torch.full((sample_count,), -1.0, dtype=torch.float32)
        used_subgraphs = np.zeros(sample_count, dtype=int)
        route_modes = np.array(["direct"] * sample_count, dtype=object)

        self.model.eval()
        with torch.no_grad():
            base_out = self.model(self.x, self.edge_index)
            base_masked = base_out.index_select(0, mask_idx)

        base_confidences, _ = confidence_from_logits(base_masked)
        direct_mask = base_confidences >= float(route_confidence)
        outputs[direct_mask] = base_masked.detach().cpu()[direct_mask]

        low_conf_positions = (~direct_mask).nonzero(as_tuple=False).view(-1)
        subgraphs = self._get_subgraphs()
        total_subgraphs = len(subgraphs)

        if low_conf_positions.numel() > 0 and total_subgraphs > 0:
            low_votes = torch.zeros(
                (int(low_conf_positions.numel()), self.num_labels),
                dtype=torch.float32,
            )
            active = torch.ones(int(low_conf_positions.numel()), dtype=torch.bool)
            selected_mask_idx = mask_idx.index_select(0, low_conf_positions.to(self.device))

            with torch.no_grad():
                for subgraph in subgraphs:
                    if not active.any():
                        break
                    out_sub = self.model(subgraph.x, subgraph.edge_index)
                    active_positions = active.nonzero(as_tuple=False).view(-1)
                    active_mask_idx = selected_mask_idx.index_select(
                        0, active_positions.to(self.device)
                    )
                    preds = (
                        out_sub.index_select(0, active_mask_idx)
                        .argmax(dim=1)
                        .detach()
                        .cpu()
                    )
                    for local_pos, pred in zip(active_positions.tolist(), preds.tolist()):
                        low_votes[local_pos, pred] += 1
                        used_subgraphs[int(low_conf_positions[local_pos])] += 1
                    reached = should_early_stop_by_remaining_votes(
                        low_votes,
                        used_subgraphs[low_conf_positions.numpy()],
                        total_subgraphs,
                    )
                    active = active & (~reached)

            outputs[low_conf_positions] = low_votes
            margins[low_conf_positions] = compute_vote_margin(low_votes)
            low_used = used_subgraphs[low_conf_positions.numpy()]
            route_modes[low_conf_positions.numpy()] = "subgraph_full"
            route_modes[low_conf_positions.numpy()[low_used < total_subgraphs]] = (
                "subgraph_early_stop"
            )
        elif low_conf_positions.numel() > 0:
            outputs[low_conf_positions] = base_masked.detach().cpu()[low_conf_positions]
            route_modes[low_conf_positions.numpy()] = "direct_fallback"

        details = build_adaptive_details(
            strategy="adaptive",
            route_confidence=route_confidence,
            early_stop_ratio=early_stop_ratio,
            total_subgraphs=total_subgraphs,
            base_confidences=base_confidences.numpy(),
            route_modes=route_modes,
            used_subgraphs=used_subgraphs,
        )
        if return_details:
            return outputs, margins, details
        return outputs, margins
    

class RobustGraphClassifier():
    def __init__(self,model,Hasher,graphs,labels,train_mask, val_mask, test_mask,num_labels):

        super(RobustGraphClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.Hasher = Hasher
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_labels= num_labels
        self.graphs=graphs
        self.labels = torch.tensor(labels)
        self.subgraphs=[]
        for i in range(len(graphs)):
            self.subgraphs.append(self.Hasher.generate_graph_subgraphs(graphs[i].edge_index,graphs[i].x,graphs[i].y))
            
    def load_model(self,path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def _get_valid_subgraphs(self, graph_idx):
        return [
            subgraph
            for subgraph in self.subgraphs[graph_idx]
            if subgraph.edge_index is not None
        ]

    def predict(
        self,
        mask,
        strategy="adaptive",
        route_confidence=0.85,
        early_stop_ratio=0.5,
        return_details=False,
    ):
        if strategy == "baseline":
            out, M = self.vote(mask)
            details = {
                "strategy": "baseline",
                "total_samples": int(out.shape[0]),
            }
            if return_details:
                return out, M, details
            return out, M
        if strategy == "adaptive":
            return self.adaptive_vote(
                mask,
                route_confidence=route_confidence,
                early_stop_ratio=early_stop_ratio,
                return_details=return_details,
            )
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    def enlarge_dataset(self, graphs):
        new_graphs = []
        ys = []
        for i in range(len(graphs)):
            subgraphs = self.Hasher.generate_graph_subgraphs(graphs[i].edge_index,graphs[i].x,graphs[i].y)
            for j in range(len(subgraphs)):
                new_graphs.append(subgraphs[j].to(self.device))
                ys.append(subgraphs[j].y.to(self.device))
            new_graphs.append(graphs[i].to(self.device))
            ys.append(graphs[i].y.to(self.device))
        return new_graphs, ys
    
    
    def train(self, train_args ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_args["lr"])
        criterion = torch.nn.CrossEntropyLoss()
    
        best_val_acc = 0.0
        best_train_acc = 0.0
        best_epoch = 0
        
        train_graphs = self.graphs[self.train_mask]
        #agumentate train datasets
        entrain_graphs,ys = self.enlarge_dataset(train_graphs)
        for epoch in range(0, train_args["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            loss = torch.zeros(1).to(self.device)
            for i in range(len(entrain_graphs)):
                out = self.model(entrain_graphs[i].x,entrain_graphs[i].edge_index)
                loss+=criterion(out, ys[i].to(torch.long))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=train_args["clip_max"])
            optimizer.step()
            
            train_acc = 0
            test_acc = 0
            val_acc = 0
            with torch.no_grad():
                out_train,_ = self.vote(self.train_mask)
                out_val,_ = self.vote(self.val_mask)
                #out_test,_ = self.vote(self.test_mask)
                train_acc = evaluate(out_train, self.labels[self.train_mask])
                val_acc = evaluate(out_val, self.labels[self.val_mask])
                #test_acc = evaluate(out_test, self.labels[self.test_mask])


            print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss.item():.4f}")
            if val_acc == best_val_acc and train_acc>best_train_acc: # New best results
                print("Train improved")
                best_train_acc = train_acc
                best_epoch = epoch
                store_checkpoint("robust_e/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.Hasher.T), self.model, train_acc, val_acc, test_acc)

            if val_acc > best_val_acc: # New best results
                print("Val improved")
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch
                store_checkpoint("robust_e/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.Hasher.T), self.model, train_acc, val_acc, test_acc)

            if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
                break
        
     
    def test(self ):   
        out_test,M = self.vote(self.test_mask)
        test_acc = evaluate(out_test, self.labels[self.test_mask])
        return test_acc, M
        
    def vote(self, mask):
        G_test = len(self.graphs[mask])
        idxs = np.array([i for i in range(len(self.graphs))])
        test_id = idxs[mask]
        
        votes = torch.zeros((G_test,self.num_labels))
        M =torch.zeros(G_test)
        
        
        self.model.eval()
        with torch.no_grad():
            for i in range(G_test):
                subgraphs = self._get_valid_subgraphs(int(test_id[i]))
                for subgraph in subgraphs:
                    out_sub = self.model(
                        subgraph.x.to(self.device),
                        subgraph.edge_index.to(self.device),
                    ).detach().cpu()
                    preds = out_sub[0].argmax(dim=0)
                    votes[i,preds]+=1

        M = compute_vote_margin(votes)
        return votes, M

    def adaptive_vote(
        self,
        mask,
        route_confidence=0.85,
        early_stop_ratio=0.5,
        return_details=False,
    ):
        G_test = len(self.graphs[mask])
        idxs = np.array([i for i in range(len(self.graphs))])
        test_id = idxs[mask]

        outputs = torch.zeros((G_test, self.num_labels), dtype=torch.float32)
        margins = torch.full((G_test,), -1.0, dtype=torch.float32)
        used_subgraphs = np.zeros(G_test, dtype=int)
        route_modes = np.array(["direct"] * G_test, dtype=object)
        base_confidences = np.zeros(G_test, dtype=float)
        total_subgraph_list = []

        self.model.eval()
        with torch.no_grad():
            for i in range(G_test):
                graph = self.graphs[int(test_id[i])]
                base_out = self.model(
                    graph.x.to(self.device),
                    graph.edge_index.to(self.device),
                ).detach().cpu()
                confidence, _ = confidence_from_logits(base_out)
                base_confidences[i] = float(confidence[0]) if confidence.numel() else 0.0

                if base_confidences[i] >= float(route_confidence):
                    outputs[i] = base_out[0]
                    total_subgraph_list.append(len(self._get_valid_subgraphs(int(test_id[i]))))
                    continue

                subgraphs = self._get_valid_subgraphs(int(test_id[i]))
                total_subgraphs = len(subgraphs)
                total_subgraph_list.append(total_subgraphs)
                if total_subgraphs == 0:
                    outputs[i] = base_out[0]
                    route_modes[i] = "direct_fallback"
                    continue

                route_modes[i] = "subgraph_full"
                for subgraph in subgraphs:
                    out_sub = self.model(
                        subgraph.x.to(self.device),
                        subgraph.edge_index.to(self.device),
                    ).detach().cpu()
                    pred = int(out_sub[0].argmax(dim=0))
                    outputs[i, pred] += 1
                    used_subgraphs[i] += 1
                    if should_early_stop_by_remaining_votes(
                        outputs[i].unsqueeze(0),
                        used_subgraphs[i],
                        total_subgraphs,
                    )[0]:
                        route_modes[i] = "subgraph_early_stop"
                        break
                margins[i] = compute_vote_margin(outputs[i].unsqueeze(0))[0]

        details = build_adaptive_details(
            strategy="adaptive",
            route_confidence=route_confidence,
            early_stop_ratio=early_stop_ratio,
            total_subgraphs=int(max(total_subgraph_list)) if total_subgraph_list else 0,
            base_confidences=base_confidences,
            route_modes=route_modes,
            used_subgraphs=used_subgraphs,
            total_subgraphs_per_sample=total_subgraph_list,
        )
        if return_details:
            return outputs, margins, details
        return outputs, margins
       
class RobustAmazonNodeClassifier():
    def __init__(self,model,Hasher,edge_index, x, y, train_idx, valid_idx, test_idx,num_labels):

        super(RobustAmazonNodeClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.Hasher = Hasher
        self.edge_index = edge_index.to(self.device)
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.train_idx = torch.as_tensor(train_idx, dtype=torch.long, device=self.device)
        self.valid_idx = torch.as_tensor(valid_idx, dtype=torch.long, device=self.device)
        self.test_idx = torch.as_tensor(test_idx, dtype=torch.long, device=self.device)
        self.num_labels= num_labels
        self.subgraphs = Hasher.generate_amazon_subgraphs(self.edge_index,self.x,self.y)
        self.T = self.Hasher.T
    def load_model(self,path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(
        self,
        mask,
        strategy="adaptive",
        route_confidence=0.85,
        early_stop_ratio=0.5,
        return_details=False,
    ):
        if strategy == "baseline":
            out, M = self.vote(mask)
            details = {
                "strategy": "baseline",
                "total_samples": int(out.shape[0]),
                "total_subgraphs_available": int(len(self.subgraphs)),
            }
            if return_details:
                return out, M, details
            return out, M
        if strategy == "adaptive":
            return self.adaptive_vote(
                mask,
                route_confidence=route_confidence,
                early_stop_ratio=early_stop_ratio,
                return_details=return_details,
            )
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    def train(self, train_args ):
        subgraphs = self.subgraphs#Hasher.generate_node_subgraphs(self.edge_index, self.x, self.y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_args["lr"])
        criterion = torch.nn.CrossEntropyLoss()
    
        best_val_acc = 0.0
        best_epoch = 0
        for epoch in range(0, train_args["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            loss = 0.0
            
            train_batch= random.sample(range(0,self.T),5)
            for i in range(len(train_batch)):
                batch_graph = copy.deepcopy(subgraphs[train_batch[i]]).to(self.device)
                print("training: ", i," / ", len(train_batch))
                out_sub = self.model(self.x,batch_graph.edge_index)
                loss+=criterion(out_sub[self.train_idx], self.y[self.train_idx])
                del batch_graph
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=train_args["clip_max"])
            optimizer.step()
            
            train_acc = 0
            test_acc = 0
            val_acc = 0
            print(f"Epoch: {epoch},  train_loss: {loss:.4f}")
            if epoch%25==15:
                with torch.no_grad():
                    outs = self.vote_multi([self.train_idx,self.valid_idx,self.test_idx])
                    out_train = outs[0]
                    out_val = outs[1]
                    out_test = outs[2]
                    train_acc = evaluate(out_train, self.y[self.train_idx].detach().cpu())
                    val_acc = evaluate(out_val, self.y[self.valid_idx].detach().cpu())
                    test_acc = evaluate(out_test, self.y[self.test_idx].detach().cpu())
                
                print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")
                if val_acc > best_val_acc: # New best results
                    print("Val improved")
                    best_val_acc = val_acc
                    best_epoch = epoch
                    store_checkpoint("robust_e/"+train_args["paper"], train_args["dataset"]+"/{}".format(self.Hasher.T), self.model, train_acc, val_acc, test_acc)

                if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
                    break
        
     
    def test(self ):   
        
        out_test,M = self.vote(self.test_idx)
        test_acc = evaluate(out_test, self.y[self.test_idx].detach().cpu())
        return test_acc, M
        
    def vote(self, mask):
        subgraphs = self.subgraphs
        mask_idx_cpu = get_mask_indices(mask)
        mask_idx = mask_idx_cpu.to(self.device)
        V_test = int(mask_idx_cpu.shape[0])
        votes = torch.zeros((V_test,self.num_labels), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            for i in range(len(subgraphs)):
                test_subgraph = copy.deepcopy(subgraphs[i]).to(self.device)
                out_sub = self.model(self.x,test_subgraph.edge_index)
                preds = out_sub.index_select(0, mask_idx).argmax(dim=1).detach().cpu()
                votes[torch.arange(V_test), preds] += 1
                del test_subgraph
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        M = compute_vote_margin(votes)
        return votes, M

    def adaptive_vote(
        self,
        mask,
        route_confidence=0.85,
        early_stop_ratio=0.5,
        return_details=False,
    ):
        mask_idx_cpu = get_mask_indices(mask)
        mask_idx = mask_idx_cpu.to(self.device)
        sample_count = int(mask_idx_cpu.shape[0])
        outputs = torch.zeros((sample_count, self.num_labels), dtype=torch.float32)
        margins = torch.full((sample_count,), -1.0, dtype=torch.float32)
        used_subgraphs = np.zeros(sample_count, dtype=int)
        route_modes = np.array(["direct"] * sample_count, dtype=object)

        self.model.eval()
        with torch.no_grad():
            base_out = self.model(self.x, self.edge_index)
            base_masked = base_out.index_select(0, mask_idx)

        base_confidences, _ = confidence_from_logits(base_masked)
        direct_mask = base_confidences >= float(route_confidence)
        outputs[direct_mask] = base_masked.detach().cpu()[direct_mask]

        low_conf_positions = (~direct_mask).nonzero(as_tuple=False).view(-1)
        total_subgraphs = len(self.subgraphs)

        if low_conf_positions.numel() > 0 and total_subgraphs > 0:
            low_votes = torch.zeros(
                (int(low_conf_positions.numel()), self.num_labels),
                dtype=torch.float32,
            )
            active = torch.ones(int(low_conf_positions.numel()), dtype=torch.bool)
            selected_mask_idx = mask_idx.index_select(0, low_conf_positions.to(self.device))

            with torch.no_grad():
                for subgraph in self.subgraphs:
                    if not active.any():
                        break
                    test_subgraph = copy.deepcopy(subgraph).to(self.device)
                    out_sub = self.model(self.x, test_subgraph.edge_index)
                    active_positions = active.nonzero(as_tuple=False).view(-1)
                    active_mask_idx = selected_mask_idx.index_select(
                        0, active_positions.to(self.device)
                    )
                    preds = (
                        out_sub.index_select(0, active_mask_idx)
                        .argmax(dim=1)
                        .detach()
                        .cpu()
                    )
                    for local_pos, pred in zip(active_positions.tolist(), preds.tolist()):
                        low_votes[local_pos, pred] += 1
                        used_subgraphs[int(low_conf_positions[local_pos])] += 1
                    reached = should_early_stop_by_remaining_votes(
                        low_votes,
                        used_subgraphs[low_conf_positions.numpy()],
                        total_subgraphs,
                    )
                    active = active & (~reached)
                    del test_subgraph
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            outputs[low_conf_positions] = low_votes
            margins[low_conf_positions] = compute_vote_margin(low_votes)
            low_used = used_subgraphs[low_conf_positions.numpy()]
            route_modes[low_conf_positions.numpy()] = "subgraph_full"
            route_modes[low_conf_positions.numpy()[low_used < total_subgraphs]] = (
                "subgraph_early_stop"
            )
        elif low_conf_positions.numel() > 0:
            outputs[low_conf_positions] = base_masked.detach().cpu()[low_conf_positions]
            route_modes[low_conf_positions.numpy()] = "direct_fallback"

        details = build_adaptive_details(
            strategy="adaptive",
            route_confidence=route_confidence,
            early_stop_ratio=early_stop_ratio,
            total_subgraphs=total_subgraphs,
            base_confidences=base_confidences.numpy(),
            route_modes=route_modes,
            used_subgraphs=used_subgraphs,
        )
        if return_details:
            return outputs, margins, details
        return outputs, margins
    
    def vote_multi(self, masks):
        subgraphs = self.subgraphs
        val_batch= random.sample(range(0,self.T),10)
        V_test = self.x.shape[0]
        votes = torch.zeros((V_test,self.num_labels))
        self.model.eval()
        for i in range(len(val_batch)):
            print("val: ", i, " / ",len(val_batch))
            test_subgraph = copy.deepcopy(subgraphs[val_batch[i]]).to(self.device)
            out_sub = self.model(self.x,test_subgraph.edge_index)
            preds = out_sub.argmax(dim=1).detach().cpu()
            votes[torch.arange(V_test), preds] += 1
            del test_subgraph
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return [votes[get_mask_indices(mask)] for mask in masks]
