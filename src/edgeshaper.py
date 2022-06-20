### Module implementing EdgeSHAPer algorithm ###
### Author: Andrea Mastropietro © All rights reserved ###
import torch
import torch.nn.functional as F

import numpy as np
from numpy.random import default_rng

from tqdm import tqdm


def edgeshaper(model, x, E, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = 42, device = "cpu"):
    """ Compute Shapley values approximation for edge importance in GNNs
        Args:
            model (Torch.NN.Module): Torch GNN model used.
            x (tensor): 2D tensor containing node features of the graph to explain
            E (tensor): edge index of the graph to be explained.
            num_nodes (int): number of nodes in the graph.
            P (float, optional): probablity of an edge to exist in the random graph. Defaults to the original graph density.
            M (int): number of Monte Carlo sampling steps to perform.
            target_class (int, optional): index of the class prediction to explain. Defaults to class 0.
            deviation (float, optional): deviation gap from sum of Shapley values and predicted output prob. If ```None```, M sampling
                steps are computed, otherwise the procedure stops when the desired approxiamtion ```deviation``` is reached. However, after M steps
                the procedure always terminates.
            log_odds (bool, optional). Default is ```False```. If ```True```, use log odds instead of probabilty as Value for Shaply values approximation.
            seed (float, optional): seed used for the random number generator.
            device (string, optional): states if using ```cuda``` or ```cpu```.
        Returns:
                list: list of Shapley values for the edges computed by EdgeSHAPer. The order of the edges is the same as in ```E```.
        """
    
    if deviation != None:
        return edgeshaper_deviation(model, x, E, M = M, target_class = target_class, P = P, deviation = deviation, log_odds = log_odds, seed = seed, device=device)


    rng = default_rng(seed = seed)
    model.eval()
    phi_edges = []

    num_nodes = x.shape[0]
    num_edges = E.shape[1]
    
    if P == None:
        max_num_edges = num_nodes*(num_nodes-1)
        graph_density = num_edges/max_num_edges
        P = graph_density

    for j in tqdm(range(num_edges)):
        marginal_contrib = 0
        for i in range(M):
            E_z_mask = rng.binomial(1, P, num_edges)
            E_mask = torch.ones(num_edges)
            pi = torch.randperm(num_edges)

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
            selected_edge_index = np.where(pi == j)[0].item()
            for k in range(num_edges):
                if k <= selected_edge_index:
                    E_j_plus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

            for k in range(num_edges):
                if k < selected_edge_index:
                    E_j_minus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]]


            #we compute marginal contribs
            
            # with edge j
            retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
            E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            
            out = model(x, E_j_plus, batch=batch)
            out_prob = None

            if not log_odds:
                out_prob = F.softmax(out, dim = 1)
            else:
                out_prob = out #out prob variable now containts log_odds
            
            V_j_plus = out_prob[0][target_class].item()

            # without edge j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
            E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            out = model(x, E_j_minus, batch=batch)

            if not log_odds:
                out_prob = F.softmax(out, dim = 1)
            else:
                out_prob = out
            
            V_j_minus = out_prob[0][target_class].item()

            marginal_contrib += (V_j_plus - V_j_minus)

        phi_edges.append(marginal_contrib/M)
        
    return phi_edges


def edgeshaper_deviation(model, x, E, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = 42, device = "cpu"):

    rng = default_rng(seed = seed)
    model.eval()
    batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
    out = model(x, E, batch=batch)
    out_prob_real = F.softmax(out, dim = 1)[0][target_class].item()

    num_nodes = x.shape[0]
    num_edges = E.shape[1]

    phi_edges = []
    phi_edges_current = [0] * num_edges
    
    if P == None:
        max_num_edges = num_nodes*(num_nodes-1)
        graph_density = num_edges/max_num_edges
        P = graph_density

    for i in tqdm(range(M)):
    
        for j in range(num_edges):
            
            
            E_z_mask = rng.binomial(1, P, num_edges)
            E_mask = torch.ones(num_edges)
            pi = torch.randperm(num_edges)

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
            selected_edge_index = np.where(pi == j)[0].item()
            for k in range(num_edges):
                if k <= selected_edge_index:
                    E_j_plus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

            for k in range(num_edges):
                if k < selected_edge_index:
                    E_j_minus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]]


            #we compute marginal contribs
            
            # with edge j
            retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
            E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            
            out = model(x, E_j_plus, batch=batch)
            out_prob = None

            if not log_odds:
                out_prob = F.softmax(out, dim = 1)
            else:
                out_prob = out #out prob variable now containts log_odds
            
            V_j_plus = out_prob[0][target_class].item()

            # without edge j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
            E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            out = model(x, E_j_minus, batch=batch)

            if not log_odds:
                out_prob = F.softmax(out, dim = 1)
            else:
                out_prob = out
            
            V_j_minus = out_prob[0][target_class].item()

            phi_edges_current[j] += (V_j_plus - V_j_minus)

        
        phi_edges = [elem / (i+1) for elem in phi_edges_current]
        print(sum(phi_edges))
        if abs(out_prob_real - sum(phi_edges)) <= deviation:
            break
             
    return phi_edges