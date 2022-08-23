#!/usr/bin/env python
# coding: utf-8

# # EdgeSHAPer experiments

# Explain the graph classification using Shaply values for edges. This script reproduces the experiments as found in the paper

# In[1]:


import torch
from torchdrug import data
import pandas as pd
import numpy as np
import random
from numpy.random import default_rng

from rdkit import Chem
from rdkit.Chem import Draw
from pysmiles import read_smiles
import networkx as nx

from rdkit_heatmaps import mapvalues2mol
from rdkit_heatmaps.utils import transform2png

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

from tqdm import tqdm


# In[2]:


# for debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# ## Data Loading

# In[55]:

###parameters###
####do not modify those if you want reproduce paper experiments#####
DATASET_TYPE = "single" 
TARGET = 1
TRAINING_SET_SPLIT = "FULL" #None, FULL, 0, 1, 2
MODEL_NUM = 0 
TARGET_CPDS = "P14416_P42336"
DATASET_NAME = "chembl29_predicting_target_" + TARGET_CPDS + "_target_"+ str(TARGET) +"_vs_random_cpds"
CSV_DATA_PATH = "../data/"+ DATASET_NAME + ".csv"
GNNEXPLAINER_USAGE = True
SEED = 42
SAVE = True
M = 100
##################

### Reprodicubility Settings

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

### uncomment this to use the CPDs shown in the main paper, otherwise keep intact for reproducing the quantitative results of the paper on test CPDs###
CPD_SELECTION = [] #["C#Cc1ccc2sc(C(=O)NCCCCN3CCN(c4ccccc4OC)CC3)cc2c1", "Cc1ncsc1-c1nnc(SCCCN2CCC3(CC3c3ccc(C(F)(F)F)cc3)C2)n1C"] #[]

smiles_df = pd.read_csv(CSV_DATA_PATH, sep = ",")
print(smiles_df.head())
print(smiles_df.shape)


# ## Define Custom Class
# We need to define the ChEMBL datasets class in order to load the model

# In[5]:


import os

from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.ChEMBL") #only first time you launch the class
#@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("path", "smiles_field", "target_fields"))
class ChEMBL(data.MoleculeDataset):
    

    def __init__(self, path, smiles_field, target_fields, verbose=1, **kwargs):
        
        self.path = path
        self.smiles_field = smiles_field
        self.target_fields= target_fields
        

        self.load_csv(self.path, smiles_field=self.smiles_field, target_fields=self.target_fields,
                      verbose=verbose, **kwargs)


# In[6]:


target_fields = ["label"]
chembl_dataset = ChEMBL(path = CSV_DATA_PATH, smiles_field = "nonstereo_aromatic_smiles", target_fields = target_fields)


# ## Obtain edge index to use with PyG


# In[7]:
    
smiles = chembl_dataset.smiles_list
mols = []
for i in tqdm(range(len(chembl_dataset.smiles_list))):
    mols.append(read_smiles(chembl_dataset.smiles_list[i]))

mol = mols[0]
print(mol.nodes(data='element'))
labels = nx.get_node_attributes(mol, 'element') 


# Define edge index 

# In[8]:


edge_index_list = []

for mol in tqdm(mols):
    adj = nx.to_scipy_sparse_matrix(mol).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_index_list.append(edge_index)

print("Number of molecules: ", len(mols))


# Define torchdrug dataset in order to get node features

# In[9]:


mols_torchdrug_format = []
for i in tqdm(range(len(chembl_dataset.smiles_list))):
    mols_torchdrug_format.append(data.Molecule.from_smiles(chembl_dataset.smiles_list[i], with_hydrogen = False))


# ## Create Custom Dataset

# In[10]:


import pandas as pd
from torch_geometric.data import InMemoryDataset, Data

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_list = []
y = torch.LongTensor(chembl_dataset.targets["label"]).to(device)

for i in tqdm(range(len(mols))):
    data_list.append(Data(x = mols_torchdrug_format[i].node_feature, edge_index = edge_index_list[i], y = y[i], smiles = chembl_dataset.smiles_list[i]))


# In[11]:


class ChEMBLDatasetPyG(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, data_list = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data_list = data_list

        
        data_list = self.data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        


# In[12]:


dataset = ChEMBLDatasetPyG(".", data_list = data_list)


# Split data in train/val/test (0.8/0.1/0.1)

# In[13]:


lengths = [int(0.8 * len(chembl_dataset)), int(0.1 * len(chembl_dataset))]
lengths += [len(chembl_dataset) - sum(lengths)]

print(lengths)
dataset = dataset.shuffle()
train_data = dataset[:lengths[0]]
val_data = dataset[lengths[0]+1:lengths[0] + lengths[1]+1]
test_data = dataset[lengths[0] + lengths[1] + 1: ] 
len(train_data), len(val_data), len(test_data)


# In[14]:


batch_size= 32
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


# ## GCN Definition

# In[15]:


from torch_geometric.nn import GCNConv, Linear
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(chembl_dataset.node_feature_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels=256).to(device)
print(model)


# ## Load the Model

# In[16]:


import os.path as osp

if TRAINING_SET_SPLIT == None:
    MODEL_PATH = "../models/PyG/" + DATASET_NAME + "_model_" + str(MODEL_NUM) + ".ckpt"
else:
    MODEL_PATH = "../models/PyG/" + DATASET_NAME + "_training_set_" + str(TRAINING_SET_SPLIT) + "_model_" + str(MODEL_NUM) + ".ckpt"

ckpt_path = osp.join(MODEL_PATH)
model.load_state_dict(torch.load(ckpt_path))
model.to(device)


# ## Test the Model

# In[17]:


def test(loader):
     model.eval()

     correct = 0
     for data in loader:  
         data = data.to(device)
         
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  
         correct += int((pred == data.y).sum()) 
     return correct / len(loader.dataset)  

test_acc = test(test_loader)    
print(f'Test Acc: {test_acc:.4f}')



# In[84]:


model.eval()
test_compounds_indices =  []
#randomly sample cpds

if len(CPD_SELECTION) == 0: 
    num_random_samples = 20
    test_compounds_indices_class_0 = []
    
    rng = default_rng(seed = 42)


    print("Choosing random samples for target compound...\n")
    while len(test_compounds_indices_class_0) < num_random_samples:
        random_index = rng.integers(low = 0, high = len(test_data), size = 1)[0]
        if random_index not in test_compounds_indices_class_0:
            real_class = test_data[random_index].y

            if real_class == 0:
                test_cpd = test_data[random_index].to(device)

                batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
                out = model(test_cpd.x, test_cpd.edge_index, batch=batch)
                out_prob = F.softmax(out, dim = 1)
                target_class = torch.argmax(out_prob[0]).item()

                if target_class == 0:
                    test_compounds_indices_class_0.append(random_index)

    print("Done!\n")

    test_compounds_indices = test_compounds_indices_class_0

else:
    for selected_cpd in CPD_SELECTION:
        for i in range(len(test_data)):
            if test_data[i].smiles == selected_cpd:
                test_compounds_indices.append(i)
                break

    print("Done!\n")

fidelity_values_gs = []
infidelity_values_gs = []

fidelity_values_ge = []
infidelity_values_ge = []

num_edges_min_top_k_gs = []
num_edges_pert_pos_gs = []

num_edges_min_top_k_ge = []
num_edges_pert_pos_ge = []

for test_set_index in tqdm(test_compounds_indices):

    print("Explaining test compound: ", test_set_index)
    test_cpd = test_data[test_set_index].to(device)
    print("SMILES: ", test_cpd.smiles)
    mol_test = data.Molecule.from_smiles(test_cpd.smiles, with_hydrogen=False)
    


    # all nodes belong to same graph
    batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
    out = model(test_cpd.x, test_cpd.edge_index, batch=batch)
    out_prob = F.softmax(out, dim = 1)
    target_class = torch.argmax(out_prob[0]).item()


    # Creating folder and saving smiles



    if DATASET_TYPE == "dual":
        FOLDER_NAME = "../results/explanations/" + DATASET_TYPE + "/" + TARGET_CPDS
    else:
        FOLDER_NAME = "../results/explanations/" + DATASET_TYPE + "/" + TARGET_CPDS +"/Target " + str(TARGET) + " vs Random"

    CPD_FOLDER_NAME = FOLDER_NAME + "/" + test_cpd.smiles

    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)

    if not os.path.exists(CPD_FOLDER_NAME):
        os.makedirs(CPD_FOLDER_NAME)


    mol_test.visualize(save_file = CPD_FOLDER_NAME + "/" + test_cpd.smiles + ".png")


    E = test_cpd.edge_index
    num_nodes = test_cpd.x.shape[0]
    max_num_edges = num_nodes*(num_nodes-1)
    num_edges = E.shape[1]
    graph_density = num_edges/max_num_edges
    P = graph_density 


    # In[89]:


    
    phi_edges = []
    
    rng = default_rng(seed = 42) 
    model.eval()

    log_odds = False
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

            batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
            
            out = model(test_cpd.x, E_j_plus, batch=batch)
            out_prob = None

            if not log_odds:
                out_prob = F.softmax(out, dim = 1)
            else:
                out_prob = out #out prob variable now containts log_odds
            
            V_j_plus = out_prob[0][target_class].item()

            # without edge j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
            E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)

            batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
            out = model(test_cpd.x, E_j_minus, batch=batch)

            if not log_odds:
                out_prob = F.softmax(out, dim = 1)
            else:
                out_prob = out
            
            V_j_minus = out_prob[0][target_class].item()

            marginal_contrib += (V_j_plus - V_j_minus)

        phi_edges.append(marginal_contrib/M)     
            
    sum(phi_edges)


    # In[83]:



    # In[76]:
    if SAVE:
        INFO_EXPLANATIONS = "info_explanations"
        if TRAINING_SET_SPLIT != None:
            INFO_EXPLANATIONS += "_training_split_" + str(TRAINING_SET_SPLIT)

        with open(CPD_FOLDER_NAME + "/" + INFO_EXPLANATIONS + ".txt", "w+") as saveFile:
            saveFile.write("Test set index: " + str(test_set_index) + "\n\n")

        with open(CPD_FOLDER_NAME + "/" + INFO_EXPLANATIONS + ".txt", "a") as saveFile:
            saveFile.write("Target Class: " + str(target_class) + "\n\n")

        tagert_compound_sample = "TARGET" if target_class == 0 else "RANDOM"

        with open(CPD_FOLDER_NAME + "/" + INFO_EXPLANATIONS + ".txt", "a") as saveFile:
            saveFile.write("Target compound: " + str(tagert_compound_sample) + "\n\n")

        with open(CPD_FOLDER_NAME + "/" + INFO_EXPLANATIONS + ".txt", "a") as saveFile:
            saveFile.write("Sum of Shapley Values: " + str(sum(phi_edges)) + "\n\n")

        with open(CPD_FOLDER_NAME + "/" + INFO_EXPLANATIONS + ".txt", "a") as saveFile:
            saveFile.write("Shapley values:\n")
            for phi_val in phi_edges:
                saveFile.write(str(phi_val) + "\n")


    # ## Visualize Explanations

    # In[78]:


    

    important_edges_ranking = np.argsort(-np.array(phi_edges))
    edge_index = E.to(device)

    test_mol = Chem.MolFromSmiles(test_cpd.smiles)
    test_mol = Draw.PrepareMolForDrawing(test_mol)

    num_bonds = len(test_mol.GetBonds())
    num_atoms = len(test_mol.GetAtoms())

    rdkit_bonds = {}

    for i in range(num_bonds):
        init_atom = test_mol.GetBondWithIdx(i).GetBeginAtomIdx()
        end_atom = test_mol.GetBondWithIdx(i).GetEndAtomIdx()
        bond_type = test_mol.GetBondWithIdx(i).GetBondType()
        #print("Bond: ", i, " " , init_atom, "-" , end_atom, " ", bond_type)
        rdkit_bonds[(init_atom, end_atom)] = i

    rdkit_bonds_phi = [0]*num_bonds
    for i in range(len(phi_edges)):
        phi_value = phi_edges[i]
        init_atom = edge_index[0][i].item()
        end_atom = edge_index[1][i].item()
        
        if (init_atom, end_atom) in rdkit_bonds:
            bond_index = rdkit_bonds[(init_atom, end_atom)]
            rdkit_bonds_phi[bond_index] += phi_value
        if (end_atom, init_atom) in rdkit_bonds:
            bond_index = rdkit_bonds[(end_atom, init_atom)]
            rdkit_bonds_phi[bond_index] += phi_value

    plt.clf()
    canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi, atom_width=0.2, bond_length=0.5, bond_width=0.5) #TBD: only one direction for edges? bonds weights is wrt rdkit bonds order?
    img = transform2png(canvas.GetDrawingText())

    
    if SAVE:
        if TRAINING_SET_SPLIT == None:
            img.save(CPD_FOLDER_NAME + "/" + "EdgeSHAPer_MC_" + str(M) + "_heatmap.png", dpi = (300,300))
        else:
            img.save(CPD_FOLDER_NAME + "/" + "EdgeSHAPer_MC_" + str(M) + "_train_" + str(TRAINING_SET_SPLIT) + "_heatmap.png", dpi = (300,300)) 
    

    # ## GNNExplainer
    # Explain the classification of a test cpd using GNNExplainer

    # In[79]:


    import os.path as osp

    import torch
    import torch.nn.functional as F


    from torch_geometric.nn import GNNExplainer

    if GNNEXPLAINER_USAGE:
        epochs = 100
        
        x, edge_index, edge_weight = test_cpd.x, test_cpd.edge_index, None

        explainer = GNNExplainer(model, epochs=epochs, return_type='log_prob')

        node_feat_mask, edge_mask = explainer.explain_graph(x, edge_index)



    


        from matplotlib.pyplot import figure

        figure(figsize=(16, 12), dpi=80)
        edge_mask = edge_mask.to("cpu")


        rdkit_bonds_GNNExpl_importance = [0]*num_bonds
        for i in range(len(edge_mask)):
            GNNExpl_importance_value = edge_mask[i]
            init_atom = edge_index[0][i].item()
            end_atom = edge_index[1][i].item()
            
            if (init_atom, end_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(init_atom, end_atom)]
                rdkit_bonds_GNNExpl_importance[bond_index] += GNNExpl_importance_value.item()
            if (end_atom, init_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(end_atom, init_atom)]
                rdkit_bonds_GNNExpl_importance[bond_index] += GNNExpl_importance_value.item()

        plt.clf()
        canvas = mapvalues2mol(test_mol, None, rdkit_bonds_GNNExpl_importance, atom_width=0.2, bond_length=0.5, bond_width=0.5)
        img = transform2png(canvas.GetDrawingText())

        if SAVE:
            if TRAINING_SET_SPLIT == None:
                img.save(CPD_FOLDER_NAME + "/" + "GNNExplainer_" + "heatmap.png", dpi = (300,300))
            else:
                img.save(CPD_FOLDER_NAME + "/" + "GNNExplainer_" + "train_" + str(TRAINING_SET_SPLIT) + "_heatmap.png", dpi = (300,300))


    batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
    out_log_odds = model(test_cpd.x, test_cpd.edge_index, batch=batch)
    out_prob = F.softmax(out_log_odds, dim = 1)
    original_pred_prob = out_prob[0][target_class].item()

    #mninimal top-k for EdgeSHAPer
    pertinent_set_indices = []
    pertinent_set_edge_index = None
    edge_index = E.to(device)
    print("Complete graph predicts class: ", target_class)

    model.eval()
    for i in range(important_edges_ranking.shape[0]):
        index_of_edge_to_remove = important_edges_ranking[i]
        pertinent_set_indices.append(index_of_edge_to_remove)

        reduced_edge_index = torch.index_select(edge_index, dim = 1, index = torch.LongTensor(important_edges_ranking[i:]).to(device))
        
        # all nodes belong to same graph
        batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
        out = model(test_cpd.x, reduced_edge_index, batch=batch)
        out_prob = F.softmax(out, dim = 1)
        # print(out_prob)
        predicted_class = torch.argmax(out_prob[0]).item()

        if predicted_class != target_class:
            pred_prob = out_prob[0][target_class].item()
            fidelity = original_pred_prob - pred_prob
            fidelity_values_gs.append(fidelity)
            print("FID+ using Minimal top-k: ", fidelity)
            break

    pertinent_set_edge_index = torch.index_select(edge_index, dim = 1, index = torch.LongTensor(pertinent_set_indices).to(device))
    num_edges_min_top_k_gs.append(len(pertinent_set_indices))
    

    if SAVE:
        with open(CPD_FOLDER_NAME + "/" + INFO_EXPLANATIONS + ".txt", "a") as saveFile:
                saveFile.write("\nMinimal top-k Set:\n")
                saveFile.write("Set indices on edge ranking: " + str(pertinent_set_indices))
                saveFile.write("\nSet edge index: " + str(pertinent_set_edge_index))
                saveFile.write("\nFID+: " + str(fidelity))

    ####visualize EdgeSHAPer Minimal top-k set####

    rdkit_bonds_phi_pertinent_set = [0]*num_bonds
    for i in range(len(pertinent_set_indices)):
        
        init_atom = pertinent_set_edge_index[0][i].item()
        end_atom = pertinent_set_edge_index[1][i].item()
        
        
        if (init_atom, end_atom) in rdkit_bonds:
            bond_index = rdkit_bonds[(init_atom, end_atom)]
            if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]
        if (end_atom, init_atom) in rdkit_bonds:
            bond_index = rdkit_bonds[(end_atom, init_atom)]
            if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]

    plt.clf()
    canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5)
    img = transform2png(canvas.GetDrawingText())

    if SAVE:
        if TRAINING_SET_SPLIT == None:
            img.save(CPD_FOLDER_NAME + "/" + "EdgeSHAPer_min_top_k_" + "heatmap.png", dpi = (300,300))
        else:
            img.save(CPD_FOLDER_NAME + "/" + "EdgeSHAPer_min_top_k_" + "train_" + str(TRAINING_SET_SPLIT) + "_heatmap.png", dpi = (300,300))

    #pertinent positive for EdgeSHAPer
    for i in range(important_edges_ranking.shape[0]+1):
        reduced_edge_index = torch.index_select(edge_index, dim = 1, index = torch.LongTensor(important_edges_ranking[0:i]).to(device))
        batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
        out = model(test_cpd.x, reduced_edge_index, batch=batch)
        out_prob = F.softmax(out, dim = 1)
        # print(out_prob)
        predicted_class = torch.argmax(out_prob[0]).item()
        if (predicted_class == target_class):
            
            
            pred_prob = out_prob[0][target_class].item()
            infidelity = original_pred_prob-pred_prob
            infidelity_values_gs.append(infidelity)

            if SAVE:
                with open(CPD_FOLDER_NAME + "/" + INFO_EXPLANATIONS + ".txt", "a") as saveFile:
                    saveFile.write("\n\nPertinent Positive Set:\n")
                    saveFile.write("\nPertinent set edge index: " + str(reduced_edge_index))
                    saveFile.write("\nFID-: " + str(infidelity))

            print("FID- using pertinent positive: ", infidelity)
            break
    
    num_edges_pert_pos_gs.append(reduced_edge_index.shape[1])

    ### viz pertinent positive for EdgeSHAPer
    rdkit_bonds_phi_pertinent_set = [0]*num_bonds
    pertinent_set_edge_index = reduced_edge_index
    for i in range(pertinent_set_edge_index.shape[1]):
        
        init_atom = pertinent_set_edge_index[0][i].item()
        end_atom = pertinent_set_edge_index[1][i].item()
        
        
        if (init_atom, end_atom) in rdkit_bonds:
            bond_index = rdkit_bonds[(init_atom, end_atom)]
            if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]
        if (end_atom, init_atom) in rdkit_bonds:
            bond_index = rdkit_bonds[(end_atom, init_atom)]
            if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]

    plt.clf()
    canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5)
    img = transform2png(canvas.GetDrawingText())

    if SAVE:
        if TRAINING_SET_SPLIT == None:
            img.save(CPD_FOLDER_NAME + "/" + "EdgeSHAPer_pert_pos_" + "heatmap.png", dpi = (300,300))
        else:
            img.save(CPD_FOLDER_NAME + "/" + "EdgeSHAPer_pert_pos_" + "train_" + str(TRAINING_SET_SPLIT) + "_heatmap.png", dpi = (300,300))


    if GNNEXPLAINER_USAGE:
        #Minimal top-k for GNNExplainer        

        pertinent_set_indices = []
        pertinent_set_edge_index = None
        important_gnn_expl_edges_ranking = np.argsort(-np.array(edge_mask))
        for i in range(important_gnn_expl_edges_ranking.shape[0]):
            index_of_edge_to_remove = important_gnn_expl_edges_ranking[i]
            pertinent_set_indices.append(index_of_edge_to_remove)

            reduced_edge_index = torch.index_select(edge_index, dim = 1, index = torch.LongTensor(important_gnn_expl_edges_ranking[i:]).to(device))
            
            # all nodes belong to same graph
            batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
            out = model(test_cpd.x, reduced_edge_index, batch=batch)
            out_prob = F.softmax(out, dim = 1)
            
            predicted_class = torch.argmax(out_prob[0]).item()

            if predicted_class != target_class:
                pred_prob = out_prob[0][target_class].item()
                fidelity =  original_pred_prob - pred_prob
                fidelity_values_ge.append(fidelity)
                print("FID+ using Minimal top-k: ", fidelity)
                break

        pertinent_set_edge_index = torch.index_select(edge_index, dim = 1, index = torch.LongTensor(pertinent_set_indices).to(device))
        num_edges_min_top_k_ge.append(len(pertinent_set_indices))

        if SAVE:
            with open(CPD_FOLDER_NAME + "/" + INFO_EXPLANATIONS + ".txt", "a") as saveFile:
                    saveFile.write("\nMinimal top-k Set GNNExplainer:\n")
                    saveFile.write("Set indices on edge ranking: " + str(pertinent_set_indices))
                    saveFile.write("\nSet edge index: " + str(pertinent_set_edge_index))
                    saveFile.write("\nFID+: " + str(fidelity))

        ####visualize GNNExpl Minimal top-k set####

        rdkit_bonds_gnn_expl_pertinent_set = [0]*num_bonds
        for i in range(len(pertinent_set_indices)):
            
            init_atom = pertinent_set_edge_index[0][i].item()
            end_atom = pertinent_set_edge_index[1][i].item()
            
            
            if (init_atom, end_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(init_atom, end_atom)]
                if rdkit_bonds_gnn_expl_pertinent_set[bond_index] == 0:
                    rdkit_bonds_gnn_expl_pertinent_set[bond_index] += rdkit_bonds_GNNExpl_importance[bond_index]
            if (end_atom, init_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(end_atom, init_atom)]
                if rdkit_bonds_gnn_expl_pertinent_set[bond_index] == 0:
                    rdkit_bonds_gnn_expl_pertinent_set[bond_index] += rdkit_bonds_GNNExpl_importance[bond_index]

        plt.clf()
        canvas = mapvalues2mol(test_mol, None, rdkit_bonds_gnn_expl_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5)
        img = transform2png(canvas.GetDrawingText())

        if SAVE:
            if TRAINING_SET_SPLIT == None:
                img.save(CPD_FOLDER_NAME + "/" + "GNNExplainer_min_top_k_" + "heatmap.png", dpi = (300,300))
            else:
                img.save(CPD_FOLDER_NAME + "/" + "GNNExplainer_min_top_k_" + "train_" + str(TRAINING_SET_SPLIT) + "_heatmap.png", dpi = (300,300))

        #pertinent positive for GNNExplainer
        for i in range(important_gnn_expl_edges_ranking.shape[0]+1):
            reduced_edge_index = torch.index_select(edge_index, dim = 1, index = torch.LongTensor(important_gnn_expl_edges_ranking[0:i]).to(device))
            batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
            out = model(test_cpd.x, reduced_edge_index, batch=batch)
            out_prob = F.softmax(out, dim = 1)
            
            predicted_class = torch.argmax(out_prob[0]).item()
            if (predicted_class == target_class):
                
                
                pred_prob = out_prob[0][target_class].item()
                infidelity = original_pred_prob-pred_prob
                infidelity_values_ge.append(infidelity)

                if SAVE:
                    with open(CPD_FOLDER_NAME + "/" + INFO_EXPLANATIONS + ".txt", "a") as saveFile:
                        saveFile.write("\n\nPertinent Positive Set GNNExplainer:\n")
                        saveFile.write("\nPertinent set edge index: " + str(reduced_edge_index))
                        saveFile.write("\nFID-: " + str(infidelity))
                print("FID- using pertinent positive: ", infidelity)
                break

        num_edges_pert_pos_ge.append(reduced_edge_index.shape[1])

        ### viz pertinent positive for GNNExplainer
        rdkit_bonds_gnn_expl_pertinent_set = [0]*num_bonds
        pertinent_set_edge_index = reduced_edge_index
        for i in range(pertinent_set_edge_index.shape[1]):
            
            init_atom = pertinent_set_edge_index[0][i].item()
            end_atom = pertinent_set_edge_index[1][i].item()
            
            
            if (init_atom, end_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(init_atom, end_atom)]
                if rdkit_bonds_gnn_expl_pertinent_set[bond_index] == 0:
                    rdkit_bonds_gnn_expl_pertinent_set[bond_index] += rdkit_bonds_GNNExpl_importance[bond_index]
            if (end_atom, init_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(end_atom, init_atom)]
                if rdkit_bonds_gnn_expl_pertinent_set[bond_index] == 0:
                    rdkit_bonds_gnn_expl_pertinent_set[bond_index] += rdkit_bonds_GNNExpl_importance[bond_index]

        plt.clf()
        canvas = mapvalues2mol(test_mol, None, rdkit_bonds_gnn_expl_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5)
        img = transform2png(canvas.GetDrawingText())

        if SAVE:
            if TRAINING_SET_SPLIT == None:
                img.save(CPD_FOLDER_NAME + "/" + "GNNExplainer_pert_pos_" + "heatmap.png", dpi = (300,300))
            else:
                img.save(CPD_FOLDER_NAME + "/" + "GNNExplainer_pert_pos_" + "train_" + str(TRAINING_SET_SPLIT) + "_heatmap.png", dpi = (300,300))


    plt.close("all")


print("Avg EdgeSHAPer FID+: ", round(sum(fidelity_values_gs)/len(fidelity_values_gs), 3))
print("Avg EdgeSHAPer FID-: ", round(sum(infidelity_values_gs)/len(infidelity_values_gs), 3))

if GNNEXPLAINER_USAGE:
    print("Avg GNNExplainer FID+: ", round(sum(fidelity_values_ge)/len(fidelity_values_ge), 3))
    print("Avg GNNExplainer FID-: ", round(sum(infidelity_values_ge)/len(infidelity_values_ge), 3))

print("=========================================================")

print("Avg num edges minimal top-k set EdgeSHAPer: ", round(sum(num_edges_min_top_k_gs)/len(num_edges_min_top_k_gs), 3))
print("Avg num edges pertinent pos set EdgeSHAPer: ", round(sum(num_edges_pert_pos_gs)/len(num_edges_pert_pos_gs), 3))

if GNNEXPLAINER_USAGE:
    print("Avg num edges minimal top-k set GNNExplainer: ", round(sum(num_edges_min_top_k_ge)/len(num_edges_min_top_k_ge), 3))
    print("Avg num edges pertinent pos set GNNExplainer: ", round(sum(num_edges_pert_pos_ge)/len(num_edges_pert_pos_ge), 3))
