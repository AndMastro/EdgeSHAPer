#!/usr/bin/env python
# coding: utf-8

# # GraphSHAPer implementation

# Explain the graph classification using Shaply values for edges.
# Determine the contribution of each edge towards the output value (predicted problability)
# 
# Starting point:
# 
# TODO:
# 
# * Method 1 (jackknife-style sampling): compute approx shapley value for edge $e$ considering all possibile coalitions of $E - 1$ edges, where $E$ is the number of edges.
# * Method 2 (montecarlo sampling)
# * Method 3 (exahustive search)
# * Method 4 (node sampling): sample a number of nodes and work in the nodes k-hop subgraph similarly to a node classification explanation 

# In[1]:


import torch
from torchdrug import data
import pandas as pd
import numpy as np
import random
from numpy.random import default_rng

from tqdm import tqdm


# In[2]:


# for debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# ## Reprodicubility Settings

# In[3]:


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ## Data Loading

# In[55]:


DATASET_TYPE = "single" #"single", dual
TARGET = 1
MODEL_NUM = 2
TARGET_CPDS = "P14416_P42336"
DATASET_NAME = "chembl29_predicting_target_" + TARGET_CPDS + "_target_"+ str(TARGET) +"_vs_random_cpds" # "chembl29_dt_cpds_" + TARGET_CPDS + "_balanced" #"chembl29_predicting_target_P14416_P42336_target_1_vs_random_cpds"
CSV_DATA_PATH = "../data/"+ DATASET_NAME + ".csv"

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
        # path = os.path.expanduser(path)# if not os.path.exists(path):
        #     os.makedirs(path)
        self.path = path
        self.smiles_field = smiles_field
        self.target_fields= target_fields
        #print(self.path)
        # zip_file = utils.download(self.url, path, md5=self.md5)
        # csv_file = utils.extract(zip_file)

        self.load_csv(self.path, smiles_field=self.smiles_field, target_fields=self.target_fields,
                      verbose=verbose, **kwargs)


# In[6]:


target_fields = ["label"]
chembl_dataset = ChEMBL(path = CSV_DATA_PATH, smiles_field = "nonstereo_aromatic_smiles", target_fields = target_fields)


# ## Obtain edge index to use with PyG

# Visualize molecules using NetworkX

# In[7]:


from pysmiles import read_smiles
import networkx as nx
    
smiles = chembl_dataset.smiles_list
mols = []
for i in tqdm(range(len(chembl_dataset.smiles_list))):
    mols.append(read_smiles(chembl_dataset.smiles_list[i]))

mol = mols[0]
print(mol.nodes(data='element'))
labels = nx.get_node_attributes(mol, 'element') 
# nx.draw(mol, labels = labels, pos=nx.spring_layout(mol))


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
        #self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_list = data_list

        # Read data into huge `Data` list.
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
test_data = dataset[lengths[0] + lengths[1] : ]
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

MODEL_PATH = "../models/PyG/" + DATASET_NAME + "_model_" + str(MODEL_NUM) + ".ckpt"
ckpt_path = osp.join(MODEL_PATH)
model.load_state_dict(torch.load(ckpt_path))
model.to(device)


# ## Test the Model

# In[17]:


def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(device)
         
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

test_acc = test(test_loader)    
print(f'Test Acc: {test_acc:.4f}')



# In[84]:


#test nums 78, 89 is predicted to interact with target cpd, sample 0 is random for single target dataset with target 1. class 0 interacts, class 1 is random
# 78 is predicted to interact with target for dataset with target 2. 0 interacts with random
# for dual target Q9Y5N1_P31645 sample 0 has class 0
# for dual target P27338_P29274 sample 0 is class 0
# for dual targe P27338_P22303 sample 4 is class 0

#we randomly sample 10 cmps from class 0 and 10 from class 1
num_random_samples = 10
test_compounds_indices_class_0 = []
test_compounds_indices_class_1 = []
rng = default_rng(seed = 42)
model.eval()

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

print("Choosing random samples for random target...\n")
while len(test_compounds_indices_class_1) < num_random_samples:
    random_index = rng.integers(low = 0, high = len(test_data), size = 1)[0]
    if random_index not in test_compounds_indices_class_1:
        real_class = test_data[random_index].y

        if real_class == 1:
            test_cpd = test_data[random_index].to(device)

            batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
            out = model(test_cpd.x, test_cpd.edge_index, batch=batch)
            out_prob = F.softmax(out, dim = 1)
            target_class = torch.argmax(out_prob[0]).item()

            if target_class == 1:
                test_compounds_indices_class_1.append(random_index)

print("Done!\n")

test_compounds_indices = test_compounds_indices_class_0 + test_compounds_indices_class_1

for test_set_index in test_compounds_indices:

    print("Explaining test compound: ", test_set_index)
    test_cpd = test_data[test_set_index].to(device)
    print("SMILES: ", test_cpd.smiles)
    mol_test = data.Molecule.from_smiles(test_cpd.smiles, with_hydrogen=False)
    


    # all nodes belong to same graph
    batch = torch.zeros(test_cpd.x.shape[0], dtype=int, device=test_cpd.x.device)
    out = model(test_cpd.x, test_cpd.edge_index, batch=batch)
    out_prob = F.softmax(out, dim = 1)
    target_class = torch.argmax(out_prob[0]).item()
    print("out", out)
    print("out_prob ", out_prob)
    print("target_class: ", target_class)


    # Creating folder and saving smiles



    if DATASET_TYPE == "dual":
        FOLDER_NAME = "../results/explanations/" + DATASET_TYPE + "/" + TARGET_CPDS
    else:
        FOLDER_NAME = "../results/explanations/" + DATASET_TYPE + "/" + TARGET_CPDS +"/Target " + str(TARGET) + " vs Random"

    CPD_FOLDER_NAME = FOLDER_NAME + "/" + test_cpd.smiles

    if not os.path.exists(FOLDER_NAME):
        os.mkdir(FOLDER_NAME)

    if not os.path.exists(CPD_FOLDER_NAME):
        os.mkdir(CPD_FOLDER_NAME)


    mol_test.visualize(save_file = CPD_FOLDER_NAME + "/" + test_cpd.smiles + ".png")


    print(test_cpd.edge_index)
    print(test_cpd.edge_index.shape)


    


    E = test_cpd.edge_index
    num_nodes = test_cpd.x.shape[0]
    max_num_edges = num_nodes*(num_nodes-1)
    num_edges = E.shape[1]
    graph_density = num_edges/max_num_edges
    P = graph_density #this is to be discussed


    # In[89]:


    M = 100
    phi_edges = []
    # marginal_contrib = 0

    rng = default_rng(seed = 42) #to seed or not to seed?
    model.eval()

    log_odds = False
    for j in tqdm(range(num_edges)):
        marginal_contrib = 0
        for i in range(M):
            E_z_mask = rng.binomial(1, P, num_edges)
            #E_z_index = torch.IntTensor(torch.nonzero(torch.IntTensor(E_z_mask)).tolist()).to(device).squeeze()
            #E_z = torch.index_select(E, dim = 1, index = E_z_index)
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
            
            V_j_plus = out_prob[0][target_class].item() #probably the predicted class changes when selecting/deselecing certain edges for class 1: more iterations needed?

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


    print(out_prob)
    print(phi_edges)


    # In[76]:


    with open(CPD_FOLDER_NAME + "/info_explanations.txt", "w+") as saveFile:
        saveFile.write("Test set index: " + str(test_set_index) + "\n\n")

    with open(CPD_FOLDER_NAME + "/info_explanations.txt", "a") as saveFile:
        saveFile.write("Target Class: " + str(target_class) + "\n\n")

    tagert_compound_sample = "TARGET" if target_class == 0 else "RANDOM"

    with open(CPD_FOLDER_NAME + "/info_explanations.txt", "a") as saveFile:
        saveFile.write("Target compound: " + str(tagert_compound_sample) + "\n\n")

    with open(CPD_FOLDER_NAME + "/info_explanations.txt", "a") as saveFile:
        saveFile.write("Sum of Shapley Values: " + str(sum(phi_edges)) + "\n\n")

    with open(CPD_FOLDER_NAME + "/info_explanations.txt", "a") as saveFile:
        saveFile.write("Shapley values:\n")
        for phi_val in phi_edges:
            saveFile.write(str(phi_val) + "\n")


    # ## Visualize Explanation

    # In[78]:


    from matplotlib.pyplot import figure
    import matplotlib.pyplot as plt

    important_edges_ranking = np.argsort(-phi_edges)
    print(important_edges_ranking)
    sorted_phi_edges = sorted(phi_edges, reverse = True)
    print(sorted_phi_edges)
    print(sum(phi_edges))

    threshold = np.median(phi_edges) #to discuss when an edge in important or not
    hard_edge_mask = (torch.FloatTensor(phi_edges) > threshold).to(torch.float) #>=
    print(hard_edge_mask.shape)

    important_edges_index = torch.nonzero(hard_edge_mask == 1)
    print(important_edges_index)

    edge_index = E.to(device)
    important_edges_index = important_edges_index.to(device)

    important_edges = torch.index_select(edge_index, dim = 1, index = important_edges_index.squeeze())
    print(important_edges)

    edges_color = []
    mol = read_smiles(test_cpd.smiles)

    #standard visualization for importan egdes GNNExplainer-like
    for edge in mol.edges:
        found_from = False
        found_to = False
        for i in range(important_edges.shape[1]):
            if edge[0] == important_edges[0][i] and edge[1] == important_edges[1][i]:
                found_from = True
            if edge[1] == important_edges[0][i] and edge[0] == important_edges[1][i]:
                found_to = True
        if found_from and found_to:
            edges_color.append("red")
        elif found_from or found_to:
            edges_color.append("orange")
        else:
            edges_color.append("black")   

    #visualization for indentifying unimportant edges (edge is important if both directins are above threshold)
    # for edge in mol.edges:
    #     found_from = False
    #     found_to = False
    #     for i in range(important_edges.shape[1]):
    #         if edge[0] == important_edges[0][i] and edge[1] == important_edges[1][i]:
    #             found_from = True
    #         if edge[1] == important_edges[0][i] and edge[0] == important_edges[1][i]:
    #             found_to = True
    #     if found_from and found_to:
    #         edges_color.append("red")
    #     # elif found_from or found_to:
    #     #     edges_color.append("red")
    #     else:
    #         edges_color.append("black")   

    figure(figsize=(16, 12), dpi=80)
    #mol = nx.DiGraph(mol)
    labels = nx.get_node_attributes(mol, 'element') 
    nx.draw(mol, with_labels = True, edge_color = edges_color, pos=nx.spring_layout(mol))
    plt.savefig(CPD_FOLDER_NAME + "/" + test_cpd.smiles + "_GraphSHAPer_MC_" + str(M), dpi=300, bbox_inches='tight')
    


    # ## GNNExplainer
    # Explain the classification of a test cpd using GNNExplainer

    # In[79]:


    import os.path as osp

    import torch
    import torch.nn.functional as F


    from torch_geometric.nn import GNNExplainer

    torch.manual_seed(42)
    epochs = 100
    # test_cpd = test_data[0].to(device)

    #model = Net().to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    x, edge_index, edge_weight = test_cpd.x, test_cpd.edge_index, None

    explainer = GNNExplainer(model, epochs=epochs, return_type='log_prob')

    node_feat_mask, edge_mask = explainer.explain_graph(x, edge_index)


    # ### Plot Explaination Subgraph

    # In[80]:


    from matplotlib.pyplot import figure

    figure(figsize=(16, 12), dpi=80)
    threshold = 0.75
    edge_mask = edge_mask.to("cpu")
    ax, G = explainer.visualize_subgraph(edge_index = edge_index, edge_mask = edge_mask, node_idx = -1, y=None, threshold=threshold)
    plt.savefig(CPD_FOLDER_NAME + "/" + test_cpd.smiles + "_GNNExplainer_viz0", dpi=300, bbox_inches='tight')
    


    # In[81]:


    hard_edge_mask = (edge_mask >= threshold).to(torch.float)
    hard_edge_mask.shape

    important_edges_index = torch.nonzero(hard_edge_mask == 1)
    print(important_edges_index)

    edge_index = edge_index.to(device)
    important_edges_index = important_edges_index.to(device)

    important_edges = torch.index_select(edge_index, dim = 1, index = important_edges_index.squeeze())
    print(important_edges)

    edges_color = []
    mol = read_smiles(test_cpd.smiles)

    for edge in mol.edges:
        found_from = False
        found_to = False
        for i in range(important_edges.shape[1]):
            if edge[0] == important_edges[0][i] and edge[1] == important_edges[1][i]:
                found_from = True
            if edge[1] == important_edges[0][i] and edge[0] == important_edges[1][i]:
                found_to = True
        if found_from and found_to:
            edges_color.append("red")
        elif found_from or found_to:
            edges_color.append("orange")
        else:
            edges_color.append("black")   

    figure(figsize=(16, 12), dpi=80)
    #mol = nx.DiGraph(mol)
    labels = nx.get_node_attributes(mol, 'element') 
    nx.draw(mol, with_labels = True, edge_color = edges_color, pos=nx.spring_layout(mol))
    plt.savefig(CPD_FOLDER_NAME + "/" + test_cpd.smiles + "_GNNExplainer_viz1", dpi=300, bbox_inches='tight')
    

