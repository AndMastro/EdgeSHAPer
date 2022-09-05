#import needed libraries
import argparse

import random
import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from torchdrug import data
from torchdrug.core import Registry as R
from torchdrug.utils import doc
from pysmiles import read_smiles


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_usage():
    print(' ')
    print('usage: python trainer_script.py --DATA_PATH --TRAIN_DATA_PATH --VALIDATION_DATA_PATH --TEST_DATA_PATH --SMILES_FIELD_NAME --LABEL_FIELD_NAME --MODEL_SAVE_PATH --SEED')
    print('-----------------------------------------------------------------')
    print('DATA_PATH:path in which your .csv dataset file is located.')
    print('   (default: "experiments/data/chembl29_predicting_target_P14416_P42336_target_1_vs_random_cpds.csv."')
    print('TRAIN_DATA_PATH :location in which your training .txt file is located.')
    print('  (default: "experiments/data/train_val_test_splits/P14416_P42336_target_1_vs_random_cpds/training.txt")')
    print('VALIDATION_DATA_PATH :location in which your validation .txt file is located (optional). If this is not provided, the validation set will be obtained as the 10% of the training set')
    print('  (default: "experiments/data/train_val_test_splits/P14416_P42336_target_1_vs_random_cpds/validation.txt")')
    print('TEST_DATA_PATH :location in which your test .txt file is located.')
    print('  (default: "experiments/data/train_val_test_splits/P14416_P42336_target_1_vs_random_cpds/test.txt")')
    print('SMILES_FIELD_NAME :column name for the SMILES field.')
    print('LABEL_FIELD_NAME :column name for the label field.')
    print('MODEL_SAVE_PATH :location in which the trained model will be saved.')
    print('  (default: "experiments/models")')
    print('HIDDEN_CHANNELS :number of hidden channels for the GCN.')
    print('  (default: 256)')
    print('BATCH_SIZE :batch size for the training.')
    print('  (default: 32)')
    print('EPOCHS :number of epochs for which the model will be trained.')
    print('  (default: 100)')
    print('SEED :seed for the random number generator for reproducible results.')
    print(' (optional, default: None)')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set needed arguments for the training script.')
    parser.add_argument('--DATA_PATH', type=str, default="experiments/data/chembl29_predicting_target_P14416_P42336_target_1_vs_random_cpds.csv.",
                    help='path in which your .csv dataset file is located (default: "experiments/data/chembl29_predicting_target_P14416_P42336_target_1_vs_random_cpds.csv.\\"')
    parser.add_argument('--TRAIN_DATA_PATH', type=str, default=None,
                    help='location in which your training .txt file is located (default: None).\\')
    parser.add_argument('--VALIDATION_DATA_PATH', type=str, default=None,
                    help='location in which your validation .txt file is located (default: None).\\')
    parser.add_argument('--TEST_DATA_PATH', type=str, default=None,
                    help='location in which your test .txt file is located(default: None).\\')
    parser.add_argument('--SMILES_FIELD_NAME', type=str, default=None,
                help='column name for the SMILES field\\')
    parser.add_argument('--LABEL_FIELD_NAME', type=str, default=None,
                help='column name for the label field\\')
    parser.add_argument('--MODEL_SAVE_PATH', type=str, default="experiments/models",
                help='location in which the trained model will be saved. (default: "experiments/models")\\.')
    parser.add_argument('--HIDDEN_CHANNELS', type=int, default=32,
        help='HIDDEN_CHANNELS :number of hidden channels for the GCN (default: 256).\\')            
    parser.add_argument('--BATCH_SIZE', type=int, default=32,
        help='BATCH_SIZE :batch size for the training (default: 32).\\')
    parser.add_argument('--EPOCHS', type=int, default=100,
        help='EPOCHS :number of epochs for which the model will be trained (default: 100).\\')
    parser.add_argument('--SEED', type=int, default=None,
    help='seed for the random number generator (default: 42).\\')               

    return parser.parse_args()

def set_all_seeds(SEED):
    '''
    Set all seeds for reproducibility.
    '''
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    random.seed(SEED)

def load_data(DATA_PATH, SMILES_FIELD_NAME, LABEL_FIELD_NAME):
    '''
    Load the data from a .csv file.
    '''
    df_data = pd.read_csv(DATA_PATH, sep = ",")
    df_data = df_data[[SMILES_FIELD_NAME, LABEL_FIELD_NAME]]
    
    return df_data


def save_model(model, MODEL_SAVE_PATH):
    '''
    Save the trained model.
    '''
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d-%H_%M_%S")

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    MODEL_PATH = MODEL_SAVE_PATH + "/model_" + current_time + ".pt"
    torch.save(model.state_dict(), MODEL_PATH)

def create_edge_index(mol):
    """
    Create edge index for a molecule.
    """
    adj = nx.to_scipy_sparse_matrix(mol).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

@R.register("datasets.ChEMBL")
class ChEMBL(data.MoleculeDataset):
    '''
    Class for the molecule dataset.
    '''
    def __init__(self, path, smiles_field, target_fields, verbose=1, **kwargs):
    
        self.path = path
        self.smiles_field = smiles_field
        self.target_fields= target_fields

        self.load_csv(self.path, smiles_field=self.smiles_field, target_fields=self.target_fields,
                    verbose=verbose, **kwargs)

class ChEMBLDatasetPyG(InMemoryDataset):
    '''
    Class for the PyG version of the dataset.
    '''
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, data_list = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data_list = data_list

        data_list = self.data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)

class GCN(torch.nn.Module):
    '''
    4-layer GCN model class.
    '''
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(chembl_dataset.node_feature_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x.float(), edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

if __name__ == "__main__":
    args = parse_args()

    DATA_PATH       = args.DATA_PATH
    TRAIN_DATA_PATH = args.TRAIN_DATA_PATH
    VALIDATION_DATA_PATH    = args.VALIDATION_DATA_PATH
    TEST_DATA_PATH       = args.TEST_DATA_PATH
    TRAIN_DATA_PATH = args.TRAIN_DATA_PATH
    SMILES_FIELD_NAME    = args.SMILES_FIELD_NAME
    LABEL_FIELD_NAME       = args.LABEL_FIELD_NAME
    MODEL_SAVE_PATH = args.MODEL_SAVE_PATH
    HIDDEN_CHANNELS = args.HIDDEN_CHANNELS
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS
    SEED    = args.SEED

    if not len(sys.argv) > 1:
        print_usage()
        print('ERROR: Not enough arguments provided.')
        sys.exit(1)
        

    if SMILES_FIELD_NAME is None:
        print_usage()
        print('ERROR: SMILES_FIELD_NAME is not provided.')
        exit(1)

    if LABEL_FIELD_NAME is None:
        print_usage()
        print('ERROR: LABEL_FIELD_NAME is not provided.')
        exit(1)
        
    if SEED is not None:
        set_all_seeds(SEED)

    # Load the dataset
    df_data = load_data(DATA_PATH, SMILES_FIELD_NAME, LABEL_FIELD_NAME)

    # instantiate custom class from TorchDrug
    target_fields = [LABEL_FIELD_NAME]
    chembl_dataset = ChEMBL(path = DATA_PATH, smiles_field = SMILES_FIELD_NAME, target_fields = target_fields)

    #create edge index for each molecule
    smiles = chembl_dataset.smiles_list
    mols = []
    for i in tqdm(range(len(smiles))):
        mols.append(read_smiles(smiles[i]))

    edge_index_list = []
    for mol in tqdm(mols):
        edge_index_list.append(create_edge_index(mol))

    mols_torchdrug_format = []
    for i in tqdm(range(len(smiles))):
        mols_torchdrug_format.append(data.Molecule.from_smiles(smiles[i], with_hydrogen = False))

    #instantiating the dataset
    data_list = []
    y = torch.LongTensor(chembl_dataset.targets[LABEL_FIELD_NAME]).to(device)

    for i in tqdm(range(len(mols))):
        data_list.append(Data(x = mols_torchdrug_format[i].node_feature, edge_index = edge_index_list[i], y = y[i], smiles = chembl_dataset.smiles_list[i]))

    dataset = ChEMBLDatasetPyG(".", data_list = data_list)

    #splitting the dataset
    train_data, val_data, test_data = None, None, None
    
    if TRAIN_DATA_PATH is None:
        lengths = [int(0.8 * len(chembl_dataset)), int(0.1 * len(chembl_dataset))]
        lengths += [len(chembl_dataset) - sum(lengths)]

        dataset = dataset.shuffle()
        train_data = dataset[:lengths[0]]
        val_data = dataset[lengths[0]+1:lengths[0] + lengths[1]+1]
        test_data = dataset[lengths[0] + lengths[1] + 1: ]
       
    else:
        #TBD load from files
        pass
    #create dataloaders

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    #model instantiation
    model = GCN(hidden_channels=256).to(device)
    
    #training the network
    lr = lr=1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = EPOCHS
    criterion = torch.nn.CrossEntropyLoss() # this is equivalent to the combination of LogSoftmax torch.nn.NLLLoss.

    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    for epoch in range(epochs):
        train()
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    test_acc = test(test_loader)    
    print(f'Test Acc: {test_acc:.4f}')

    #save the model
    if MODEL_SAVE_PATH is not None:
        save_model(model, MODEL_SAVE_PATH)

