#import needed libraries
import yaml
import sys

from time import time, strftime, gmtime
from tqdm.auto import tqdm

import torch
from torch_geometric.data import  Data
from torch_geometric.loader import DataLoader

from pysmiles import read_smiles

from src.utils import *
from rdkit import Chem
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    start = time()
    
    args = None
    with open("parameters.yml") as paramFile:  
        args = yaml.load(paramFile, Loader=yaml.FullLoader)


    DATA_FILE       = args["trainer"]["DATA_FILE"]
    TRAIN_DATA_FILE = args["trainer"]["TRAIN_DATA_FILE"]
    VALIDATION_DATA_FILE    = args["trainer"]["VALIDATION_DATA_FILE"]
    TEST_DATA_FILE       = args["trainer"]["TEST_DATA_FILE"]
    SAVE_FOLDER_DATA_SPLIT = args["trainer"]["SAVE_FOLDER_DATA_SPLIT"]
    SMILES_FIELD_NAME    = args["trainer"]["SMILES_FIELD_NAME"]
    LABEL_FIELD_NAME       = args["trainer"]["LABEL_FIELD_NAME"]
    MODEL_SAVE_FOLDER = args["trainer"]["MODEL_SAVE_FOLDER"]
    HIDDEN_CHANNELS = args["trainer"]["HIDDEN_CHANNELS"]
    BATCH_SIZE = args["trainer"]["BATCH_SIZE"]
    EPOCHS = args["trainer"]["EPOCHS"]
    SEED    = args["trainer"]["SEED"]
        
    print("TRAIN_DATA_FILE: {}".format(TRAIN_DATA_FILE))
    print("VALIDATION_DATA_FILE: {}".format(VALIDATION_DATA_FILE))
    print("TEST_DATA_FILE: {}".format(TEST_DATA_FILE))

        
    if SEED is not None:
        set_reproducibility(SEED)

    # Load the dataset
    df_data = load_data(DATA_FILE, SMILES_FIELD_NAME, LABEL_FIELD_NAME)
    
    
    # chembl_dataset = ChEMBL(path = DATA_FILE, smiles_field = SMILES_FIELD_NAME, target_fields = target_fields)
    df = pd.read_csv(DATA_FILE)
    smiles_list = df[SMILES_FIELD_NAME].tolist()
    labels_list = df[LABEL_FIELD_NAME].tolist()
    
    #create edge index for each molecule
    
    
    mols = []
    for i in tqdm(range(len(smiles_list))):
        # Read the SMILES string 
        mols.append(read_smiles(smiles_list[i]))

    edge_index_list = []
    for mol in tqdm(mols):
        edge_index_list.append(create_edge_index(mol))

    feature_list = []
    for i in tqdm(range(len(smiles_list))):
        mol = Chem.MolFromSmiles(smiles_list[i])
        
        # Add atom features using RDKit
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
                atom.GetTotalNumHs(),
            ]
            atom_features.append(features)
        atom_features_tensor = torch.tensor(atom_features, dtype=torch.float)
        feature_list.append(atom_features_tensor)

    node_feature_dim = feature_list[0].shape[1] if feature_list else 0
    #instantiating the dataset
    data_list = []
    y = torch.LongTensor(labels_list).to(device)

    for i in tqdm(range(len(mols))):
        data_list.append(Data(x = feature_list[i], edge_index = edge_index_list[i], y = y[i], smiles = smiles_list[i]))

    dataset = ChEMBLDatasetPyG(".", data_list = data_list)

    #splitting the dataset
    train_data, val_data, test_data = [], [], []
    
    if TRAIN_DATA_FILE is None and VALIDATION_DATA_FILE is None and TEST_DATA_FILE is None:
        lengths = [int(0.8 * len(data_list)), int(0.1 * len(data_list))]
        lengths += [len(data_list) - sum(lengths)]

        dataset = dataset.shuffle()
        train_data = dataset[:lengths[0]]
        val_data = dataset[lengths[0]+1:lengths[0] + lengths[1]+1]
        test_data = dataset[lengths[0] + lengths[1] + 1:]
       
    elif TRAIN_DATA_FILE is not None and VALIDATION_DATA_FILE is not None and TEST_DATA_FILE is not None:
        print("Loading training data from {}".format(TRAIN_DATA_FILE))
        train_molecules = []
        with open(TRAIN_DATA_FILE, 'r') as f:
            train_molecules = f.read().splitlines()
        for data_sample in dataset:
            if data_sample.smiles in train_molecules:
                train_data.append(data_sample)

        print("Loading validation data from {}".format(VALIDATION_DATA_FILE))
        val_molecules = []
        with open(VALIDATION_DATA_FILE, 'r') as f:
            val_molecules = f.read().splitlines()
        for data_sample in dataset:
            if data_sample.smiles in val_molecules:
                val_data.append(data_sample)
        
        print("Loading test data from {}".format(TEST_DATA_FILE))
        test_molecules = []
        with open(TEST_DATA_FILE, 'r') as f:
            test_molecules = f.read().splitlines()
        for data_sample in dataset:
            if data_sample.smiles in test_molecules:
                test_data.append(data_sample)

        rng = np.random.default_rng(SEED)

        train_data = shuffle_list_with_numpy(train_data, rng)
        val_data = shuffle_list_with_numpy(val_data, rng)
        test_data = shuffle_list_with_numpy(test_data, rng)

    else:
        print("ERROR: Please provide either all or none of the following: TRAIN_DATA_FILE, VALIDATION_DATA_FILE, TEST_DATA_FILE.")
        sys.exit(1) 
    
    #save data split
    if SAVE_FOLDER_DATA_SPLIT is not None:
        if not os.path.exists(SAVE_FOLDER_DATA_SPLIT):
            os.makedirs(SAVE_FOLDER_DATA_SPLIT)

        with open(os.path.join(SAVE_FOLDER_DATA_SPLIT, "training.txt"), "w+") as trainFile:
            for i in range(len(train_data)):
                trainFile.write(train_data[i].smiles + "\n")
        with open(os.path.join(SAVE_FOLDER_DATA_SPLIT, "validation.txt"), "w+") as valFile:
            for i in range(len(val_data)):
                valFile.write(val_data[i].smiles + "\n")
        with open(os.path.join(SAVE_FOLDER_DATA_SPLIT, "test.txt"), "w+") as testFile:
            for i in range(len(test_data)):
                testFile.write(test_data[i].smiles + "\n")        

    #create dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    #model instantiation
    model = GCN(node_features_dim = node_feature_dim, num_classes =dataset.num_classes, hidden_channels=256).to(device)
    
    #training the network
    lr = 1e-3

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


    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    print("🚀 Training the model...")
    for epoch in range(epochs):
        train()
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        tqdm.write(f'\rEpoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}', end='')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict()

    # Save the last model 
    last_model_state = model.state_dict()
    if MODEL_SAVE_FOLDER is not None:
        save_model(model, MODEL_SAVE_FOLDER, model_name="last_model", timestamp=True)
        print(f"\n💾 Last model saved to {MODEL_SAVE_FOLDER}")

    # Load best model before testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nBest model found at epoch {best_epoch} with validation accuracy: {best_val_acc:.4f}')
        
    test_acc = test(test_loader)    
    print(f'Test accuracy with the best model: {test_acc:.4f}')

    #save the model
    if MODEL_SAVE_FOLDER is not None:
        save_model(model, MODEL_SAVE_FOLDER, model_name="best_model", timestamp=True)
        print(f"🏆 Best model saved to {MODEL_SAVE_FOLDER}")
    
    end = time()
    elapsed = end - start
    
    print("⏰ Elapsed time : {}".format(strftime("%Hh%Mm%Ss", gmtime(elapsed))))

