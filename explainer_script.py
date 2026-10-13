### Andrea Mastropietro 2022 © All rights reserved ###
### This script is used to explain molecules using a pretrained model.
### It uses the Edgeshaper explainer to compute Shapley values for edges in the molecular graph.

import os
import sys
from time import time, strftime, gmtime


import yaml
import pandas as pd
from tqdm.auto import tqdm
from pysmiles import read_smiles

import torch
from torch_geometric.data import Data

# Custom module imports
from src.utils import *
from src.edgeshaper import *
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    start = time()
    
    args = None
    with open("parameters.yml") as paramFile:
        args = yaml.load(paramFile, Loader=yaml.FullLoader)


    MODEL_PATH = args["explainer"]["MODEL"]
    DATA_FILE = args["explainer"]["DATA_FILE"]
    TEST_DATA_FILE = args["explainer"]["TEST_DATA_FILE"]
    MOLECULES_TO_EXPLAIN = args["explainer"]["MOLECULES_TO_EXPLAIN"]
    TARGET_CLASS = args["explainer"]["TARGET_CLASS"]
    SMILES_FIELD_NAME = args["explainer"]["SMILES_FIELD_NAME"]
    LABEL_FIELD_NAME = args["explainer"]["LABEL_FIELD_NAME"]
    MINIMAL_SETS = args["explainer"]["MINIMAL_SETS"]
    SAVE_FOLDER_PATH = args["explainer"]["SAVE_FOLDER_PATH"]
    SAMPLING_STEPS = args["explainer"]["SAMPLING_STEPS"]
    HIDDEN_CHANNELS = args["explainer"]["HIDDEN_CHANNELS"]
    VISUALIZATION = args["explainer"]["VISUALIZATION"]
    TOLERANCE = args["explainer"]["TOLERANCE"]
    SEED = args["explainer"]["SEED"]
    
    
    if SEED is not None:
        set_reproducibility(SEED)

    # load data

    df_data = load_data(DATA_FILE, SMILES_FIELD_NAME, LABEL_FIELD_NAME)
    
    df = pd.read_csv(DATA_FILE)
    smiles_list = df[SMILES_FIELD_NAME].tolist()
    labels_list = df[LABEL_FIELD_NAME].tolist()

    #create edge index for each molecule
    
    mols = []
    for i in tqdm(range(len(smiles_list))):
        mols.append(read_smiles(smiles_list[i]))

    edge_index_list = []
    for mol in tqdm(mols):
        edge_index_list.append(create_edge_index(mol))

    #check if those steps are necessary!!!!### we could get rid of the CHEMbl dataset
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

    test_data = []
    if TEST_DATA_FILE is None:
        lengths = [int(0.8 * len(data_list)), int(0.1 * len(data_list))]
        lengths += [len(data_list) - sum(lengths)]

        dataset = dataset.shuffle()
        train_data = dataset[:lengths[0]]
        val_data = dataset[lengths[0]+1:lengths[0] + lengths[1]+1]
        test_data = dataset[lengths[0] + lengths[1] + 1: ]
       
    else:
        test_molecules = []
        with open(TEST_DATA_FILE, 'r') as f:
            test_molecules = f.read().splitlines()
        for data_sample in dataset:
            if data_sample.smiles in test_molecules:
                test_data.append(data_sample)

        rng = np.random.default_rng(SEED)
        test_data = shuffle_list_with_numpy(test_data, rng)

    # load model
    model = GCN(node_features_dim = node_feature_dim, num_classes = dataset.num_classes, hidden_channels=HIDDEN_CHANNELS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)

    #read list of molecules to explain
    molecules_to_explain = []
    test_cpd_indices = []
    if isinstance(MOLECULES_TO_EXPLAIN, int):
        print("ℹ️ Selecting {} test molecules predicted as class 0 (active) by the model to be explained...".format(MOLECULES_TO_EXPLAIN))
        # Select the first MOLECULES_TO_EXPLAIN molecules predicted as class 0 (active) by the model
        if MOLECULES_TO_EXPLAIN <= 0:
            raise ValueError("MOLECULES_TO_EXPLAIN must be a positive integer.")
        model.eval()
        count = 0
        with torch.no_grad():
            for i, data in enumerate(test_data):

                if data.y == 0:  
                    data = data.to(device)
                    batch = torch.zeros(data.x.shape[0], dtype=int, device=data.x.device)
                    out = model(data.x, data.edge_index, batch=batch)
                    out_prob = F.softmax(out, dim = 1)

                    # print("Compound: ", data.smiles, " - Out prob: ", out_prob, " - Predicted class: ", torch.argmax(out_prob[0]).item())

                    pred = torch.argmax(out_prob[0]).item()
                    if pred == 0:
                        molecules_to_explain.append(data.smiles)
                        count += 1
                        test_cpd_indices.append(i)
                        if count >= MOLECULES_TO_EXPLAIN:
                            break
    elif os.path.exists(MOLECULES_TO_EXPLAIN):
        with open(MOLECULES_TO_EXPLAIN, 'r') as f:
            molecules_to_explain = f.read().splitlines()
    else:
        raise ValueError("MOLECULES_TO_EXPLAIN must be an integer or a valid file path.")

    print("Molecules to explain: ", molecules_to_explain)
    

    fidelities = []
    infidelities = []
    #explain the molecules
    for test_index in tqdm(test_cpd_indices):

        print("🔍 Explaining test compound: ", test_data[test_index].smiles)
        assert test_data[test_index].smiles in molecules_to_explain, "The test compound smiles is not in the list of molecules to explain."

        test_cpd = test_data[test_index].to(device)

        edgeshaper_explainer = Edgeshaper(model, test_cpd.x, test_cpd.edge_index, device = device)
        phi_edges = edgeshaper_explainer.explain(M = SAMPLING_STEPS, target_class = TARGET_CLASS, P = None, deviation = TOLERANCE, log_odds = False, seed = SEED, progress_bar = False)
        original_prob = edgeshaper_explainer.compute_original_predicted_probability()

        if SAVE_FOLDER_PATH is not None:
            SAVE_FOLDER_PATH_COMPLETE = SAVE_FOLDER_PATH + "/"  + test_cpd.smiles
            if not os.path.exists(SAVE_FOLDER_PATH_COMPLETE):
                os.makedirs(SAVE_FOLDER_PATH_COMPLETE)
            INFO_EXPLANATIONS = SAVE_FOLDER_PATH_COMPLETE + "/info_explanations.txt"
            

            with open(INFO_EXPLANATIONS, "w+") as saveFile:
                saveFile.write("Explaining class " + str(TARGET_CLASS) + " for compound: " + test_cpd.smiles + "\n\n")

                saveFile.write("Shapley values for edges:\n")
                for i in range(len(phi_edges)):
                    saveFile.write("(" + str(test_cpd.edge_index[0][i].item()) + "," + str(test_cpd.edge_index[1][i].item()) + "): " + str(phi_edges[i]) + "\n")

                saveFile.write("\nSum of Shapley values: " + str(sum(phi_edges)) + "\n\n")
                saveFile.write("Original predicted probability: " + str(original_prob) + "\n\n")
                

        if MINIMAL_SETS:
            pert_pos, inf = edgeshaper_explainer.compute_pertinent_positive_set(verbose=True)
            min_top_k, fid = edgeshaper_explainer.compute_minimal_top_k_set(verbose=True)

            fidelities.append(fid)
            infidelities.append(inf)

            with open(INFO_EXPLANATIONS, "a+") as saveFile:
                saveFile.write("Minimal top k set edge index:\n")
                saveFile.write(str(min_top_k.tolist()) + "\n\n")
                saveFile.write("FID+: " + str(fid) + "\n\n")

                saveFile.write("Pertinent positive set edge index:\n")
                saveFile.write(str(pert_pos.tolist()) + "\n\n")
                saveFile.write("FID-: " + str(inf) + "\n\n")

        if VISUALIZATION:
            VISUALIZATION_SAVE_FOLDER_PATH_COMPLETE = SAVE_FOLDER_PATH + "/"  + test_cpd.smiles

            if not os.path.exists(VISUALIZATION_SAVE_FOLDER_PATH_COMPLETE):
                os.makedirs(VISUALIZATION_SAVE_FOLDER_PATH_COMPLETE)

            edgeshaper_explainer.visualize_molecule_explanations(test_cpd.smiles, save_path = VISUALIZATION_SAVE_FOLDER_PATH_COMPLETE, pertinent_positive=True, minimal_top_k=True)

    if MINIMAL_SETS:
        print("📈 Average FID+: ", sum(fidelities) / len(fidelities))
        print("📉 Average FID-: ", sum(infidelities) / len(infidelities))

    end = time()
    elapsed = end - start 
    
    print("\n⏰Elapsed time : {}".format(strftime("%Hh%Mm%Ss", gmtime(elapsed))))