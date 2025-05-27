### Andrea Mastropietro © all rights reserve
# run this script to obtain explanations for given molecules using a pretrained model ###

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_usage():
    print(' ')
    print('usage: python explainer_script.py --MODEL_PATH --DATA_FILE --MOLECULES_TO_EXPLAIN --TARGET_CLASS --MINIMAL_SETS --SAVE_FOLDER_PATH --SAMPLING_STEPS --VISUALIZATION_SAVE_FOLDER_PATH --TOLERANCE --SEED')
    print('-----------------------------------------------------------------')
    print('MODEL_PATH: path in which your model is located.')

    print('DATA_FILE: path in which your .csv dataset file is located.')
    print('    default: "experiments/data/chembl29_predicting_target_P14416_P42336_target_1_vs_random_cpds.csv."')

    print('MOLECULES_TO_EXPLAIN: path in which your .txt file with the molecules to explain is located.')
    print('    default: "TBD"')

    print('TARGET_CLASS: target class for which the explanations will be computed.')
    print('    default: 0')

    print('SMILES_FIELD_NAME :column name for the SMILES field.')
    print('LABEL_FIELD_NAME :column name for the label field.')

    print('MINIMAL_SETS: boolean indicating whether to compute minimal informative sets.')
    print('    default: False')

    print('SAVE_FOLDER_PATH: path in which the explanations will be saved.')
    print('    default: "results"')

    print('SAMPLING_STEPS: number of Monte Carlo sampling steps to perform.')
    print('    default: 100')

    print('VISUALIZATION: if to sve visualizations.')
    print('    (optional, default: False)')

    print('TOLERANCE: desired deviation between predicted probability and sum of Shapley values.')
    print('    (optional, default: None')

    print('SEED: seed for the random number generator.')
    print('    (optional, default: None')
   


if __name__ == "__main__":
    start = time()
    
    args = None
    with open("parameters.yml") as paramFile:
        args = yaml.load(paramFile, Loader=yaml.FullLoader)


    MODEL_PATH = args["explainer"]["MODEL"]
    DATA_FILE = args["explainer"]["DATA_FILE"]
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
    
    

    if MODEL_PATH is None:
        print_usage()
        print('ERROR: No model path provided.')
        sys.exit(1)
    
    if SEED is not None:
        set_all_seeds(SEED)

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

    # load model
    model = GCN(node_features_dim = node_feature_dim, num_classes =dataset.num_classes, hidden_channels=HIDDEN_CHANNELS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)

    #read list of molecules to explain

    with open(MOLECULES_TO_EXPLAIN, 'r') as f:
        molecules_to_explain = f.read().splitlines()
    
    test_cpd_indices = []
    for molecule in molecules_to_explain:
        test_cpd_indices.append(smiles_list.index(molecule)) #check if this is correct

    fidelities = []
    infidelities = []
    #explain the molecules
    for test_index in test_cpd_indices:
        #good idea to define an explainer class, see if we want to implement it
        # explainer = Explainer(model, dataset, test_index, SAMPLING_STEPS, TOLERANCE)
        # explanation = explainer.explain()
        # if MINIMAL_SETS:
        #     explanation = explainer.compute_minimal_sets(explanation)
        # if SAVE_FOLDER_PATH is not None:
        #     explainer.save_explanation(explanation, SAVE_FOLDER_PATH)
        # if VISUALIZATION_SAVE_FOLDER_PATH is not None:
        #     explainer.visualize(explanation, VISUALIZATION_SAVE_FOLDER_PATH)

        print("Explaining test compound: ", dataset[test_index].smiles)
        test_cpd = dataset[test_index].to(device)

        # phi_edges = edgeshaper(model, test_cpd.x, test_cpd.edge_index, M = SAMPLING_STEPS, target_class = TARGET_CLASS, P = None, deviation = TOLERANCE, log_odds = False, seed = SEED, device = device)
        edgeshaper_explainer = Edgeshaper(model, test_cpd.x, test_cpd.edge_index, device = device)
        phi_edges = edgeshaper_explainer.explain(M = SAMPLING_STEPS, target_class = TARGET_CLASS, P = None, deviation = TOLERANCE, log_odds = False, seed = SEED)
        # print("Shapley values for edges: ", phi_edges)

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

            # visualize_explanations(test_cpd, phi_edges, VISUALIZATION_SAVE_FOLDER_PATH_COMPLETE)
            edgeshaper_explainer.visualize_molecule_explanations(test_cpd.smiles, save_path = VISUALIZATION_SAVE_FOLDER_PATH_COMPLETE, pertinent_positive=True, minimal_top_k=True)

    if MINIMAL_SETS:
        print("\nAverage FID+: ", sum(fidelities) / len(fidelities))
        print("Average FID-: ", sum(infidelities) / len(infidelities))

    end = time()
    elapsed = end - start 
    
    print("\nElapsed time : {}".format(strftime("%Hh%Mm%Ss", gmtime(elapsed))))