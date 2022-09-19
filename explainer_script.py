### run this script to obtain explanations for given molecules using a pretrained model ###
import argparse
import os
import sys
import yaml

from time import time, strftime, gmtime

import torch
from torch_geometric.data import  Data
from torchdrug import data

from pysmiles import read_smiles

from tqdm import tqdm

#custom modules
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
   

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set needed arguments for the explainer script.')

    parser.add_argument('--MODEL_PATH', type=str, required=True,
            help='path in which your model is located')   

    parser.add_argument('--DATA_FILE', type=str, default="experiments/data/chembl29_predicting_target_P14416_P42336_target_1_vs_random_cpds.csv.",
                    help='path in which your .csv dataset file is located (default: "experiments/data/chembl29_predicting_target_P14416_P42336_target_1_vs_random_cpds.csv.\\"')

    parser.add_argument('--MOLECULES_TO_EXPLAIN', type=str, required=True, help='path in which your .txt file with the molecules to explain is located (default: "TBD")')

    parser.add_argument('--TARGET_CLASS', type=int, default=0, help='target class for which the explanations will be computed (default: 0)')

    parser.add_argument('--SMILES_FIELD_NAME', type=str, default=None, required=True,
                help='column name for the SMILES field\\')

    parser.add_argument('--LABEL_FIELD_NAME', type=str, default=None,required=True,
                help='column name for the label field\\')

    parser.add_argument('--MINIMAL_SETS', type=bool, default=False, help='boolean indicating whether to compute minimal informative sets (default: False)')

    parser.add_argument('--SAVE_FOLDER_PATH', type=str, default="results", help='path in which the explanations will be saved (optional, default: "TBD")')

    parser.add_argument('--SAMPLING_STEPS', type=int, default=100, help='number of Monte Carlo sampling steps to perform (default: 100)')

    parser.add_argument('--VISUALIZATION', type=bool, default=False, help=' if to save visualization (optional, default: False)')

    parser.add_argument('--TOLERANCE', type=float, default=None, help='desired deviation between predicted probability and sum of Shapley values (optional, default: None)')

    parser.add_argument('--SEED', type=int, default=None, help='seed for the random number generator (optional, default: None)')
    return parser.parse_args()


if __name__ == "__main__":
    start = time()
    # parse arguments

    # args = parse_args()

    # MODEL_PATH = args.MODEL_PATH
    # DATA_FILE = args.DATA_FILE
    # MOLECULES_TO_EXPLAIN = args.MOLECULES_TO_EXPLAIN
    # TARGET_CLASS = args.TARGET_CLASS
    # SMILES_FIELD_NAME = args.SMILES_FIELD_NAME
    # LABEL_FIELD_NAME = args.LABEL_FIELD_NAME
    # MINIMAL_SETS = args.MINIMAL_SETS
    # SAVE_FOLDER_PATH = args.SAVE_FOLDER_PATH
    # SAMPLING_STEPS = args.SAMPLING_STEPS
    # VISUALIZATION = args.VISUALIZATION
    # TOLERANCE = args.TOLERANCE
    # SEED = args.SEED
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
    
    # if not len(sys.argv) > 1:
    #     print_usage()
    #     print('ERROR: Not enough arguments provided.')
    #     sys.exit(1)

    if MODEL_PATH is None:
        print_usage()
        print('ERROR: No model path provided.')
        sys.exit(1)
    
    if SEED is not None:
        set_all_seeds(SEED)

    # load data

    df_data = load_data(DATA_FILE, SMILES_FIELD_NAME, LABEL_FIELD_NAME)
    
    target_fields = [LABEL_FIELD_NAME]
    chembl_dataset = ChEMBL(path = DATA_FILE, smiles_field = SMILES_FIELD_NAME, target_fields = target_fields)

    #create edge index for each molecule
    smiles = chembl_dataset.smiles_list
    mols = []
    for i in tqdm(range(len(smiles))):
        mols.append(read_smiles(smiles[i]))

    edge_index_list = []
    for mol in tqdm(mols):
        edge_index_list.append(create_edge_index(mol))

    #check if those steps are necessary!!!!### we could get rid of the CHEMbl dataset
    mols_torchdrug_format = []
    for i in tqdm(range(len(smiles))):
        mols_torchdrug_format.append(data.Molecule.from_smiles(smiles[i], with_hydrogen = False))

    #instantiating the dataset
    data_list = []
    y = torch.LongTensor(chembl_dataset.targets[LABEL_FIELD_NAME]).to(device)

    for i in tqdm(range(len(mols))):
        data_list.append(Data(x = mols_torchdrug_format[i].node_feature, edge_index = edge_index_list[i], y = y[i], smiles = chembl_dataset.smiles_list[i]))

    dataset = ChEMBLDatasetPyG(".", data_list = data_list)
    # load model
    model = GCN(node_features_dim = chembl_dataset.node_feature_dim, num_classes =dataset.num_classes, hidden_channels=HIDDEN_CHANNELS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)

    #read list of molecules to explain

    with open(MOLECULES_TO_EXPLAIN, 'r') as f:
        molecules_to_explain = f.read().splitlines()
    
    test_cpd_indices = []
    for molecule in molecules_to_explain:
        test_cpd_indices.append(chembl_dataset.smiles_list.index(molecule)) #check if this is correct

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
            pert_pos, inf = edgeshaper_explainer.compute_pertinent_positivite_set(verbose=True)
            min_top_k, fid = edgeshaper_explainer.compute_minimal_top_k_set(verbose=True)

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
            edgeshaper_explainer.visualize_molecule_explanations(test_cpd.smiles, SAVE_FOLDER_PATH = VISUALIZATION_SAVE_FOLDER_PATH_COMPLETE, pertinent_positive=True, minimal_top_k=True)

    end = time()
    elapsed = end - start
    
    print("Elapsed time : {}".format(strftime("%Hh%Mm%Ss", gmtime(elapsed))))