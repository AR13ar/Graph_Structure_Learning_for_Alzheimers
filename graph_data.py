import numpy as np
import os
import fnmatch
import glob
from medpy.io import load
import cv2
from tqdm import tqdm
import time
import pickle as pkl
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

''' Load Data, APoe, psen1 , psen2 and imaging features
'''

file_path_save = "Path to imaging features"
with open(file_path_save + "//" + "FIle Name", "rb") as fp:
  features = pkl.load(fp)

apoe = pd.read_csv(file_path_save + "//" + "APOE file")

psen1 = pd.read_csv(file_path_save + "//" + "PSEN1 file")

psen2 = pd.read_csv(file_path_save + "//" + "PSEN2 file")

dt_features = []
labels = []

for i in range(len(features)):
    for j in range(len(features)):
        if filenames[0][i] == features[j][1][0]:
            dt_features.append(features[j][0])
            labels.append(features[j][2])


dt_features = torch.tensor(dt_features).squeeze()
dt_apoe = torch.tensor(apoe.iloc[:,1:].T.values.tolist())
dt_psen1 = torch.tensor(psen1.iloc[:,1:].T.values.tolist())
dt_psen2 = torch.tensor(psen2.iloc[:,1:].T.values.tolist())

''' Make Graph data this is for all 4 Nodes
'''
data_list = []
for i in range(len(features)):
    num_nodes_per_type = {'node_type_1': 1, 'node_type_2':1, 'node_type_3': 1, 'node_type_4': 1}
    num_nodes = {}
    x_dict = {}
    for k in num_nodes_per_type.keys():

        if k == 'node_type_1':
            x_dict[k] = dt_features[i].reshape(1,-1)
        elif k == 'node_type_2':
            x_dict[k] = dt_apoe[i].reshape(1,-1)
        elif k == 'node_type_3':
            x_dict[k] = dt_psen1[i].reshape(1,-1)
        elif k == 'node_type_4':
            x_dict[k] = dt_psen2[i].reshape(1,-1)

    edge_index_dict = {}
    edge_type_1_2, edge_type_1_3, edge_type_1_4 = 0, 1, 2
    edge_index = torch.tensor([[0], [0]])
    edge_index_dict[('node_type_1', 'node_type_2', edge_type_1_2)] = edge_index
    edge_index_dict[('node_type_1', 'node_type_3', edge_type_1_3)] = edge_index
    edge_index_dict[('node_type_1', 'node_type_4', edge_type_1_4)] = edge_index

    y = {'node_type_1':labels[i] }

    data = Data( edge_index=edge_index_dict,x=x_dict, y = y, edge_attr=None, num_nodes = 1)
    data_list.append(data)

data_list = shuffle(data_list)
data_list_train, data_list_test = train_test_split(data_list, test_size=0.2,
                                                    random_state= 25)
data_loader = DataLoader(data_list_train, batch_size=1, shuffle=True)

