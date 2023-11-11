from torchvision import transforms, models
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset, Data
from torch.nn import Linear
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling, global_mean_pool, ChebConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import DataLoader

''' Message passing with edge weight learning
'''
class GNNLayer(MessagePassing):
    def __init__(self, in_channels_1,in_channels_2 ):
        super(GNNLayer, self).__init__(aggr='mean') #mean, max, min, add
        self.lin = torch.nn.Linear(in_channels_1, in_channels_2)
        self.lin1 = torch.nn.Linear(1, 512)
        self.lin2 = torch.nn.Linear(1, 4*512)
        self.lin3 = torch.nn.Linear(1, 6*512)

    def forward(self, x,z, edge_index, edge_attr):
        ''' Edge connection from Node 1 -> Node 2
        '''
        if x.shape[1] == 512:
            if z.shape[1] == 1:
                eg1 = torch.mm(z.view(1,-1), F.relu(self.lin1(edge_attr.float())))
            if z.shape[1] == 4:
                eg1 = torch.mm(z.view(1,-1), F.relu(self.lin2(edge_attr.float())).view(4,512))/4
            if z.shape[1] == 6:
                eg1 = torch.mm(z.view(1,-1), F.relu(self.lin3(edge_attr.float())).view(6,512))/6
            eg2 = torch.mm(eg1, x.T)

        ''' Edge connection from Node 2 -> Node 1
        '''

        if z.shape[1] == 512:
            if x.shape[1] == 1:
                eg1 = torch.mm(x.view(1,-1), F.relu(self.lin1(edge_attr.float())))
            if x.shape[1] == 4:
                eg1 = torch.mm(x.view(1,-1), F.relu(self.lin2(edge_attr.float())).view(4,512))/4
            if x.shape[1] == 6:
                eg1 = torch.mm(x.view(1,-1), F.relu(self.lin3(edge_attr.float())).view(6,512))/6
            eg2 = torch.mm(eg1, z.T)

        x = self.lin(x)

        return self.propagate(edge_index, x=x,edge_attr = eg2), eg2

    def message(self,x_j):
        return x_j

''' Graph neural network function
'''
class GNN(torch.nn.Module):
    def __init__(self,  num_node_features):
        super(GNN, self).__init__()
        self.gnn1 = GNNLayer(num_node_features,64)
        self.gnn2 = GNNLayer(512,64)
        self.gnn3 = GNNLayer(512,64)
        self.gnn4 = GNNLayer(1,64)
        self.gnn5 = GNNLayer(4,64)
        self.gnn6 = GNNLayer(6,64)

        self.lin1 = torch.nn.Linear(64*6, 64)
        self.lin2 = torch.nn.Linear(64, 2)

        self.edge_attr1 = torch.nn.Parameter(torch.tensor(np.random.normal(0,1,1)).reshape(1,1))
        self.edge_attr2 = torch.nn.Parameter(torch.tensor(np.random.normal(0,1,1)).reshape(1,1))
        self.edge_attr3 = torch.nn.Parameter(torch.tensor(np.random.normal(0,1,1)).reshape(1,1))

    def forward(self, x_dict, edge_index_dict):

        # Compute node embeddings using graph neural network layers
        a1,b1 = self.gnn1(x_dict.x['node_type_1'],x_dict.x['node_type_2'], edge_index_dict[('node_type_1', 'node_type_2', edge_type_1_2)], self.edge_attr1)
        a2,b2 = self.gnn2(x_dict.x['node_type_1'],x_dict.x['node_type_3'], edge_index_dict[('node_type_1', 'node_type_3', edge_type_1_3)], self.edge_attr2)
        a3,b3 = self.gnn3(x_dict.x['node_type_1'],x_dict.x['node_type_4'], edge_index_dict[('node_type_1', 'node_type_4', edge_type_1_4)], self.edge_attr3)
        a4, b4 = self.gnn4(x_dict.x['node_type_2'], x_dict.x['node_type_1'],edge_index_dict[('node_type_1', 'node_type_2', edge_type_1_2)], self.edge_attr1)
        a5, b5 = self.gnn5(x_dict.x['node_type_3'], x_dict.x['node_type_1'],edge_index_dict[('node_type_1', 'node_type_3', edge_type_1_3)], self.edge_attr2)
        a6, b6 = self.gnn6(x_dict.x['node_type_4'],x_dict.x['node_type_1'], edge_index_dict[('node_type_1', 'node_type_4', edge_type_1_4)], self.edge_attr3)

        # Perform a readout operation to obtain node-level embeddings
        x = torch.cat([a1.mean(dim=0),a4.mean(dim=0),a3.mean(dim=0), a4.mean(dim=0),a5.mean(dim=0), a6.mean(dim=0)], dim=0)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x, [b1,b2, b3, b4, b5, b6]