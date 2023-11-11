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
import torch.optim as optim
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score

''' Initialize GNN
'''
gnn = GNN(512)
optimizer = optim.Adam(gnn.parameters(), lr = 0.001)
criterion =  nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()

weights = []
''' Model Training
'''
for epoch in range(800):
    running_loss = 0.0
    loss = 0
    for b in data_loader:
            # Forward pass

            y_true = b.y['node_type_1']
            yy = tf.keras.utils.to_categorical(y_true, num_classes=2)
            y_pred, weis = gnn(b, b.edge_index)
            loss_mse = criterion2(y_pred, torch.tensor(yy).squeeze())
            loss +=  criterion(y_pred.reshape(1,-1), torch.tensor(y_true))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    weights.append(weis)
    running_loss += loss.item()
    print('Epoch {}, Loss: {:.4f}'.format(epoch + 1, running_loss / len(data)))

print('Training finished!')

test_pred = []
test_true = []
wei_test = []
for test in data_list_test:
    test_out, wei_t = gnn(test, test.edge_index)
    test_pred.append(test_out.detach())
    test_true.append(test.y['node_type_1'])
    wei_test.append(wei_t)

test_p = []
for i in range(len(test_pred)):
    test_p.append(list(test_pred[i]).index(test_pred[i].max()))

target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(test_true, test_p, target_names=target_names))
print(metrics.confusion_matrix(test_true, test_p))
print("Precision score: {}".format(precision_score(test_true,test_p)))
print("Recall score: {}".format(recall_score(test_true,test_p)))
print("F1 Score: {}".format(f1_score(test_true,test_p)))

''' Compute average weights per sample
'''
wei_posi = []
wei_nega = []
wei_2 = []
for i in range(len(test_true)):
    if test_true[i] == 1:
        wei_posi.append(wei_test[i])
    elif test_true[i] == 0:
        wei_nega.append(wei_test[i])
    else:
        wei_2.append(wei_test[i])

wei_posi2 = []
for i in range(len(wei_posi)):
    po = []
    for j in wei_posi[i]:
        po.append(j.detach().numpy())
    wei_posi2.append(po)
wei_posi2 = np.array(wei_posi2).squeeze()

wei_nega2 = []
for i in range(len(wei_nega)):
        ne = []
        for j in wei_nega[i]:
            ne.append(j.detach().numpy())
        wei_nega2.append(ne)
wei_nega2 = np.array(wei_nega2).squeeze()


wei_22 = []
for i in range(len(wei_2)):
        ne2 = []
        for j in wei_2[i]:
            ne2.append(j.detach().numpy())
        wei_22.append(ne2)
wei_22 = np.array(wei_22).squeeze()

mean_posi = np.mean(wei_posi2, axis = 0)
mean_nega = np.mean(wei_nega2, axis = 0)
mean_2 = np.mean(wei_22, axis = 0)

print("mean_positive", mean_posi)
print("mean_negative", mean_nega)
print("mean_2", mean_2)