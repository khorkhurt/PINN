import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.optim as optim
import os


def conductivity(u):
    # u_split = torch.split(u, [2000, u.shape[0]-2000])
    # u_split = torch.split(u, [12001, 8000])
    # u_split_1 = u_split[0]
    # u_split_2 = u_split[1]
    # k1 = torch.full(size=(u_split_1.shape), fill_value=9)
    # k2 = 0.0205 * (u_split_1 - 1260) + 34.6
    # k3 = 0.6476 * (u_split_1 - 1260) + 34.6
    # k4 = torch.full(size=(u_split_2.shape), fill_value=513.8240)
    # k1 = torch.full(size=(u.shape), fill_value=9)
    k1 = 0.01 * (u - 1260) + 34.6
    k2 = 0.0205 * (u - 1260) + 34.6
    k3 = 0.6476 * (u - 1260) + 34.6
    combined = torch.cat((torch.maximum(k1, k2).reshape(-1, 1), torch.maximum(k1, k3).reshape(-1, 1),
                          torch.maximum(k2, k3).reshape(-1, 1)), 1)
    k, inds = torch.max(combined, 1)
    '''
    k_split, inds = torch.max(combined, 1)
    k = torch.cat((k_split, k4), 0)
    '''
    return k


# u_array = np.arange(-10000, 10001)
u_array = np.arange(-1000, 3001)
u_float = u_array.astype(np.float32)
u_train_tensor_unreshaped = torch.from_numpy(u_float)
k_train_tensor_unreshaped = conductivity(u_train_tensor_unreshaped)
u_train_tensor = u_train_tensor_unreshaped.reshape(-1, 1)
k_train_tensor = k_train_tensor_unreshaped.reshape(-1, 1)
print(u_train_tensor, u_train_tensor.shape)
print(k_train_tensor, k_train_tensor.shape)
plt.plot(u_train_tensor, k_train_tensor)
plt.show()

# model = nn.Sequential(nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 1),)
model = nn.Sequential(nn.Linear(1, 40), nn.Tanh(), nn.Linear(40, 40), nn.Tanh(), nn.Linear(40, 1),)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_model = nn.MSELoss(reduction='mean')
epochs = 200000

# Model Path
folder_path = "C:/Users/yasa/Desktop/PinnsLV_v3/PinnsLV"
model_path = folder_path + "/Trained_Model_for_Second_Network"
if not os.path.isdir(model_path):
    os.makedirs(model_path)
os.path.isdir(model_path)

# Loading the Model

weights = torch.load(model_path + "/model.pkl")
model.load_state_dict(weights.state_dict(), strict=False)
values1 = model(u_train_tensor)
print(values1)

'''
print("Training starts")
model.train()
losses = []
for epoch in range(epochs):
    k_prediction = model(u_train_tensor)
    loss = loss_model(k_prediction, k_train_tensor)
    print("epoch #", epoch)
    print(loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''

# Saving the Model
torch.save(model, model_path + "/model.pkl")
torch.save(model.state_dict(), model_path + "/model2.pkl")

'''
plt.plot(range(1, epochs + 1), losses)
plt.xlabel("epoch")
plt.ylabel("loss train")
plt.show()
'''

values1 = model(u_train_tensor).detach().numpy()
values2 = (k_train_tensor).detach().numpy()

plt.plot(u_train_tensor, values1, label='Neural Network')
plt.plot(u_train_tensor, values2, label='Function')
plt.xlabel('Temperature')
plt.ylabel('Conductivity')
plt.legend()
plt.savefig(model_path + "/Neural_Network_Exact_Function_Comparison.png", dpi=3000)
plt.show()
