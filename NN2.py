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
from scipy.interpolate import interp1d


def heat_capacity(u):
    cp = torch.empty(u.shape)
    '''
    mask1 = ~torch.ge(u, 25)
    cp[mask1] = 485.9975
    '''
    mask2 = torch.ge(u, -3000) & ~torch.ge(u, 2061)
    cp[mask2] = 0.3483 * u[mask2] + 477.29
    mask2 = torch.ge(u, 25) & ~torch.ge(u, 1260)
    cp[mask2] = 0.3483 * u[mask2] + 477.29
    mask3 = torch.ge(u, 1260) & ~torch.ge(u, 1460)
    cp[mask3] = 0.3483 * u[mask3] + 1857.3
    '''
    mask4 = torch.ge(u, 1460) & ~torch.ge(u, 2061)
    cp[mask4] = 0.3483 * u[mask4] + 477.29
    '''
    mask4 = torch.ge(u, 1460) & ~torch.ge(u, 10000)
    cp[mask4] = 0.3483 * u[mask4] + 477.29
    # mask5 = torch.ge(u, 2061)
    # cp[mask5] = 1195.1363
    return cp


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

# u_array = np.arange(0, 2061)
u_array = np.arange(0, 3000)
u_float = u_array.astype(np.float32)
u_train_tensor_unreshaped = torch.from_numpy(u_float)
cp_train_tensor_unreshaped = heat_capacity(u_train_tensor_unreshaped)
u_train_tensor = u_train_tensor_unreshaped.reshape(-1, 1)
cp_train_tensor = cp_train_tensor_unreshaped.reshape(-1, 1)
print(u_train_tensor, u_train_tensor.shape)
print(cp_train_tensor, cp_train_tensor.shape)
np.savetxt("u.csv", u_float, fmt='%1.3f')

cp_train_numpy = cp_train_tensor_unreshaped.detach().numpy()
np.savetxt("cp.csv", cp_train_numpy, fmt='%1.3f')
cp_train_smoothened_unreshaped = savitzky_golay(cp_train_numpy, 51, 2) # window size 51, polynomial order 3
plt.plot(u_train_tensor, cp_train_tensor)
plt.plot(u_train_tensor, cp_train_smoothened_unreshaped)
plt.show()
quit()
cp_train_smoothened_unreshaped_float = cp_train_smoothened_unreshaped.astype(np.float32)
cp_train_smoothened = torch.from_numpy(cp_train_smoothened_unreshaped_float).reshape(-1, 1)


# model = nn.Sequential(nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 1),)
model = nn.Sequential(nn.Linear(1, 100), nn.Tanh(), nn.Linear(100, 100), nn.Tanh(), nn.Linear(100, 1),)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_model = nn.MSELoss(reduction='mean')
epochs = 150000
'''
optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter, max_eval=50000, history_size=100,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
'''

# Model Path
folder_path = "C:/Users/yasa/Desktop/PinnsLV_v3/PinnsLV"
model_path = folder_path + "/Trained_Model_for_Third_Network"
if not os.path.isdir(model_path):
    os.makedirs(model_path)
os.path.isdir(model_path)

# Loading the Model

weights = torch.load(model_path + "/model.pkl")
model.load_state_dict(weights.state_dict(), strict=False)
values1 = model(u_train_tensor)
# print(values1)

'''
print("Training starts")
model.train()
losses = []
for epoch in range(epochs):
    cp_prediction = model(u_train_tensor)
    loss = loss_model(cp_prediction, cp_train_smoothened)
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
plt.xlabel("iteration")
plt.ylabel("loss train")
plt.show()
'''

values1 = model(u_train_tensor).detach().numpy()
values2 = (cp_train_smoothened).detach().numpy()

plt.plot(u_train_tensor, values1, label='Neural Network')
plt.plot(u_train_tensor, values2, label='Function')
plt.xlabel('Temperature')
plt.ylabel('Heat Capacity')
plt.legend()
plt.savefig(model_path + "/Neural_Network_Exact_Function_Comparison.png", dpi=3000)
plt.show()
