from ImportFile import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
folder_path = "Paper_Model"
# Plotting ang Assessing Performance
images_path = folder_path + "/Images"
model_path = folder_path + "/TrainedModel/"

model = torch.load(model_path + "/model3.pkl", map_location=torch.device('cpu'))
model = model.eval()
umax = 3000
tmax = 0.0037
xmax = 2.8
ymax = 2


def FV_0(input, x, y, xsize, ysize):
    FV = torch.ones_like(input[:, 1])
    X = [x, y, xsize, ysize]
    mask = torch.gt(input[:, 1], X[0] - X[2] / 2) & ~torch.ge(input[:, 1], X[0] + X[2] / 2) & torch.gt(input[:, 2],
                                                                                                       X[1] - X[
                                                                                                           3] / 2) & ~\
        torch.ge(
        input[:, 2], X[1] + X[3] / 2)
    FV[mask] = 0

    FV = torch.ones_like(input[:, 1])
    '''plt.scatter(input[:,1].cpu().detach(),input[:,2].cpu().detach(),c=FV.cpu().detach())
    plt.show()'''
    return FV


def diffusivity(x, u, FV_x):
    rho = 8220
    cp = 605
    T = u*umax

    k = torch.zeros_like(x[:, 1])
    # is powder?
    mask = torch.eq(FV_x, 0)
    k[mask] = 124 + (T[mask] - 25) * (169 - 124) / (1260 - 25)

    # is dense?
    maskA = torch.eq(FV_x, 1)
    # is solid?
    maskB = ~torch.ge(T, 1100)
    mask = maskA.logical_and(maskB)
    k[mask] = 9200 + (T[mask] - 25) * (34000 - 9200) / (1260 - 25)
    # is transition

    maskB = torch.ge(T, 1100) & ~torch.ge(T, 1400)
    mask = maskA.logical_and(maskB)
    y1 = 30787;
    y2 = 32795;
    y3 = 85000;
    y4 = 212500;
    k[mask] = (1 - (T[mask] - 1100) / 300) ** 3 * y1 + 3 * (T[mask] - 1100) / 300 * (
                1 - (T[mask] - 1100) / 300) ** 2 * y2 + 3 * (T[mask] - 1100) / 300 ** 2 * (
                          1 - (T[mask] - 1100) / 300) * y3 + (T[mask] - 1100) / 300 ** 3 * y4;

    # is liquid?
    maskB = torch.ge(T, 1400)
    mask = maskA.logical_and(maskB)
    k[mask] = 34000 + (T[mask] - 1260) * (544000 - 34000) / (1660 - 1260)

    # is sintered?
    mask = ~torch.eq(FV_x, 1) & ~torch.eq(FV_x, 0)
    k1 = torch.zeros_like(k)
    k2 = torch.zeros_like(k)
    k1[mask] = 124 + (T[mask] - 25) * (169 - 124) / (1260 - 25)
    k2[mask] = 9200 + (T[mask] - 25) * (34000 - 9200) / (1260 - 25)
    k[mask] = FV_x[mask] * k2[mask] + (1 - FV_x[mask]) * k1[mask]
    d = k / (rho * cp)
    return d


def source(x):
    sig = 0.05
    q = 200
    n = 0.7
    # make all x physical
    t = tmax*x[:, 0]
    x_phys = x[:, 1]*xmax-1
    y_phys = x[:, 2]*ymax-1
    timestep = -0.83 + t/0.0037 * 3.33
    heat = 6*np.sqrt(3)*n*q/(pi*np.sqrt(pi)*sig**3) * torch.exp(-3*((x_phys-timestep)**2 +
                                                                    (y_phys)**2)/sig**2) * (1-np.exp(-3))

    # heat=heat/torch.max(heat)
    return heat


extrema = torch.tensor([[0, 1], [0, 1], [0, 1]])

x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], 100), [100, 1])
y = torch.reshape(torch.linspace(extrema[2, 0], extrema[2, 1], 100), [100, 1])
time_steps = [0.0, 0.25, 0.5, 0.75, 1]
t1 = 0
t2 = 0.5
param_steps = [0.0, 0.25, 0.5, 0.75, 1]
p1 = 0.1
p2 = 0.1
scale_vec = np.linspace(0.65, 1.55, len(time_steps))

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
X = x.numpy()
Y = y.numpy()
X, Y = np.meshgrid(X, Y)
x_t = torch.reshape(torch.from_numpy(X), [10000, 1])
y_t = torch.reshape(torch.from_numpy(Y), [10000, 1])
# plot_var1 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t1),
# x_t, y_t,torch.tensor(()).new_full(size=(10000, 1), fill_value=p1),
# torch.tensor(()).new_full(size=(10000, 1), fill_value=p2)], 1)
plot_var1 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t1), x_t, y_t], 1)

'''Z = exact(plot_var1)[:, 0]
Z = torch.reshape(Z, [100, 100])
Z = Z.numpy()'''

# plot_var2 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t2),
# x_t, y_t,torch.tensor(()).new_full(size=(10000, 1), fill_value=p1),
# torch.tensor(()).new_full(size=(10000, 1), fill_value=p2)], 1)

plot_var2 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t2), x_t, y_t], 1)
'''Z2 = exact(plot_var2)[:, 0]
Z2 = torch.reshape(Z2, [100, 100])
Z2 = Z2.numpy()
fig, (ax, ax2) = plt.subplots(ncols=2)
fig.subplots_adjust(wspace=0.01)
sns.heatmap(Z, cmap="rocket", ax=ax, cbar=False)
fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
sns.heatmap(Z2, cmap="rocket", ax=ax2, cbar=False)
fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
ax2.yaxis.tick_right()
ax2.tick_params(rotation=0)
plt.show()
plt.savefig(images_path + "/exact.png", dpi=500)'''

Z3 = model(plot_var1)[:, 0]
Z3 = torch.reshape(Z3, [100, 100])
Z3 = Z3.detach().numpy()*umax
Z4 = model(plot_var2)[:, 0]
Z4 = torch.reshape(Z4, [100, 100])
Z4 = Z4.detach().numpy()*umax
fig, (ax, ax2) = plt.subplots(ncols=2)
fig.subplots_adjust(wspace=0.01)
sns.heatmap(Z3, cmap="rocket", ax=ax, cbar=False)
fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
sns.heatmap(Z4, cmap="rocket", ax=ax2, cbar=False)
fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
ax2.yaxis.tick_right()
ax2.tick_params(rotation=0)
plt.show()
plt.savefig(images_path + "/prediction.png", dpi=500)

Z3 = source(plot_var1)
Z3 = torch.reshape(Z3, [100, 100])
Z3 = Z3.detach().numpy()
Z4 = source(plot_var2)
Z4 = torch.reshape(Z4, [100, 100])
Z4 = Z4.detach().numpy()
fig, (ax, ax2) = plt.subplots(ncols=2)
fig.subplots_adjust(wspace=0.01)
sns.heatmap(Z3, cmap="rocket", ax=ax, cbar=False)
fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
sns.heatmap(Z4, cmap="rocket", ax=ax2, cbar=False)
fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
ax2.yaxis.tick_right()
ax2.tick_params(rotation=0)
plt.show()
plt.savefig(images_path + "/source.png", dpi=500)

Z3 = model(plot_var1)[:, 0]
FV_0 = FV_0(plot_var1, 1, 1, 1, 1)
Z3 = diffusivity(plot_var1, Z3, FV_0)
Z3 = torch.reshape(Z3, [100, 100])
Z3 = Z3.detach().numpy()
Z4 = model(plot_var2)[:, 0]

Z4 = diffusivity(plot_var2, Z4, FV_0)
Z4 = torch.reshape(Z4, [100, 100])
Z4 = Z4.detach().numpy()
fig, (ax, ax2) = plt.subplots(ncols=2)
fig.subplots_adjust(wspace=0.01)
sns.heatmap(Z3, cmap="rocket", ax=ax, cbar=False)
fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
sns.heatmap(Z4, cmap="rocket", ax=ax2, cbar=False)
fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
ax2.yaxis.tick_right()
ax2.tick_params(rotation=0)
plt.show()
plt.savefig(images_path + "/diffuse.png", dpi=500)


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count

fig, (ax) = plt.subplots()
fig.subplots_adjust(wspace=0.01)
steps = 100
time = extrema[0, 1]


def init():
    plot_var2 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=0), x_t, y_t], 1)
    Z3 = source(plot_var2)
    Z3 = torch.reshape(Z3, [100, 100])
    Z3 = Z3.detach().numpy()
    sns.heatmap(Z3, cmap="rocket", cbar=True, ax=ax, vmin=0, vmax=umax)


def animate(i):
    t2 = time/steps*i
    plot_var2 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t2), x_t, y_t], 1)
    Z3 = source(plot_var2)
    Z3 = torch.reshape(Z3, [100, 100])
    Z3 = Z3.detach().numpy()
    sns.heatmap(Z3, cmap="rocket", cbar=False, ax=ax, vmin=0, vmax=umax)
    ax.set_title('time='+str(t2))


ani = FuncAnimation(fig=fig, func=animate, init_func=init, frames=steps)
# ani.show()
