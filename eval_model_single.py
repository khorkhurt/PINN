from ImportFile import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time

walltime= time.time()
cputime= time.process_time()

#Setting file paths
folder_path = "C:/Users/sarve/Desktop/PinnsLV_v3/PinnsLV/Paper_Model/"
images_path = folder_path + "Images/"
model_path = folder_path + "TrainedModel/"
# val_path =  "D:\polybox\Masterthesis\PINN\Taniya\\fem\\"
#loading trained model to evaluate
model = torch.load(model_path + "/model3.pkl")
model=model.eval()

#setting model constants
umax=3000
tmax=0.0015
xmax=1.8
ymax=2
extrema =torch.tensor([[-0.1, 1],[-1, 1.8], [-1, 1], [-1,0.03]])
#FEM Timesteps for laser position
times = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013,
         0.0014, 0.0015]

#Setting plotting values
plot_dim=1000 #resolution
t1 = 0.0 #plot at time t=t1
plotsize = 0.5 #size of markers
z1 = 0.03 #evaluated z-height for x-y-plots
start=-2 #starting frame
stop=15 #ending frame (max 15 due to FEM) set =start for one frame output

#set dimension of plots
height=10
width=8
plt.rcParams['figure.figsize'] = [height, width]
plt.rcParams.update({'font.family':'DejaVu Sans'})
plot_dpi=1800

'''
# read in data set of FEM
node_coord_df = pd.read_csv(val_path + 'nodes.csv', index_col='label')
temp_df = pd.read_csv(val_path + 'temp100.csv', index_col='label')
test_inp = node_coord_df.to_numpy()
Exact = temp_df.to_numpy()

#define helper functions:
#calculates and plots "exact vs predicted" plot
def error_predict(model):

    model = model.cpu()
    model.eval()
    node_coord_df = pd.read_csv(val_path + 'nodes.csv', index_col='label')
    temp_df = pd.read_csv(val_path + 'temp.csv', index_col='label')
    times = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013,
             0.0014, 0.0015]
    test_inp = node_coord_df.to_numpy()
    Exact = temp_df.to_numpy()
    # create dataset with different times, mostly shaping into correct form
    for t in times:
        t_i = np.ones(test_inp.shape[0]) * t
        wanted = (t_i * 10000)
        test_time = t_i / tmax
        temp_inp_i = np.column_stack([test_time, test_inp])
        temp_out_i = Exact[np.arange(len(Exact)), wanted.astype(int)]
        temp_out_i = temp_out_i[:, None] / umax
        if t == 0:
            test_input = temp_inp_i
            output = temp_out_i
        else:
            test_input = np.row_stack((test_input, temp_inp_i))
            output = np.row_stack((output, temp_out_i))
    test_inp = test_input.astype(np.float32)
    test_inp_tens = torch.from_numpy(test_inp)
    Exact = output * umax
    test_out = model(test_inp_tens)[:, 0].detach().numpy().reshape(-1, 1) * umax
    assert (Exact.shape[1] == test_out.shape[1])
    L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))

    print("Error Test:", L2_test)
    rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
    print("Relative Error Test:", rel_L2_test)
    if images_path is not None:
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(Exact, test_out)
        plt.xlabel(r'Exact Values')
        plt.ylabel(r'Predicted Values')
        plt.savefig(images_path + "/Score.png", dpi=400)
#outputs rosenthal solution for given location to compare with model
def rosenthal(x):
    #c=conductivity, a=diffusivity
    q = 150
    n = 0.5
    T0 = 25
    # make all x physical
    c = 0.1*1000
    rho = 8220
    cp = 605
    a=c/(rho*cp)
    x_phys = x[:, 1] /1000 # *self.xmax-1
    y_phys = x[:, 2]/1000  # *self.ymax-1
    z_phys = x[:, 3]/1000
    v = 1.5/tmax/1000
    timestep = x[:, 0] * 1.5/1000
    r = ((x_phys-timestep)**2+y_phys**2+(z_phys-0.03)**2)**0.5
    T = T0 + (q*n)/(2*pi*c*r) * torch.exp(-(v*(x_phys-timestep+r))/(2*a))
    # heat=heat/torch.max(heat)
    print(max(T))
    return T
#Plots FEM and PINN meltpool (xy and xz) at a given timestep in [0,15]
def plot_meltpool(timestep):
    assert timestep<16
    assert timestep>=0
    print("Plotting Meltpool of FEM and PINN")
    time = float(0.0001*timestep/0.0015)
    print("debug:", int(round(time * 15)))
    select = Exact[:, int(round(time * 15))]
    edata = np.column_stack([test_inp, select])
    maskY = (edata[:, 2] == z1)  # keep only points at surface for x-y plot
    edataY = edata[maskY, :]
    maskZ = (edata[:, 1] == 0)  # keep only points in the center for x-z plot
    edataZ = edata[maskZ, :]
    # read out actual needed data
    x_y = edataY[:, 0]
    y_y = edataY[:, 1]
    z_y = edataY[:, 2]
    x_z = edataZ[:, 0]
    y_z = edataZ[:, 1]
    z_z = edataZ[:, 2]
    # only T changes with each loop but its easier to read out entire dataframe
    T_y = edataY[:, 3]
    T_z = edataZ[:, 3]
    # create x-z-plot for FEM (exact) calculation
    fig, (ax) = plt.subplots(ncols=1)
    fig.subplots_adjust(wspace=0.01)
    Melt = T_z > 1260
    scatter = ax.scatter(x_z, z_z, s=plotsize, c=Melt)  # ,  cmap='viridis')
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.set_title('FEM Meltpool x-z')
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Liquid=")
    ax.add_artist(legend1)
    plt.savefig(images_path + "/FEM_melt_xz.png", dpi=plot_dpi)

    # create x-y-plot for FEM (exact) calculation
    fig, (ax) = plt.subplots(ncols=1)
    fig.subplots_adjust(wspace=0.01)
    Melt = T_y > 1260
    scatter = ax.scatter(x_y, y_y, s=plotsize, c=Melt)  # ,  cmap='viridis')
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.set_title('FEM Meltpool x-y')
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Liquid=")
    ax.add_artist(legend1)
    plt.savefig(images_path + "/FEM_melt_xy.png", dpi=plot_dpi)

    # Evaluate PINN Model Prediction in x-y
    x = np.array(x_y, dtype=np.float32)
    y = np.array(y_y, dtype=np.float32)
    z = np.array(z_y, dtype=np.float32)
    x_t = torch.reshape(torch.from_numpy(x), [np.size(x), 1])
    y_t = torch.reshape(torch.from_numpy(y), [np.size(x), 1])
    z_t = torch.reshape(torch.from_numpy(z), [np.size(x), 1])
    t_t = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=time)
    plot_var2 = torch.cat([t_t, x_t, y_t, z_t], 1)
    T_pred_y = model(plot_var2.to('cuda'))[:, 0]
    T_pred_y = T_pred_y.detach().cpu().numpy() * umax
    # Plot PINN prediction
    fig, (ax) = plt.subplots(ncols=1)
    fig.subplots_adjust(wspace=0.01)
    Melt = T_pred_y > 1260
    scatter = ax.scatter(x, y, s=plotsize, c=Melt)  # ,  cmap='viridis')
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.set_title('Predicted Meltpool')
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Liquid=")
    ax.add_artist(legend1)
    plt.savefig(images_path + "/PINN_melt_xy.png", dpi=plot_dpi)

    # Evaluate PINN Model Prediction in x-z
    x = np.array(x_z, dtype=np.float32)
    y = np.array(y_z, dtype=np.float32)
    z = np.array(z_z, dtype=np.float32)
    x_t = torch.reshape(torch.from_numpy(x), [np.size(x), 1])
    y_t = torch.reshape(torch.from_numpy(y), [np.size(x), 1])
    z_t = torch.reshape(torch.from_numpy(z), [np.size(x), 1])
    t_t = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=time)
    plot_var2 = torch.cat([t_t, x_t, y_t, z_t], 1)
    T_pred_z = model(plot_var2.to('cuda'))[:, 0]
    T_pred_z = T_pred_z.detach().cpu().numpy() * umax
    # Plot PINN prediction
    fig, (ax) = plt.subplots(ncols=1)
    fig.subplots_adjust(wspace=0.01)
    Melt = T_pred_z > 1260
    scatter = ax.scatter(x, z, s=plotsize, c=Melt)  # ,  cmap='viridis')
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.set_title('Predicted Meltpool')
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Liquid=")
    ax.add_artist(legend1)
    plt.savefig(images_path + "/PINN_melt_xz.png", dpi=plot_dpi)
    plt.show()
    print("All plots saved to ",images_path)
#Plot the relative temperature difference between FEM and PINN at a given timestep in [0,15]
def plot_difference(timestep):
    t_i = float(0.0001 * timestep / 0.0015)

    # plot exact vs. prediction plot and output test error (makes eval.py slow, as it has to plot every evaluated point of grid 16 times = looots of points )

    # error_predict(model)

    print("debug:", int(round(t_i * 15)))
    select = Exact[:, int(round(t_i * 15))]
    edata = np.column_stack([test_inp, select])
    maskY = (edata[:, 2] == z1)  # keep only points at surface for x-y plot
    edataY = edata[maskY, :]
    maskZ = (edata[:, 1] == 0)  # keep only points in the center for x-z plot
    edataZ = edata[maskZ, :]
    # read out actual needed data
    x_y = edataY[:, 0]
    y_y = edataY[:, 1]
    z_y = edataY[:, 2]
    x_z = edataZ[:, 0]
    y_z = edataZ[:, 1]
    z_z = edataZ[:, 2]
    # only T changes with each loop but its easier to read out entire dataframe
    T_y = edataY[:, 3]
    T_z = edataZ[:, 3]

    x_source = t_i * 1.5  # x-coordinate of source at evaluated timestep for further use

    # Evaluate PINN Model Prediction in x-y
    x = np.array(x_y, dtype=np.float32)
    y = np.array(y_y, dtype=np.float32)
    z = np.array(z_y, dtype=np.float32)
    x_t = torch.reshape(torch.from_numpy(x), [np.size(x), 1])
    y_t = torch.reshape(torch.from_numpy(y), [np.size(x), 1])
    z_t = torch.reshape(torch.from_numpy(z), [np.size(x), 1])
    t_t = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=t_i)
    plot_var2 = torch.cat([t_t, x_t, y_t, z_t], 1)
    T_pred_y = model(plot_var2.to('cuda'))[:, 0]
    T_pred_y = T_pred_y.detach().cpu().numpy() * umax
    pred_max = max(T_pred_y)
    print('XY max predict ', pred_max)
    print('XY max exact', max(T_y))
    # also calculate the difference to FEM for top layer
    Dif = (T_y - T_pred_y) / max(T_y)
    fig, (ax) = plt.subplots(ncols=1)
    fig.subplots_adjust(wspace=0.01)
    scatter = ax.scatter(x, y, s=plotsize, c=Dif, cmap='seismic', vmin=-0.05, vmax=0.05)
    print('XY max dif ', max(abs(Dif)))
    # ax.set(xticklabels=[])
    # ax.set(yticklabels=[])
    ax.set_title('Relative Temperature Difference in top layer')
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="(T-FEM)/FEMmax=")
    ax.add_artist(legend1)
    savepath=images_path + "/difference_xy" + str(timestep) + ".png"
    plt.savefig(savepath, dpi=plot_dpi)
    plt.close()
    # Evaluate PINN Model Prediction in x-z
    x = np.array(x_z, dtype=np.float32)
    y = np.array(y_z, dtype=np.float32)
    z = np.array(z_z, dtype=np.float32)
    x_t = torch.reshape(torch.from_numpy(x), [np.size(x), 1])
    y_t = torch.reshape(torch.from_numpy(y), [np.size(x), 1])
    z_t = torch.reshape(torch.from_numpy(z), [np.size(x), 1])
    t_t = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=t_i)
    plot_var2 = torch.cat([t_t, x_t, y_t, z_t], 1)
    T_pred_z = model(plot_var2.to('cuda'))[:, 0]
    T_pred_z = T_pred_z.detach().cpu().numpy() * umax
    pred_max = max(T_pred_z)
    print('XZ max predict ', pred_max)
    print('XZ max exact', max(T_z))
    # also calculate the difference to FEM for top layer
    Dif = (T_z - T_pred_z) / max(T_z)
    fig, (ax) = plt.subplots(ncols=1)
    fig.subplots_adjust(wspace=0.01)
    scatter = ax.scatter(x, z, s=plotsize, c=Dif, cmap='seismic', vmin=-0.05, vmax=0.05)
    print('XZ max dif ', max(abs(Dif)))
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.set_title('Relative Temperature Difference in center slice')
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="(T-FEM)/FEMmax=")
    ax.add_artist(legend1)
    savepath=images_path + "/difference_xz" + str(timestep) + ".png"
    plt.savefig(savepath, dpi=plot_dpi)
    plt.close()
#calculate the spread of the meltpool from the point of laser impact at a given normalized time in [0,1]
def calc_meltpool_dim(t_i):
    x_source = t_i * 1.5  # x-coordinate of source at evaluated timestep for further use
    # Evaluate PINN Model Prediction in x-y
    T_solidus = 1260
    imax = 1000
    ### Meltpool dimesnion 1: x+
    i = 0
    x = x_source
    T_pred = 3000
    inc = 0.1
    while i < imax:
        i = i + 1
        x = np.array(x + inc, dtype=np.float32)
        y = np.array(0, dtype=np.float32)
        z = np.array(z1, dtype=np.float32)
        x_t = torch.reshape(torch.from_numpy(x), [np.size(x), 1])
        y_t = torch.reshape(torch.from_numpy(y), [np.size(x), 1])
        z_t = torch.reshape(torch.from_numpy(z), [np.size(x), 1])
        t_t = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=t_i)
        plot_var2 = torch.cat([t_t, x_t, y_t, z_t], 1)
        T_pred = model(plot_var2.to('cuda'))[:, 0]
        T_pred = T_pred.detach().cpu().numpy() * umax
        if T_pred < T_solidus:
            x = x - inc
            inc = inc / 2
        if inc < 0.000001:
            i = imax
    print("Melpool size in front of Laser", round(x - x_source, 5))
    x_max = x
    ### Meltpool dimesnion 2: x-
    i = 0
    x = x_source
    T_pred = 3000
    inc = 0.1
    while i < imax:
        i = i + 1
        x = np.array(x - inc, dtype=np.float32)
        y = np.array(0, dtype=np.float32)
        z = np.array(z1, dtype=np.float32)
        x_t = torch.reshape(torch.from_numpy(x), [np.size(x), 1])
        y_t = torch.reshape(torch.from_numpy(y), [np.size(x), 1])
        z_t = torch.reshape(torch.from_numpy(z), [np.size(x), 1])
        t_t = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=t_i)
        plot_var2 = torch.cat([t_t, x_t, y_t, z_t], 1)
        T_pred = model(plot_var2.to('cuda'))[:, 0]
        T_pred = T_pred.detach().cpu().numpy() * umax
        if T_pred < T_solidus:
            x = x + inc
            inc = inc / 2
        if inc < 0.000001:
            i = imax
    x_min = x

    ### Meltpool dimesnion 3: y+
    print("Melpool size behind Laser", abs(round(x - x_source, 5)))
    i = 0
    y = 0
    T_pred = 3000
    inc = 0.1
    while i < imax:
        i = i + 1
        x = np.array(x_source, dtype=np.float32)
        y = np.array(y + inc, dtype=np.float32)
        z = np.array(z1, dtype=np.float32)
        x_t = torch.reshape(torch.from_numpy(x), [np.size(x), 1])
        y_t = torch.reshape(torch.from_numpy(y), [np.size(x), 1])
        z_t = torch.reshape(torch.from_numpy(z), [np.size(x), 1])
        t_t = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=t_i)
        plot_var2 = torch.cat([t_t, x_t, y_t, z_t], 1)
        T_pred = model(plot_var2.to('cuda'))[:, 0]
        T_pred = T_pred.detach().cpu().numpy() * umax
        if T_pred < T_solidus:
            y = y - inc
            inc = inc / 2
        if inc < 0.000001:
            i = imax
    y_max = y
    print("Melpool size left of Laser", round(y, 5))
    i = 0
    y = 0
    T_pred = 3000
    inc = 0.1
    while i < imax:
        i = i + 1
        x = np.array(x_source, dtype=np.float32)
        y = np.array(y - inc, dtype=np.float32)
        z = np.array(z1, dtype=np.float32)
        x_t = torch.reshape(torch.from_numpy(x), [np.size(x), 1])
        y_t = torch.reshape(torch.from_numpy(y), [np.size(x), 1])
        z_t = torch.reshape(torch.from_numpy(z), [np.size(x), 1])
        t_t = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=t_i)
        plot_var2 = torch.cat([t_t, x_t, y_t, z_t], 1)
        T_pred = model(plot_var2.to('cuda'))[:, 0]
        T_pred = T_pred.detach().cpu().numpy() * umax
        if T_pred < T_solidus:
            y = y + inc
            inc = inc / 2
        if inc < 0.000001:
            i = imax
    y_min = y
    print("Melpool size right of Laser", round(abs(y), 5))
    i = 0
    T_pred = 3000
    inc = 0.1
    while i < imax:
        i = i + 1
        x = np.array(x_source, dtype=np.float32)
        y = np.array(0, dtype=np.float32)
        z = np.array(z - inc, dtype=np.float32)
        x_t = torch.reshape(torch.from_numpy(x), [np.size(x), 1])
        y_t = torch.reshape(torch.from_numpy(y), [np.size(x), 1])
        z_t = torch.reshape(torch.from_numpy(z), [np.size(x), 1])
        t_t = torch.tensor(()).new_full(size=(np.size(x), 1), fill_value=t_i)
        plot_var2 = torch.cat([t_t, x_t, y_t, z_t], 1)
        T_pred = model(plot_var2.to('cuda'))[:, 0]
        T_pred = T_pred.detach().cpu().numpy() * umax
        if T_pred < T_solidus:
            z = z + inc
            inc = inc / 2
        if inc < 0.000001:
            i = imax
    z_min = z
    print("Melpool size below Laser", round(z1-z_min, 5))
    return [x_max, abs(x_min), y_max, abs(y_min), z1-z_min]
'''

def plot_model(timestep):
    print(timestep)
    t_i = float(0.0001 * timestep / 0.0015)
    print(t_i)
    x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], plot_dim), [plot_dim, 1])
    y = torch.reshape(torch.linspace(extrema[2, 0], extrema[2, 1], plot_dim), [plot_dim, 1])
    # Plot the surface.
    X = x.numpy()
    Y = y.numpy()
    X, Y = np.meshgrid(X, Y)
    x_t = torch.reshape(torch.from_numpy(X), [plot_dim ** 2, 1])
    y_t = torch.reshape(torch.from_numpy(Y), [plot_dim ** 2, 1])
    plot_var = torch.cat(
                        [torch.tensor(()).new_full(size=(plot_dim ** 2, 1), fill_value=t_i),
                        x_t,
                        y_t,
                        torch.tensor(()).new_full(size=(plot_dim ** 2, 1), fill_value=z1)]
                                                                                            , 1)

    Z4 = model(plot_var.to('cuda'))[:, 0]
    Z4 = torch.reshape(Z4, [plot_dim, plot_dim])
    Z4 = Z4.detach().cpu().numpy() * umax
    fig, (ax2) = plt.subplots(ncols=1)
    sns.heatmap(Z4, cmap="rocket", ax=ax2, cbar=False)
    fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
    ax2.set(xticklabels=[])
    ax2.set(yticklabels=[])
    plt.xticks([0,357, plot_dim], [-1,0,1.8])
    plt.yticks([0,500, plot_dim], [1,0,-1])
    ax2.set_title('Predicted Temperature')

    # plt.show()
    plt.savefig(images_path + "/prediction" + str(timestep) + ".png", dpi=500)

#loop through time steps of FEM to compare predictions
for i_t in range (start,stop+1):
    #time of plotting:
    t_i = float(0.0001*i_t/0.0015)

    #error_predict(model)
    #plot_meltpool(i_t)
    #plot_difference(i_t)
    #calc_meltpool_dim(t_i)
    plot_model(i_t)


walltime2= time.time()
cputime2= time.process_time()
wall_dif=walltime2-walltime
CPU_dif=cputime2-cputime
print("walltime=", wall_dif)
print("CPUtime=", CPU_dif)
print("Finished Eval!")