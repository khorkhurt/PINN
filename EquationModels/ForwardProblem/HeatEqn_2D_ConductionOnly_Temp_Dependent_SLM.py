import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
from ImportFile import *
from EquationBaseClass import EquationBaseClass
from GeneratorPoints import generator_points
from SquareDomain import SquareDomain
from BoundaryConditions import PeriodicBC, DirichletBC, AbsorbingBC, NeumannBC


class EquationClass(EquationBaseClass):

    def __init__(self, ):
        EquationBaseClass.__init__(self)

        # The type of collocation points which are defined in GeneratorPoints file.
        self.type_of_points = "sobol"
        # If you want to have more than one output (like maybe temperature and heat transfer) you need to write
        # number of outputs that you want to extract. But in our case, it is always 1.
        self.output_dimension = 1
        # The dimension of the domain, for 2D case 2, for 3D case 3.
        self.space_dimensions = 2
        # The time dimension is always 1
        self.time_dimensions = 1
        self.parameters_values = None
        self.parameter_dimensions = 0
        # The size of the domain. We always try to keep the values between 0 and 1, because under these conditions the
        # network trains better. We later multiply the values below with the normalization factors in order to
        # achieve the actual size of the domain.
        self.extrema_values = torch.tensor([[0, 1],
                                            [0, 1],
                                            [0, 1]])
        # Time normalization
        self.norm_t = 0.5
        # Normalization in the x direction
        self.norm_x = 0.002
        # Normalization in the y direction
        self.norm_y = 0.001
        # Temperature normalization
        self.norm_temp = 3000
        # Definitions of the boundary condition at the each corner. The first elements are x0 and xL,
        # the last two elements are y0 and yL
        self.list_of_BC = list([[self.ub1, self.ub0], [self.ub1, self.ub0]])
        self.extrema_values = self.extrema_values if self.parameters_values is None else \
            torch.cat([self.extrema_values, self.parameters_values], 0)

        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          self.list_of_BC,
                                          self.extrema_values,
                                          self.type_of_points,
                                          vel_wave=self.a)

        ###########
        # Definition of density
        self.rho = 8220

    def add_collocation_points(self, n_coll, random_seed):
        return self.square_domain.add_collocation_points(n_coll, random_seed)

    def add_boundary_points(self, n_boundary, random_seed):
        return self.square_domain.add_boundary_points(n_boundary, random_seed)

    def add_initial_points(self, n_initial, random_seed):
        extrema_0 = self.extrema_values[:, 0]
        extrema_f = self.extrema_values[:, 1]
        x_time_0 = generator_points(n_initial, self.time_dimensions + self.space_dimensions + self.parameter_dimensions,
                                    random_seed, self.type_of_points, True)
        x_time_0[:, 0] = torch.full(size=(n_initial,), fill_value=0.0)
        x_time_0 = x_time_0 * (extrema_f - extrema_0) + extrema_0

        y_time_0 = self.u0(x_time_0)

        return x_time_0, y_time_0

    def a(self, x):
        return 1

    def apply_bc(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):

        self.square_domain.apply_boundary_conditions(model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)

    def apply_ic(self, model, x_u_train, u_train, u_pred_var_list, u_train_var_list):
        for j in range(self.output_dimension):
            if x_u_train.shape[0] != 0:
                x_u_train.requires_grad = True
                u = model(x_u_train)[:, j]
                # v = self.v0(x_u_train)[:, j]
                grad_u = torch.autograd.grad(u, x_u_train, grad_outputs=torch.ones_like(u).to(self.device),
                                             create_graph=True)[0]
                grad_u_t = grad_u[:, 0]
                u_pred_var_list.append(u)
                u_train_var_list.append(u_train[:])
                # u_pred_var_list.append(grad_u_t)
                # u_train_var_list.append(v)

    def add_internal_points(self, n_internal, random_seed):
        val_path = "C:/Users/yasa/Desktop/PinnsLV/PinnsLV/"
        node_coords = pd.read_csv(val_path + 'Nodes.csv', index_col='label')
        temp = pd.read_csv(val_path + 'Temperature.csv', index_col='label')
        times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
                 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
                 430, 440, 450, 460, 470, 480, 490, 500]
        node_coords_numpy = node_coords.to_numpy()
        node_coords_numpy = np.delete(node_coords_numpy, 2, 1) / [self.norm_x, self.norm_y]
        Exact = temp.to_numpy() / self.norm_temp
        for t in times:
            t_i = np.ones(node_coords_numpy.shape[0]) * t
            # wanted = ((t_i - 10) / 10)
            wanted = ((t_i) / 10)
            temp_input_i = np.column_stack([t_i / self.norm_t, node_coords_numpy])
            temp_output_i = Exact[np.arange(len(Exact)), wanted.astype(int)]
            temp_output_i = temp_output_i[:, None]
            if t == 0:
                test_input = temp_input_i
                output = temp_output_i
            else:
                test_input = np.row_stack((test_input, temp_input_i))
                output = np.row_stack((output, temp_output_i))
        test_inp = test_input.astype(np.float32)
        test_inp_tens = torch.from_numpy(test_inp)
        x_intern = test_inp_tens
        y_intern = torch.from_numpy(output.astype(np.float32))
        y_intern = torch.reshape(y_intern, (-1,))
        print(x_intern, x_intern.shape)
        print(y_intern, y_intern.shape)
        return x_intern, y_intern

    def compute_res(self, network, x_f_train, solid_object):
        x_f_train.requires_grad = True
        u = network(x_f_train.float())[:, 0].reshape(-1, )
        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                     create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_y = grad_u[:, 2]

        cp = self.heat_capacity(u)
        k = self.conductivity(u)
        q = self.source(x_f_train)
        # grad_u_tt = torch.autograd.grad(grad_u_t, x_f_train, grad_outputs=torch.ones_like(u),
        # create_graph=True)[0][:, 0]
        grad_u_xx = torch.autograd.grad(k * grad_u_x, x_f_train, grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0][:, 1]
        grad_u_yy = torch.autograd.grad(k * grad_u_y, x_f_train, grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0][:, 2]

        residual = grad_u_t.reshape(-1, ) - (1 / self.rho / self.cp) * (grad_u_xx.reshape(-1, ) / (self.norm_x ** 2) +
                                                                        grad_u_yy.reshape(-1, ) / (
                                                                                self.norm_y ** 2)) * self.norm_t \
                   - q * self.norm_t / (self.rho * self.cp * self.norm_temp)

        return residual

    def v0(self, x):

        return torch.zeros((x.shape[0], 1))

    def ub0(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub1(self, t):
        type_BC = [DirichletBC()]
        amin = np.fmin(np.ones(t.shape[0]), t[:, 0].detach().numpy() * self.norm_t)
        # u = torch.from_numpy((150/self.norm_temp * amin).reshape(-1, 1))
        u = torch.from_numpy((0 / self.norm_temp * amin).reshape(-1, 1))
        return u, type_BC

    def u0(self, x):
        u0 = torch.zeros((x.shape[0], 1)).reshape(-1)
        return u0

    def source(self, x):
        heat = torch.exp(-2 * ((((x[:, 1] * self.norm_x - (0.001)).double() ** 2) +
                                (((x[:, 2] * self.norm_y - (0.0005))).double() ** 2)) / (0.00005 ** 2)))
        return heat

    def conductivity(self, u):
        u = u * self.norm_temp
        if 0 <= u <= 1260:
            k = 0.0205 * u + 8.6923
        elif 1260 <= u <= 2061:
            k = 0.6476 * u - 781.37
        return k

    def heat_capacity(self, u):
        u = u * self.norm_temp
        if 0 <= u <= 1260 and 1460 < u:
            cp = 0.3483 * u + 477.29
        elif 1260 <= u <= 1460:
            cp = 0.3483 * u + 1857.3
        return cp

    def compute_generalization_error(self, model, val_path, images_path):
        node_coords = pd.read_csv(val_path + 'Nodes.csv', index_col='label')
        temp = pd.read_csv(val_path + 'Temperature.csv', index_col='label')
        model = model.cpu()
        model.eval()
        t = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
             230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430,
             440, 450, 460, 470, 480, 490, 500]
        node_coords_numpy = node_coords.to_numpy()
        node_coords_numpy = np.delete(node_coords_numpy, 2, 1) / [self.norm_x, self.norm_y]
        x_t = node_coords_numpy[:, 0].reshape(-1, 1)
        x_t = torch.from_numpy(x_t)
        y_t = node_coords_numpy[:, 1].reshape(-1, 1)
        y_t = torch.from_numpy(y_t)
        times = np.random.choice(t, node_coords_numpy.shape[0]) / self.norm_t
        # wanted = ((times - 10) / 10)
        wanted = ((times / 10)) * self.norm_t
        node_coords_numpy = np.column_stack([times, node_coords_numpy])
        Exact = temp.to_numpy()
        Exact = Exact[np.arange(len(Exact)), wanted.astype(int)]
        Exact = Exact[:, None] / self.norm_temp
        node_coords_numpy = node_coords_numpy.astype(float)
        node_coords_tensor = torch.from_numpy(node_coords_numpy)
        test_out = model(node_coords_tensor.float())[:, 0].detach().numpy().reshape(-1, 1)
        assert (Exact.shape[1] == test_out.shape[1])
        L2_test = np.sqrt((Exact - test_out) ** 2)
        max_L2_test = np.max(L2_test)
        Mean_L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
        print("Mean Error Test:", Mean_L2_test)
        print("Max Error Test", max_L2_test)
        rel_L2_test = L2_test / np.mean(Exact)
        max_rel_L2_test = np.max(rel_L2_test)
        Mean_rel_L2_test = Mean_L2_test / np.sqrt(np.mean(Exact ** 2))
        print("Mean Relative Error Test:", Mean_rel_L2_test)
        print("Max Relative Error Test", max_rel_L2_test)

        for t in t:
            wanted = t / 10
            t = t / self.norm_t
            values = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t),
                                x_t, y_t], 1)
            pinn_output = model(values.float())[:, 0].detach().numpy().reshape(-1, 1) * self.norm_temp
            np.savetxt("PINN_Result_time_%.f s.csv" % (t), pinn_output)
            Exact = temp.to_numpy()
            Temp_FE = Exact[:, int(wanted)].reshape(-1, 1)
            assert (Temp_FE.shape[1] == pinn_output.shape[1])

            FE_PINN_Comparison = images_path + "/FE_PINN_Comparison"
            if not os.path.isdir(FE_PINN_Comparison):
                os.makedirs(FE_PINN_Comparison)
            os.path.isdir(FE_PINN_Comparison)

            if t == 0:
                L2_test = np.sqrt((Temp_FE - pinn_output) ** 2)
                fig, (axis) = plt.subplots(ncols=1, figsize=(20, 10))
                fig.subplots_adjust(wspace=0.04)
                L2_test = torch.from_numpy(L2_test)
                # TODO: Change the dimensions in the line below
                L2_test = torch.reshape(L2_test, [301, 201])
                L2_test = L2_test.detach().numpy()
                sns.heatmap(L2_test, cmap="rocket", ax=axis, cbar=False, square=True)
                fig.colorbar(axis.collections[0], ax=axis, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
                axis.invert_yaxis()
                axis.set_title(r'L2 Error')
                axis.set_xlabel(r'x')
                axis.set_ylabel(r'y')
                axis.set_xticks([])
                axis.set_yticks([])
                # TODO: Change the tick values in the lines below in case you solve a different model in Abaqus
                plt.setp(axis, xticks=[0, 100, 201], xticklabels=[0, 0.05, 0.1], yticks=[0, 150, 301],
                         yticklabels=[0, 0.075, 0.15])
                # TODO: Change the name
                fig.suptitle("PINN and FE comparison at t=0 s", fontsize=32)
                plt.savefig(FE_PINN_Comparison + "/Absolute_and_Relative_Error_0 s.png")

            else:
                L2_test = np.sqrt((Temp_FE - pinn_output) ** 2)
                rel_L2_test = L2_test / np.mean(Temp_FE)

                fig, (axis, axis2) = plt.subplots(ncols=2, figsize=(20, 10))
                fig.subplots_adjust(wspace=0.04)
                L2_test = torch.from_numpy(L2_test)
                # TODO: Change the dimensions in the line below
                L2_test = torch.reshape(L2_test, [301, 201])
                L2_test = L2_test.detach().numpy()
                rel_L2_test = torch.from_numpy(rel_L2_test)
                # TODO: Change the dimensions in the line below
                rel_L2_test = torch.reshape(rel_L2_test, [301, 201])
                rel_L2_test = rel_L2_test.detach().numpy()
                sns.heatmap(L2_test, cmap="rocket", ax=axis, cbar=False, square=True)
                fig.colorbar(axis.collections[0], ax=axis, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
                sns.heatmap(rel_L2_test, cmap="rocket", ax=axis2, cbar=False, square=True)
                fig.colorbar(axis2.collections[0], ax=axis2, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
                axis.invert_yaxis()
                axis.set_title(r'L2 Error')
                axis2.set_title(r'Relative Error')
                axis.set_xlabel(r'x')
                axis2.set_xlabel(r'x')
                axis.set_ylabel(r'y')
                axis2.set_ylabel(r'y')
                axis2.invert_yaxis()
                axis2.yaxis.tick_right()
                axis2.tick_params(rotation=0)
                axis.set_xticks([])
                axis.set_yticks([])
                axis2.set_xticks([])
                axis2.set_yticks([])
                # TODO: Change the tick values in the lines below in case you solve a different model in Abaqus
                plt.setp(axis, xticks=[0, 100, 201], xticklabels=[0, 0.05, 0.1], yticks=[0, 150, 301],
                         yticklabels=[0, 0.075, 0.15])
                plt.setp(axis2, xticks=[0, 100, 201], xticklabels=[0, 0.05, 0.1], yticks=[0, 150, 301],
                         yticklabels=[0, 0.075, 0.15])
                # TODO: Change the name
                fig.suptitle("PINN and FE comparison at t=%.f s" % (t * self.norm_t), fontsize=32)
                plt.savefig(FE_PINN_Comparison + "/Absolute_and_Relative_Error_%.f s.png" % (t * self.norm_t))

        return Mean_L2_test, max_L2_test, Mean_rel_L2_test, max_rel_L2_test

    def plotting(self, model, val_path, images_path, extrema):
        model.cpu()
        model = model.eval()
        times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
                 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
                 430, 440, 450, 460, 470, 480, 490, 500]
        # t = torch.reshape(torch.linspace(extrema[0, 0], extrema[0, 1], 100), [100, 1])
        x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], 200), [200, 1])
        y = torch.reshape(torch.linspace(extrema[2, 0], extrema[2, 1], 300), [300, 1])
        X = x.numpy()
        Y = y.numpy()
        X, Y = np.meshgrid(X, Y)
        x_t = torch.reshape(torch.from_numpy(X), [60000, 1])
        y_t = torch.reshape(torch.from_numpy(Y), [60000, 1])

        for t in times:
            t = t / self.norm_t
            plot_var1 = torch.cat([torch.tensor(()).new_full(size=(60000, 1), fill_value=t), x_t, y_t], 1)
            torch.set_printoptions(profile="full")

            output = model(plot_var1) * self.norm_temp
            output = torch.reshape(output, [300, 200])
            output = output.detach().numpy()

            fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
            sns.heatmap(output, cmap="rocket", ax=ax, cbar=False, square=True)
            fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
            ax.invert_yaxis()
            ax.set_xlabel('Time = %.f s' % (t * self.norm_t))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.suptitle("PINN Prediction at t=%.f s" % (t * self.norm_t), fontsize=32)
            # TODO: Change the tick values in the lines below in case you have different number of points in PINN
            plt.setp(ax, xticks=[0, 100, 200], xticklabels=[0, 0.05, 0.1], yticks=[0, 150, 300],
                     yticklabels=[0, 0.075, 0.15])
            PINN_directory = images_path + "/PINN_Result"
            if not os.path.isdir(PINN_directory):
                os.makedirs(PINN_directory)
            os.path.isdir(PINN_directory)
            plt.savefig(PINN_directory + "/PINN_Prediction_%.f s.png" % (t * self.norm_t))

        temp = pd.read_csv(val_path + 'Temperature.csv', index_col='label')
        Exact = temp.to_numpy()
        for t in times:
            wanted = (t / 10)
            Temp_FE = Exact[:, int(wanted)].reshape(-1, 1)
            Temp_FE = torch.from_numpy(Temp_FE)
            Temp_FE = torch.reshape(Temp_FE, [301, 201])
            Temp_FE = Temp_FE.detach().numpy()

            fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
            fig.subplots_adjust(wspace=0.04)
            sns.heatmap(Temp_FE, cmap="rocket", ax=ax, cbar=False, square=True)
            fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
            ax.set_xlabel('Time = %.f s' % (t))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
            fig.suptitle("FE Result at t=%.f s" % (t), fontsize=32)
            plt.setp(ax, xticks=[0, 100, 201], xticklabels=[0, 0.05, 0.1], yticks=[0, 150, 301],
                     yticklabels=[0, 0.075, 0.15])
            FE_directory = images_path + "/FE_Result"
            if not os.path.isdir(FE_directory):
                os.makedirs(FE_directory)
            os.path.isdir(FE_directory)
            plt.savefig(FE_directory + "/FE_Result_%.f s.png" % (t))
