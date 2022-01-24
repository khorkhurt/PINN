import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
import math
from ImportFile import *
from EquationBaseClass import EquationBaseClass
from GeneratorPoints import generator_points
from SquareDomain import SquareDomain
from BoundaryConditions import PeriodicBC, DirichletBC, AbsorbingBC, NeumannBC
from scipy import integrate


class EquationClass(EquationBaseClass):

    def __init__(self, ):
        EquationBaseClass.__init__(self)

        # The type of collocation points which are defined in GeneratorPoints class.
        self.type_of_points = "sobol_center_4"
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
        self.norm_t = 0.2
        # Normalization in the x direction
        self.norm_x = 0.002
        # Normalization in the y direction
        self.norm_y = 0.001
        # Temperature normalization
        self.norm_temp = 2000
        # Definitions of the boundary condition at the each corner. The first elements are x0 and xL,
        # the last two elements are y0 and yL
        self.list_of_BC = list([[self.ub0, self.ub0], [self.ub0, self.ub0]])
        self.extrema_values = self.extrema_values if self.parameters_values is None else \
            torch.cat([self.extrema_values, self.parameters_values], 0)

        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          self.list_of_BC,
                                          self.extrema_values,
                                          self.type_of_points,
                                          vel_wave=self.a)

        # Neural Network Model for Heat Capacity
        self.model_cp = nn.Sequential(nn.Linear(1, 100), nn.Tanh(), nn.Linear(100, 100), nn.Tanh(), nn.Linear(100, 1), )

        # Neural Network Model for Conductivity
        self.model_k = nn.Sequential(nn.Linear(1, 40), nn.Tanh(), nn.Linear(40, 40), nn.Tanh(), nn.Linear(40, 1), )

        # Model Path for Heat Capacity
        folder_path_cp = "C:/Users/yasa/Desktop/PinnsLV_v3/PinnsLV"
        model_path_cp = folder_path_cp + "/Trained_Model_for_Third_Network"
        if not os.path.isdir(model_path_cp):
            os.makedirs(model_path_cp)
        os.path.isdir(model_path_cp)

        # Model Path for Conductivity
        folder_path_k = "C:/Users/yasa/Desktop/PinnsLV_v3/PinnsLV"
        model_path_k = folder_path_k + "/Trained_Model_for_Second_Network"
        if not os.path.isdir(model_path_k):
            os.makedirs(model_path_k)
        os.path.isdir(model_path_k)

        # Loading the Model for Heat Capacity

        weights_cp = torch.load(model_path_cp + "/model.pkl")
        self.model_cp.load_state_dict(weights_cp.state_dict(), strict=False)
        self.model_cp = self.model_cp.eval()

        # Loading the Model for Conductivity

        weights_k = torch.load(model_path_k + "/model.pkl")
        self.model_k.load_state_dict(weights_k.state_dict(), strict=False)
        self.model_k = self.model_k.eval()


        '''
        self.rho = 7850
        self.cp = 460
        self.k = 45
        '''
        # Definition of conductivity
        # self.k = 45
        # Definition of density
        self.rho = 8220
        # Definition of heat capacity
        # self.cp = 485.9975
        ###########

    def add_collocation_points(self, n_coll, random_seed):
        return self.square_domain.add_collocation_points(n_coll, random_seed)

    def add_boundary_points(self, n_boundary, random_seed):
        return self.square_domain.add_boundary_points(n_boundary, random_seed)

    def add_initial_points(self, n_initial, random_seed):
        extrema_0 = self.extrema_values[:, 0]
        extrema_f = self.extrema_values[:, 1]
        x_time_0 = generator_points(n_initial, self.time_dimensions + self.space_dimensions + self.parameter_dimensions,
                                    random_seed, self.type_of_points, True)
        x_time_0[:, 0] = torch.full(size=(x_time_0.shape[0],), fill_value=0.0)
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
                u_pred_var_list.append(u)
                u_train_var_list.append(u_train[:])

    def add_internal_points(self, n_internal, random_seed):
        val_path = "C:/Users/yasa/Desktop/PinnsLV_v3/PinnsLV/"
        times = pd.read_csv(val_path + 'Times.csv', index_col='label')
        node_coords = pd.read_csv(val_path + 'Nodes.csv', index_col='label')
        temp = pd.read_csv(val_path + 'Temperature.csv', index_col='label')
        node_coords_numpy = node_coords.to_numpy()
        times = times.to_numpy()

        node_coords_numpy = np.delete(node_coords_numpy, 2, 1) / [self.norm_x, self.norm_y]
        Exact = temp.to_numpy() / self.norm_temp
        for t in range(len(times)):
            t_i = np.ones(node_coords_numpy.shape[0]) * times[t, 0]
            temp_input_i = np.column_stack([t_i / self.norm_t, node_coords_numpy])
            temp_output_i = Exact[np.arange(len(Exact)), t]
            temp_output_i = temp_output_i[:, None]
            if times[t, 0] == 0:
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
        np.savetxt("x_intern.csv", x_intern.detach().numpy(), fmt='%1.10f')
        np.savetxt("y_intern.csv", y_intern.detach().numpy(), fmt='%1.10f')
        print(y_intern, y_intern.shape)
        return x_intern, y_intern

    def compute_res(self, network, x_f_train, solid_object):
        x_f_train.requires_grad = True
        u = network(x_f_train.float())[:, 0].reshape(-1, ) * -1
        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                     create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_y = grad_u[:, 2]

        '''
        np.savetxt("u.csv", u.detach().numpy(), fmt='%1.3f')
        cp = self.model_cp(((u * self.norm_temp).reshape(-1, 1)))
        cp = cp.reshape(-1, )
        plt.scatter((u*self.norm_temp).detach().numpy(), cp.detach().numpy())
        # plt.show()
        
        k = self.model_k(((u * self.norm_temp).reshape(-1, 1)))
        k = k.reshape(-1, )
        plt.scatter((u*self.norm_temp).detach().numpy(), k.detach().numpy())
        # plt.show()
        '''

        cp = self.heat_capacity(u)
        # rho = self.density(u)
        k = self.conductivity(u)
        q = self.source(x_f_train)

        # enforce zero-gradient in the region before laser gradient
        mask_init = torch.le(x_f_train[:, 0], 0)
        # enforce temperatures above 0
        mask_temp = torch.le(u, 0)

        grad_u_xx = torch.autograd.grad(k * grad_u_x, x_f_train, grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0][:, 1]
        grad_u_yy = torch.autograd.grad(k * grad_u_y, x_f_train, grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0][:, 2]

        residual = grad_u_t.reshape(-1, ) - (self.norm_t / self.rho / cp) * (((grad_u_xx.reshape(-1, )) /
                                                                                    (self.norm_x ** 2))
                                                                                  + ((grad_u_yy.reshape(-1, )) /
                                                                                     (self.norm_y ** 2))) \
                   - q * self.norm_t / (self.rho * cp * self.norm_temp)

        residual[mask_init] = grad_u_t.reshape(-1, )[mask_init]
        residual[mask_temp] = (abs(u[mask_temp]) + 1) * abs(residual[mask_temp])
        source = q * self.norm_t / (self.rho * cp * self.norm_temp)
        residual = residual / torch.sqrt(torch.mean(source ** 2)).detach()

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
        u = torch.from_numpy((0 / self.norm_temp * amin).reshape(-1, 1))
        return u, type_BC

    def ub2(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0)
        return u, type_BC

    def u0(self, x):
        images_path = "C:/Users/yasa/Desktop/PinnsLV_v3/PinnsLV/" \
                      "HeatEqn_2D_Temp_Dep_Properties" \
                      "/Images"
        np.savetxt("x_initial_coord.csv", x.detach().numpy(), fmt='%1.10f')
        u0 = torch.zeros((x.shape[0], 1)).reshape(-1)
        return u0

    def source(self, x):
        # np.savetxt("x.csv", x.detach().numpy(), fmt='%1.3f')
        heat = (6 * (10 ** 13)) * torch.exp(-(((x[:, 0] * self.norm_t - 0.005) * 1000) ** 2)) * \
               torch.exp(-2 * (((x[:, 1] * self.norm_x - (0.001)) ** 2) +
                               (((x[:, 2] * self.norm_y - (0.0005))) ** 2)) / (0.00005 ** 2))
        # np.savetxt("heat.csv", heat.detach().numpy())
        return heat

    def conductivity(self, u):
        u = u * self.norm_temp
        k = torch.empty(u.shape)
        # k = 14.6 + 1.27 * (10 ** -2) * u
        k = 5 * (10 ** -5) * (u ** 2) + 0.005 * u + 45
        '''
        k1 = torch.full(size=(u.shape), fill_value=9)
        k2 = 0.0205 * (u - 1260) + 34.6
        k3 = 0.6476 * (u - 1260) + 34.6
        combined = torch.cat((torch.maximum(k1, k2).reshape(-1, 1), torch.maximum(k1, k3).reshape(-1, 1),
                              torch.maximum(k2, k3).reshape(-1, 1)), 1)
        k, inds = torch.max(combined, 1)
        '''
        return k

    def heat_capacity(self, u):
        u = u * self.norm_temp
        cp = torch.empty(u.shape)
        # cp = 450 + 0.28 * u - 2.91 * (10 ** -4) * (u ** 2) + 1.34 * (10 ** -7) * (u ** 3)
        # u = torch.linspace(-10000, 10000, 20000)
        # print(u, u.shape)
        cp = 477.29 + u * 0.3483 + \
             (0.25 * (10 ** 6) / (72 * ((2 * 3.141592654) ** 0.5))) * torch.exp(-0.5 * ((u - 1350) ** 2) / (72 ** 2))
        # print(cp, cp.shape)
        # plt.scatter(u.detach().numpy(), cp.detach().numpy())
        '''
        cp1 = torch.empty(u.shape)

        for i in range(u.shape[0]):
            if -10000 <= u[i] <= 1260 or 1460 < u[i]:
                cp1[i] = 0.3483 * u[i] + 477.29
            elif 1260 <= u[i] <= 1460:
                cp1[i] = 0.3483 * u[i] + 1857.3
            else:
                cp1[i] = 0.00001
            i += 1
        plt.plot(u, cp1)
        '''
        # plt.show()
        return cp

    def density(self, u):
        u = abs(u * self.norm_temp)
        rho = torch.empty(u.shape)
        rho = -0.0002 * (u ** 2) - 0.2539 * u + 8220.6
        return rho


    def heat_balance(self, network):
        network.cpu()
        network = network.eval()
        t = np.linspace(0, 0.01, 100).reshape(-1, 1)
        x = np.linspace(0, 0.002, 100).reshape(-1, 1)
        x = torch.from_numpy(x)
        y = np.linspace(0, 0.001, 100).reshape(-1, 1)
        y = torch.from_numpy(y)

        for k in range(len(t)):
            t_value = t[k, 0]
            values = torch.cat([torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=t_value),
                                x, y], 1)
            temperature_output = network(values.float())[:, 0].detach().numpy().reshape(-1, 1)

    def compute_generalization_error(self, model, val_path, images_path):
        times = pd.read_csv(val_path + 'Times.csv', index_col='label')
        node_coords = pd.read_csv(val_path + 'Nodes.csv', index_col='label')
        temp = pd.read_csv(val_path + 'Temperature.csv', index_col='label')

        model = model.cpu()
        model.eval()
        times = times.to_numpy()
        node_coords_numpy = node_coords.to_numpy()
        node_coords_numpy = np.delete(node_coords_numpy, 2, 1) / [self.norm_x, self.norm_y]
        x_t = node_coords_numpy[:, 0].reshape(-1, 1)
        x_t = torch.from_numpy(x_t)
        y_t = node_coords_numpy[:, 1].reshape(-1, 1)
        y_t = torch.from_numpy(y_t)
        test_input = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=1),
                                x_t, y_t], 1)
        test_input_max = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=0.0275),
                                x_t, y_t], 1)
        Exact = temp.to_numpy()
        Exact = Exact[:, len(times) - 1].reshape(-1, 1)
        # Exact = Exact[:, 12].reshape(-1, 1)
        # print(np.amax(Exact))
        test_out = abs(model(test_input.float())[:, 0].detach().numpy().reshape(-1, 1) * self.norm_temp)
        # print(np.amax(test_out))
        assert (Exact.shape[1] == test_out.shape[1])
        L2_test = np.sqrt((Exact - test_out) ** 2)
        max_L2_test = np.max(L2_test)
        Mean_L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
        print("Mean Error Test:", Mean_L2_test)
        print("Max Error Test", max_L2_test)
        rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
        max_rel_L2_test = np.max(rel_L2_test)
        Mean_rel_L2_test = Mean_L2_test / np.sqrt(np.mean(Exact ** 2))
        print("Mean Relative Error Test:", Mean_rel_L2_test)
        print("Max Relative Error Test", max_rel_L2_test)

        for t in range(len(times)):
            result_path = "C:/Users/yasa/Desktop/PinnsLV_v3/PinnsLV/" \
                          "HeatEqn_2D_Temp_Dep_Properties"
            PINN_Results_CSV = images_path + "/PINN_Results_CSV"
            if not os.path.isdir(PINN_Results_CSV):
                os.makedirs(PINN_Results_CSV)
            os.path.isdir(PINN_Results_CSV)
            t_value = times[t, 0] / self.norm_t
            values = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t_value),
                                x_t, y_t], 1)

            pinn_output = abs(model(values.float())[:, 0].detach().numpy().reshape(-1, 1) * self.norm_temp)
            np.savetxt(PINN_Results_CSV + "/PINN_Result_time_%fs.csv" % (t_value * self.norm_t), pinn_output)
            Exact = temp.to_numpy()
            Temp_FE = Exact[:, t].reshape(-1, 1)
            assert (Temp_FE.shape[1] == pinn_output.shape[1])

            FE_PINN_Comparison = images_path + "/FE_PINN_Comparison"
            if not os.path.isdir(FE_PINN_Comparison):
                os.makedirs(FE_PINN_Comparison)
            os.path.isdir(FE_PINN_Comparison)

            if t_value == 0:
                L2_test = np.sqrt((Temp_FE - pinn_output) ** 2)
                fig, (axis) = plt.subplots(ncols=1, figsize=(20, 10))
                fig.subplots_adjust(wspace=0.04)
                L2_test = torch.from_numpy(L2_test)
                # TODO: Change the dimensions in the line below
                L2_test = torch.reshape(L2_test, [101, 201])
                L2_test = L2_test.detach().numpy()
                sns.heatmap(L2_test, cmap="rocket", ax=axis, cbar=False, square=True)
                fig.colorbar(axis.collections[0], ax=axis, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
                axis.invert_yaxis()
                axis.set_title(r'L2 Error')
                axis.set_xlabel(r'x')
                axis.set_ylabel(r'y')
                axis.set_xticks([])
                axis.set_yticks([])
                axis.text(2, 95, "Max L2 norm is %f" % (np.amax(L2_test)), fontsize=14, color='white')
                # TODO: Change the tick values in the lines below in case you solve a different model in Abaqus
                plt.setp(axis, xticks=[0, 100, 201], xticklabels=[0, 0.001, 0.002], yticks=[0, 50, 101],
                         yticklabels=[0, 0.0005, 0.001])
                # TODO: Change the name
                fig.suptitle("PINN and FE comparison at t=0 s", fontsize=32)
                plt.savefig(FE_PINN_Comparison + "/Absolute_and_Relative_Error_0 s.png")

            else:
                L2_test = np.sqrt((Temp_FE - pinn_output) ** 2)
                rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))

                fig, (axis, axis2) = plt.subplots(ncols=2, figsize=(20, 10))
                fig.subplots_adjust(wspace=0.04)
                L2_test = torch.from_numpy(L2_test)
                # TODO: Change the dimensions in the line below
                L2_test = torch.reshape(L2_test, [101, 201])
                L2_test = L2_test.detach().numpy()
                rel_L2_test = torch.from_numpy(rel_L2_test)
                # TODO: Change the dimensions in the line below
                rel_L2_test = torch.reshape(rel_L2_test, [101, 201])
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
                axis.text(2, 95, "Max L2 norm is %f" % (np.amax(L2_test)), fontsize=14, color='white')
                axis2.text(2, 95, "Max relative error is %f" % (np.amax(rel_L2_test)), fontsize=14,
                           color='white')
                # TODO: Change the tick values in the lines below in case you solve a different model in Abaqus
                plt.setp(axis, xticks=[0, 100, 201], xticklabels=[0, 0.001, 0.002], yticks=[0, 50, 101],
                         yticklabels=[0, 0.0005, 0.001])
                plt.setp(axis2, xticks=[0, 100, 201], xticklabels=[0, 0.001, 0.002], yticks=[0, 50, 101],
                         yticklabels=[0, 0.0005, 0.001])
                # TODO: Change the name
                fig.suptitle("PINN and FE comparison at t=%f s" % (t_value * self.norm_t), fontsize=32)
                plt.savefig(FE_PINN_Comparison + "/Absolute_and_Relative_Error_%f s.png" % (t_value * self.norm_t))

        return Mean_L2_test, max_L2_test, Mean_rel_L2_test, max_rel_L2_test

    def plotting(self, model, val_path, images_path, extrema):
        times = pd.read_csv(val_path + 'Times.csv', index_col='label')
        model.cpu()
        model = model.eval()
        times = times.to_numpy()

        x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], 200), [200, 1])
        y = torch.reshape(torch.linspace(extrema[2, 0], extrema[2, 1], 100), [100, 1])
        X = x.numpy()
        Y = y.numpy()
        X, Y = np.meshgrid(X, Y)
        x_t = torch.reshape(torch.from_numpy(X), [20000, 1])
        y_t = torch.reshape(torch.from_numpy(Y), [20000, 1])

        for t in range(len(times)):
            t_value = times[t, 0] / self.norm_t
            plot_var1 = torch.cat([torch.tensor(()).new_full(size=(20000, 1), fill_value=t_value), x_t, y_t], 1)
            plot_var_final = torch.cat([torch.tensor(()).new_full(size=(20000, 1), fill_value=1), x_t, y_t], 1)
            torch.set_printoptions(profile="full")

            output = abs(model(plot_var1) * self.norm_temp)
            output = torch.reshape(output, [100, 200])
            output = output.detach().numpy()

            fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
            sns.heatmap(output, cmap="rocket", ax=ax, cbar=False, square=True)
            # sns.heatmap(output, cmap="rocket", ax=ax, cbar=False, square=True)
            fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
            ax.invert_yaxis()
            ax.set_xlabel('Time = %f s' % (t_value * self.norm_t))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(2, 95, "Tmax = %f" % (np.amax(output)), fontsize=18, color='white')
            fig.suptitle("PINN Prediction at t=%f s" % (t_value * self.norm_t), fontsize=32)
            # TODO: Change the tick values in the lines below in case you have different number of points in PINN
            plt.setp(ax, xticks=[0, 100, 200], xticklabels=[0, 0.001, 0.002], yticks=[0, 50, 100],
                     yticklabels=[0, 0.0005, 0.001])
            PINN_directory = images_path + "/PINN_Result"
            if not os.path.isdir(PINN_directory):
                os.makedirs(PINN_directory)
            os.path.isdir(PINN_directory)
            plt.savefig(PINN_directory + "/PINN_Prediction_%f s.png" % (t_value * self.norm_t))

        temp = pd.read_csv(val_path + 'Temperature.csv', index_col='label')
        Exact = temp.to_numpy()
        for t in range(len(times)):
            # TODO: Change the denominator below
            Temp_FE = Exact[:, t].reshape(-1, 1)
            Temp_FE = torch.from_numpy(Temp_FE)
            Temp_FE = torch.reshape(Temp_FE, [101, 201])
            Temp_FE = Temp_FE.detach().numpy()

            fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
            fig.subplots_adjust(wspace=0.04)
            sns.heatmap(Temp_FE, cmap="rocket", ax=ax, cbar=False, square=True)
            fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
            ax.set_xlabel('Time = %f s' % (times[t, 0]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
            ax.text(2, 95, "Tmax = %f" % (np.amax(Temp_FE)), fontsize=18, color='white')
            fig.suptitle("FE Result at t=%f s" % (times[t, 0]), fontsize=32)
            plt.setp(ax, xticks=[0, 100, 201], xticklabels=[0, 0.001, 0.002], yticks=[0, 50, 101],
                     yticklabels=[0, 0.0005, 0.001])
            FE_directory = images_path + "/FE_Result"
            if not os.path.isdir(FE_directory):
                os.makedirs(FE_directory)
            os.path.isdir(FE_directory)
            plt.savefig(FE_directory + "/FE_Result_%f s.png" % (times[t, 0]))
