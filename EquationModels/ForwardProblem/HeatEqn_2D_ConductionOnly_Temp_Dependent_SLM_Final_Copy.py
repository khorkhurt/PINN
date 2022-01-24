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

        ###########

        # Definition of conductivity
        self.k = 9.206
        # Definition of density
        self.rho = 8220
        # Definition of heat capacity
        self.cp = 485.9975

    def add_collocation_points(self, n_coll, random_seed):
        return self.square_domain.add_collocation_points(n_coll, random_seed)

    def add_boundary_points(self, n_boundary, random_seed):
        return self.square_domain.add_boundary_points(n_boundary, random_seed)

    def add_initial_points(self, n_initial, random_seed):
        extrema_0 = self.extrema_values[:, 0]
        extrema_f = self.extrema_values[:, 1]
        x_time_0 = generator_points(n_initial, self.time_dimensions + self.space_dimensions + self.parameter_dimensions,
                                    random_seed, self.type_of_points, True)
        # x_time_0[:, 0] = torch.full(size=(n_initial,), fill_value=0.0)
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
                # v = self.v0(x_u_train)[:, j]
                u_pred_var_list.append(u)
                u_train_var_list.append(u_train[:])
                # u_pred_var_list.append(grad_u_t)
                # u_train_var_list.append(v)

    def add_internal_points(self, n_internal, random_seed):
        val_path = "C:/Users/yasa/Desktop/PinnsLV/PinnsLV/"
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
        u = network(x_f_train.float())[:, 0].reshape(-1, )
        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                     create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_y = grad_u[:, 2]
        '''
        val_path = "C:/Users/yasa/Desktop/PinnsLV/PinnsLV/"
        times = pd.read_csv(val_path + 'Times.csv', index_col='label')
        times = times.to_numpy()

        x = torch.reshape(torch.linspace(0, 1, 200), [200, 1])
        y = torch.reshape(torch.linspace(0, 1, 100), [100, 1])
        X = x.numpy()
        Y = y.numpy()
        X, Y = np.meshgrid(X, Y)
        x_t = torch.reshape(torch.from_numpy(X), [20000, 1])
        y_t = torch.reshape(torch.from_numpy(Y), [20000, 1])

        for t in range(len(times)):
            t = times[t, 0] / self.norm_t
            plot_var1 = torch.cat([torch.tensor(()).new_full(size=(20000, 1), fill_value=t), x_t, y_t], 1)
            torch.set_printoptions(profile="full")
            q = self.source(plot_var1)
            q = torch.reshape(q, [100, 200])
            q = q.detach().numpy()
            fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
            sns.heatmap(q, cmap="rocket", ax=ax, cbar=False, square=True)
            fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
            ax.invert_yaxis()
            ax.set_title(r'q')
            ax.set_xlabel(r'x')
            ax.set_ylabel(r'y')
            ax.set_xticks([])
            ax.set_yticks([])
            # TODO: Change the tick values in the lines below in case you solve a different model in Abaqus
            plt.setp(ax, xticks=[0, 100, 201], xticklabels=[0, 0.001, 0.002], yticks=[0, 50, 101],
                     yticklabels=[0, 0.0005, 0.001])
            images_path = "C:/Users/yasa/Desktop/PinnsLV/PinnsLV/" \
                          "HeatEqn_2D_ConductionOnly_Temp_Independent_SLM/Images"
            Heat_Source_directory = images_path + "/Heat_Source"
            if not os.path.isdir(Heat_Source_directory):
                os.makedirs(Heat_Source_directory)
            os.path.isdir(Heat_Source_directory)
            plt.savefig(Heat_Source_directory + "/Heat_Source_at_%f s.png" % (t * self.norm_t))
        quit()
        '''
        # q = self.source(x_f_train)
        # grad_u_tt = torch.autograd.grad(grad_u_t, x_f_train, grad_outputs=torch.ones_like(u),
        # create_graph=True)[0][:, 0]
        # Temp Dependent Part

        '''
        from NN2 import model as third_neural_network
        values_cp = third_neural_network(abs((u * self.norm_temp).reshape(-1, 1)))
        cp = values_cp.reshape(-1, )
        # print(cp, cp.shape)
        '''

        cp = self.heat_capacity(u)
        rho = self.density(u)
        k = self.conductivity(u)

        q = self.source(x_f_train)
        '''
        grad_k_x = torch.autograd.grad(k, x_f_train, grad_outputs=torch.ones_like(u),
                                       create_graph=True)[0][:, 1]
        grad_k_y = torch.autograd.grad(k, x_f_train, grad_outputs=torch.ones_like(u),
                                       create_graph=True)[0][:, 2]

        np.savetxt("grad_k_x.csv", grad_k_x.detach().numpy())
        np.savetxt("grad_u_x.csv", grad_u_x.detach().numpy())
        '''
        grad_u_xx = torch.autograd.grad(k * grad_u_x, x_f_train, grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0][:, 1]
        grad_u_yy = torch.autograd.grad(k * grad_u_y, x_f_train, grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0][:, 2]


        residual = grad_u_t.reshape(-1, ) - (self.norm_t / rho / cp) * (((grad_u_xx.reshape(-1, )) /
                                                                                   (self.norm_x ** 2))
                                                                                  + ((grad_u_yy.reshape(-1, )) /
                                                                                     (self.norm_y ** 2))) \
                   - q * self.norm_t / (rho * cp * self.norm_temp)


        '''
        residual = (self.rho * cp) * grad_u_t.reshape(-1, ) - (self.norm_t) * (((grad_u_xx.reshape(-1, )) /
                                                                                   (self.norm_x ** 2))
                                                                                  + ((grad_u_yy.reshape(-1, )) /
                                                                                     (self.norm_y ** 2))) \
                   - q * self.norm_t / (self.norm_temp)
        '''

        '''
        residual = grad_u_t.reshape(-1, ) - (self.norm_t / self.rho / self.cp) * (grad_u_xx.reshape(-1, ) /
                                                                                  (self.norm_x ** 2)
                                                                                  + grad_u_yy.reshape(-1, ) /
                                                                                  (self.norm_y ** 2)) \
                   - q * self.norm_t / (self.rho * self.cp * self.norm_temp)
        '''
        source = q * self.norm_t / (rho * cp * self.norm_temp)
        residual = residual / torch.sqrt(torch.mean(source ** 2)).detach()
        '''
        # Temp Independent Part

        grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0][:, 1]
        grad_u_yy = torch.autograd.grad(grad_u_y, x_f_train, grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0][:, 2]

        residual = grad_u_t.reshape(-1, ) - (self.k * self.norm_t / self.rho / self.cp) * (grad_u_xx.reshape(-1, ) /
                                                                                           (self.norm_x ** 2)
                                                                                           + grad_u_yy.reshape(-1, ) /
                                                                                           (self.norm_y ** 2)) \
                   - q * self.norm_t / (self.rho * self.cp * self.norm_temp)
        '''
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

    def ub2(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0)
        return u, type_BC

    def u0(self, x):
        images_path = "C:/Users/yasa/Desktop/PinnsLV/PinnsLV/" \
                      "HeatEqn_2D_ConductionOnly_Temp_Independent_SLM" \
                      "/Images"
        np.savetxt("x_initial_coord.csv", x.detach().numpy(), fmt='%1.10f')
        '''
        u0 = self.norm_temp * torch.exp(-((x[:, 1] * self.norm_x - 0.001) ** 2 + (x[:, 2] * self.norm_y - 0.0005) ** 2)
                                        * 10000 ** 2) / self.norm_temp
        np.savetxt("u0.csv", u0.detach().numpy(), fmt='%1.5f')

        X = np.linspace(0, 0.002, 2000)
        Y = np.linspace(0, 0.001, 1000)
        X, Y = np.meshgrid(X, Y)
        u1 = self.norm_temp * torch.exp(-((torch.from_numpy(X) - 0.001) ** 2 +
                                          (torch.from_numpy(Y) - 0.0005) ** 2) * 10000 ** 2)

        Initial_Temperature = images_path + "/Initial_Temperature"
        if not os.path.isdir(Initial_Temperature):
            os.makedirs(Initial_Temperature)
        os.path.isdir(Initial_Temperature)
        u1 = u1.detach().numpy()

        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        fig.subplots_adjust(wspace=0.04)
        cp = ax.contourf(X, Y, u1, vmin=0, vmax=self.norm_temp)
        fig.colorbar(cp)
        ax.set_xlabel('Time = 0 s')
        plt.savefig(Initial_Temperature + "/Initial_Temperature_Distribution.png")
        '''
        u0 = torch.zeros((x.shape[0], 1)).reshape(-1)
        return u0

    def source(self, x):
        np.savetxt("x.csv", x.detach().numpy(), fmt='%1.3f')
        heat = (6 * (10 ** 13)) * torch.exp(-(((x[:, 0] * self.norm_t - 0.005) * 1000) ** 2)) * \
               torch.exp(-2 * (((x[:, 1] * self.norm_x - (0.001)) ** 2) +
                               (((x[:, 2] * self.norm_y - (0.0005))) ** 2)) / (0.00005 ** 2))
        np.savetxt("heat.csv", heat.detach().numpy())
        return heat

    def conductivity(self, u):
        u = abs(u * self.norm_temp)
        # u = ((u - min(u))/(max(u)-min(u)))*self.norm_temp
        # np.savetxt("u.csv", u.detach().numpy())
        k = torch.empty(u.shape)
        '''
        mask1 = ~torch.ge(u, 25)
        k[mask1] = 9
        mask2 = torch.ge(u, 25) & ~torch.ge(u, 1260)
        k[mask2] = 0.0205 * u[mask2] + 8.6923
        mask3 = torch.ge(u, 1260)
        k[mask3] = 0.6476 * u[mask3] - 781.37
        k2 = 0.0205 * u + 8.6923
        plt.figure()
        plt.scatter(u.detach(), k.detach())
        plt.scatter(u.detach(), k2.detach())
        plt.show()
        quit()
        '''

        '''
        k1 = torch.full(size=(u.shape), fill_value=9)
        k2 = 0.0205 * (u - 1260) + 34.6
        k3 = 0.6476 * (u - 1260) + 34.6
        combined = torch.cat((torch.maximum(k1, k2).reshape(-1, 1), torch.maximum(k1, k3).reshape(-1, 1),
                              torch.maximum(k2, k3).reshape(-1, 1)), 1)
        k4, inds = torch.max(combined, 1)
        '''
        '''
        k = 1.010617724 * (10 ** -31) * (u ** 10) - 3.1292516601297 * (10 ** -27) * (u ** 9) + \
            4.12709173812185 * (10 ** -23) * (u ** 8) - 3.01538623391569 * (10 ** -19) * (u ** 7) + \
            1.32724956675736 * (10 ** -15) * (u ** 6) - 3.57427621883027 * (10 ** -12) * (u ** 5) + \
            5.68277911560219 * (10 ** -9) * (u ** 4) - 0.0000048278515531167 * (u ** 3) + \
            0.00191857322253436 * (u ** 2) - 0.260757223496577 * u + 16.9532462453248
        '''
        k = 14.6 + 1.27 * (10 ** -2) * u
        # k = 9.705 + 0.0176 * u - 0.0000016 * (u ** 2)
        # plt.scatter(u.detach(), k.detach())
        # plt.scatter(u.detach(), k4.detach())
        # plt.show()
        # quit()
        # np.savetxt("k.csv", k.detach().numpy())
        return k

    def heat_capacity(self, u):
        u = abs(u * self.norm_temp)
        # np.savetxt("u.csv", u.detach().numpy())
        cp = torch.empty(u.shape)
        '''
        for i in range(u.shape[0]):
            if 0 <= u[i] <= 1260 or 1460 < u[i]:
                cp[i] = 0.3483 * u[i] + 477.29
            elif 1260 <= u[i] <= 1460:
                cp[i] = 0.3483 * u[i] + 1857.3
            else:
                cp[i] = 0.00001
            i += 1
        '''
        cp = 450 + 0.28 * u - 2.91 * (10 ** -4) * (u ** 2) + 1.34 * (10 ** -7) * (u ** 3)
        # cp = 6.683 + 0.04906 * u + 80.74 * torch.log(u)
        # cp = 6.683 + 0.04906 * u + 80.74 * (u ** (2))
        # np.savetxt("cp.csv", cp.detach().numpy())
        return cp

    def density(self, u):
        u = abs(u * self.norm_temp)
        # np.savetxt("u.csv", u.detach().numpy())
        rho = torch.empty(u.shape)

        rho = -0.0002 * (u ** 2) - 0.2539 * u + 8220.6
        # np.savetxt("rho.csv", rho.detach().numpy())
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
        Exact = temp.to_numpy()
        Exact = Exact[:, len(times) - 1].reshape(-1, 1)
        # Exact = Exact[:, 12].reshape(-1, 1)
        print(np.amax(Exact))
        test_out = model(test_input.float())[:, 0].detach().numpy().reshape(-1, 1) * self.norm_temp
        print(np.amax(test_out))
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
            result_path = "C:/Users/yasa/Desktop/PinnsLV/PinnsLV/" \
                          "HeatEqn_2D_ConductionOnly_Temp_Independent_SLM"
            PINN_Results_CSV = images_path + "/PINN_Results_CSV"
            if not os.path.isdir(PINN_Results_CSV):
                os.makedirs(PINN_Results_CSV)
            os.path.isdir(PINN_Results_CSV)
            t_value = times[t, 0] / self.norm_t
            values = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t_value),
                                x_t, y_t], 1)

            pinn_output = model(values.float())[:, 0].detach().numpy().reshape(-1, 1) * self.norm_temp
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
                sns.heatmap(L2_test, cmap="rocket", ax=axis, vmin=0, vmax=30, cbar=False, square=True)
                fig.colorbar(axis.collections[0], ax=axis, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
                axis.invert_yaxis()
                axis.set_title(r'L2 Error')
                axis.set_xlabel(r'x')
                axis.set_ylabel(r'y')
                axis.set_xticks([])
                axis.set_yticks([])
                # TODO: Change the tick values in the lines below in case you solve a different model in Abaqus
                plt.setp(axis, xticks=[0, 100, 201], xticklabels=[0, 0.001, 0.002], yticks=[0, 50, 101],
                         yticklabels=[0, 0.0005, 0.001])
                # TODO: Change the name
                fig.suptitle("PINN and FE comparison at t=0 s", fontsize=32)
                plt.savefig(FE_PINN_Comparison + "/Absolute_and_Relative_Error_0 s.png")

            else:
                L2_test = np.sqrt((Temp_FE - pinn_output) ** 2)
                # rel_L2_test = L2_test / np.mean(Exact)
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
                sns.heatmap(L2_test, cmap="rocket", ax=axis, vmin=0, vmax=30, cbar=False, square=True)
                fig.colorbar(axis.collections[0], ax=axis, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
                sns.heatmap(rel_L2_test, cmap="rocket", ax=axis2, vmin=0, vmax=0.5, cbar=False, square=True)
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

        # t = torch.reshape(torch.linspace(extrema[0, 0], extrema[0, 1], 100), [100, 1])
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
            # plot_var1 = torch.cat([torch.tensor(()).new_full(size=(1, 1), fill_value=0.025), 0.5, 0.5], 1)
            torch.set_printoptions(profile="full")

            output = model(plot_var1) * self.norm_temp
            output = torch.reshape(output, [100, 200])
            output = output.detach().numpy()

            fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
            sns.heatmap(output, cmap="rocket", ax=ax, cbar=False, square=True)
            fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
            ax.invert_yaxis()
            ax.set_xlabel('Time = %f s' % (t_value * self.norm_t))
            ax.set_xticks([])
            ax.set_yticks([])
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
            sns.heatmap(Temp_FE, cmap="rocket", ax=ax, vmin=0, vmax=2000, cbar=False, square=True)
            fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
            ax.set_xlabel('Time = %f s' % (times[t, 0]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
            fig.suptitle("FE Result at t=%f s" % (times[t, 0]), fontsize=32)
            plt.setp(ax, xticks=[0, 100, 201], xticklabels=[0, 0.001, 0.002], yticks=[0, 50, 101],
                     yticklabels=[0, 0.0005, 0.001])
            FE_directory = images_path + "/FE_Result"
            if not os.path.isdir(FE_directory):
                os.makedirs(FE_directory)
            os.path.isdir(FE_directory)
            plt.savefig(FE_directory + "/FE_Result_%f s.png" % (times[t, 0]))
