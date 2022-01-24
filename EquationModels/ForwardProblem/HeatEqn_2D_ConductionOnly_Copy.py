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

        self.type_of_points = "sobol"
        self.output_dimension = 1
        self.space_dimensions = 2
        self.time_dimensions = 1
        self.parameters_values = None
        self.parameter_dimensions = 0
        self.extrema_values = torch.tensor([[0, 1],
                                            [-1, 1],
                                            [-1, 1]])
        self.list_of_BC = list([[self.ub0, self.ub1], [self.ub2, self.ub0]])
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
        self.sigma = 1 / 9
        self.k = 1
        self.rho = 3
        self.cp = 3
        self.a_type = "constant"

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

    ''' def a(self, inputs):
        x = inputs[:, 1]
        if self.a_type == "constant":
            a = 1 * torch.ones_like(x)
            return a
        if self.a_type == "pwc":
            n = 4
            a = 0.5 * torch.ones_like(x)
            a_mod = a
            for i in range(n):
                x_i = -1 + 2 / n * i
                x_i1 = -1 + 2 / n * (i + 1)
                a_i = (0.5 - 0.25) / (1 + 1) * (x_i + 1) + 0.25
                a_mod_new = torch.where((x < x_i1) & (x > x_i), torch.tensor(a_i), a_mod)
                a_mod = a_mod_new
            return a_mod
        if self.a_type == "swc":
            a = 0.25 / (1 + torch.exp(-150 * x)) + 0.25
            return a
        if self.a_type == "linear":
            a = x
            return a '''

    def compute_res(self, network, x_f_train, solid_object):
        x_f_train.requires_grad = True
        u = network(x_f_train.float())[:, 0].reshape(-1, )
        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                     create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_y = grad_u[:, 2]
        # grad_u_tt = torch.autograd.grad(grad_u_t, x_f_train, grad_outputs=torch.ones_like(u),
        # create_graph=True)[0][:, 0]
        grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0][:, 1]
        grad_u_yy = torch.autograd.grad(grad_u_y, x_f_train, grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0][:, 2]

        residual = grad_u_t.reshape(-1, ) - 1 / 9 * (grad_u_xx.reshape(-1, ) + grad_u_yy.reshape(-1, ))

        return residual

    ''' def exact(self, x):
        x_sh_1 = x[:, 1] - self.a(x) * x[:, 0]
        x_sh_2 = x[:, 1] + self.a(x) * x[:, 0]

        u = 0.5 * (self.u0(x_sh_1)[:, 0] + self.u0(x_sh_2)[:, 0])
        v = -2 * pi * torch.sin(2 * pi * x[:, 1]) * torch.sin(2 * pi * x[:, 0])
        return torch.cat([u.reshape(-1, 1), v.reshape(-1, 1)], 1) '''

    def v0(self, x):

        return torch.zeros((x.shape[0], 1))

    def ub0(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub1(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub2(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=1.0)
        return u, type_BC

    def u0(self, x):
        u0 = torch.zeros((x.shape[0], 1)).reshape(-1)
        return u0

    def compute_generalization_error(self, model, val_path, images_path):
        node_coords = pd.read_csv(val_path + 'Nodes.csv', index_col='label')
        temp = pd.read_csv(val_path + 'Temperature.csv', index_col='label')
        model = model.cpu()
        model.eval()
        t1 = 0.01
        t2 = 0.02
        t3 = 0.03
        t4 = 0.05
        t5 = 0.09
        t6 = 0.17
        t7 = 0.33
        t8 = 0.65
        t9 = 0.91
        t10 = 1
        node_coords_numpy = node_coords.to_numpy()
        x_t = node_coords_numpy[:, 0].reshape(-1, 1)
        x_t = torch.from_numpy(x_t)
        y_t = node_coords_numpy[:, 1].reshape(-1, 1)
        y_t = torch.from_numpy(y_t)
        test_input_1 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t1),
                                  x_t, y_t], 1)
        test_input_2 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t2),
                                  x_t, y_t], 1)
        test_input_3 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t3),
                                  x_t, y_t], 1)
        test_input_4 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t4),
                                  x_t, y_t], 1)
        test_input_5 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t5),
                                  x_t, y_t], 1)
        test_input_6 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t6),
                                  x_t, y_t], 1)
        test_input_7 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t7),
                                  x_t, y_t], 1)
        test_input_8 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t8),
                                  x_t, y_t], 1)
        test_input_9 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t9),
                                  x_t, y_t], 1)
        test_input_10 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t10),
                                  x_t, y_t], 1)
        Exact = temp.to_numpy()
        Temp_1 = Exact[:, 1].reshape(-1, 1)
        Temp_2 = Exact[:, 2].reshape(-1, 1)
        Temp_3 = Exact[:, 3].reshape(-1, 1)
        Temp_4 = Exact[:, 4].reshape(-1, 1)
        Temp_5 = Exact[:, 5].reshape(-1, 1)
        Temp_6 = Exact[:, 6].reshape(-1, 1)
        Temp_7 = Exact[:, 7].reshape(-1, 1)
        Temp_8 = Exact[:, 8].reshape(-1, 1)
        Temp_9 = Exact[:, 9].reshape(-1, 1)
        Temp_10 = Exact[:, 10].reshape(-1, 1)
        test_out = model(test_input_5.float())[:, 0].detach().numpy().reshape(-1, 1)
        # TODO: Change the name
        np.savetxt("PINN_Result_time_0.09.csv", test_out)
        assert (Temp_5.shape[1] == test_out.shape[1])
        L2_test = np.sqrt((Temp_5 - test_out) ** 2)
        max_L2_test = np.max(L2_test)
        Mean_L2_test = np.sqrt(np.mean((Temp_5 - test_out) ** 2))
        print("Mean Error Test:", Mean_L2_test)
        print("Max Error Test", max_L2_test)
        rel_L2_test = L2_test / np.mean(Temp_5)
        max_rel_L2_test = np.max(rel_L2_test)
        Mean_rel_L2_test = Mean_L2_test / np.sqrt(np.mean(Temp_5 ** 2))
        print("Mean Relative Error Test:", Mean_rel_L2_test)
        print("Max Relative Error Test", max_rel_L2_test)

        fig, (axis, axis2) = plt.subplots(ncols=2, figsize=(20, 10))
        fig.subplots_adjust(wspace=0.04)
        L2_test = torch.from_numpy(L2_test)
        # TODO: Change the 41s in the line below in case you solve a different model in Abaqus.
        #  41 comes from the fact that there are 1681 nodes and the square root of 1681 is 41.
        L2_test = torch.reshape(L2_test, [41, 41])
        L2_test = L2_test.detach().numpy()
        rel_L2_test = torch.from_numpy(rel_L2_test)
        # TODO: Change the 41s in the line below in case you solve a different model in Abaqus
        rel_L2_test = torch.reshape(rel_L2_test, [41, 41])
        rel_L2_test = rel_L2_test.detach().numpy()
        # torch.set_printoptions(profile="full")
        # print(L2_test[20, ])
        sns.heatmap(L2_test[::-1], cmap="rocket", ax=axis, cbar=False, square=True)
        fig.colorbar(axis.collections[0], ax=axis, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        sns.heatmap(rel_L2_test[::-1], cmap="rocket", ax=axis2, cbar=False, square=True)
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
        plt.setp(axis, xticks=[0, 20, 41], xticklabels=[-1, 0, 1])
        plt.setp(axis2, xticks=[0, 20, 41], xticklabels=[-1, 0, 1])
        plt.setp(axis, yticks=[0, 20, 41], yticklabels=[-1, 0, 1])
        plt.setp(axis2, yticks=[0, 20, 41], yticklabels=[-1, 0, 1])
        # TODO: Change the name
        fig.suptitle("PINN and FE comparison at t=0.09s", fontsize=32)
        plt.savefig(images_path + "/Absolute_and_Relative_Error")
        plt.show()

        fig, (axis, axis2) = plt.subplots(ncols=2, figsize=(20, 10))
        fig.subplots_adjust(wspace=0.04)
        # torch.set_printoptions(profile="full")
        # print(L2_test[20, ])
        # TODO: Change the 20 in the line below in case you solve a different model in Abaqus and
        #  want a different x location
        axis.plot(L2_test[:, 20][::-1])
        # fig.colorbar(axis.collections[0], ax=axis, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # TODO: Change the 20 in the line below in case you solve a different model in Abaqus and
        #  want a different x location
        axis2.plot(rel_L2_test[:, 20][::-1])
        # fig.colorbar(axis2.collections[0], ax=axis2, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        axis.set_title(r'L2 Error')
        axis2.set_title(r'Relative Error')
        axis.set_xlabel(r'y')
        axis2.set_xlabel(r'y')
        axis.set_ylabel(r'L2 Error Values')
        axis2.set_ylabel(r'Relative Error Values')
        axis2.yaxis.tick_right()
        axis2.tick_params(rotation=0)
        axis.set_xticks([])
        axis2.set_xticks([])
        # TODO: Change the name
        fig.suptitle("The difference between PINN and FE at t=0.09s in the middle of the domain",
                     fontsize=32)
        plt.savefig(images_path + "/PINN_FEA_Comparison_at_x=0 (in the middle of domain)")
        plt.show()

        Temp_5 = torch.from_numpy(Temp_5)
        # TODO: Change the 41s in the line below in case you solve a different model in Abaqus
        Temp_5 = torch.reshape(Temp_5, [41, 41])
        Temp_5 = Temp_5.detach().numpy()
        test_out = torch.from_numpy(test_out)
        # TODO: Change the 41s in the line below in case you solve a different model in Abaqus
        test_out = torch.reshape(test_out, [41, 41])
        test_out = test_out.detach().numpy()
        fig, axis = plt.subplots(ncols=1, figsize=(20, 10))
        # torch.set_printoptions(profile="full")
        # print(Temp_begin[:, 20][::-1])
        # TODO: Change the 20 in the line below in case you solve a different model in Abaqus and
        #  want a different x location
        axis.plot(test_out[:, 20][::-1], label="PINN Estimation")
        # fig.colorbar(axis.collections[0], ax=axis, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # TODO: Change the 20 in the line below in case you solve a different model in Abaqus and
        #  want a different x location
        axis.plot(Temp_5[:, 20][::-1], label="FE Estimation")
        # fig.colorbar(axis2.collections[0], ax=axis2, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        axis.set_xlabel(r'y')
        axis.set_ylabel(r'Temperature Values')
        axis.set_xticks([])
        axis.legend()
        # TODO: Change the name
        axis.set_title('Temp. values extracted using PINN and FE at t=0.09s in the middle of the domain', fontsize=32)
        plt.savefig(images_path + "/PINN_FE_Temp_Estimation_at_x=0")
        plt.show()

        if images_path is not None:
            plt.figure(figsize=(20, 10))
            plt.title("Time = 0.09s")
            plt.grid(True, which="both", ls=":")
            plt.scatter(Temp_5, test_out)
            plt.xlabel(r'Exact Values')
            plt.ylabel(r'Predicted Values')
            plt.savefig(images_path + "/Exact_Predicted_Values_Comparison")
            plt.show()

        return Mean_L2_test, Mean_rel_L2_test

    ''' def plotting(self, model, images_path, extrema, solid):
        model.cpu()
        model = model.eval()
        x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], 100), [100, 1])
        time_steps = [0.0, 0.25, 0.5, 0.75, 1]
        scale_vec = np.linspace(0.65, 1.55, len(time_steps))

        plt.figure()
        plt.grid(True, which="both", ls=":")
        for val, scale in zip(time_steps, scale_vec):
            plot_var = torch.cat([torch.tensor(()).new_full(size=(100, 1), fill_value=val), x,
                                  torch.tensor(()).new_full(size=(100, 1), fill_value=1)], 1)
            plt.plot(x, self.exact(plot_var)[:, 0], 'b-', linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$',
                     color=self.lighten_color('grey', scale), zorder=0)
            plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var)[:, 0].detach().numpy(),
                        label=r'Predicted, $t=$' + str(val) + r'$s$', marker="o", s=14,
                        color=self.lighten_color('C0', scale), zorder=10)

        plt.xlabel(r'$x$')
        plt.ylabel(r'u')
        plt.legend()
        plt.savefig(images_path + "/Samples.png", dpi=500) '''

    def plotting(self, model, val_path, images_path, extrema):
        model.cpu()
        model = model.eval()
        x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], 100), [100, 1])
        y = torch.reshape(torch.linspace(extrema[2, 0], extrema[2, 1], 100), [100, 1])
        t1 = 0.01
        t2 = 0.02
        t3 = 0.03
        t4 = 0.05
        t5 = 0.09
        t6 = 0.17
        t7 = 0.33
        t8 = 0.65
        t9 = 0.91
        t10 = 1
        X = x.numpy()
        Y = y.numpy()
        X, Y = np.meshgrid(X, Y)
        x_t = torch.reshape(torch.from_numpy(X), [10000, 1])
        y_t = torch.reshape(torch.from_numpy(Y), [10000, 1])
        plot_var1 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t1), x_t, y_t], 1)
        plot_var2 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t2), x_t, y_t], 1)
        plot_var3 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t3), x_t, y_t], 1)
        plot_var4 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t4), x_t, y_t], 1)
        plot_var5 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t5), x_t, y_t], 1)
        plot_var6 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t6), x_t, y_t], 1)
        plot_var7 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t7), x_t, y_t], 1)
        plot_var8 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t8), x_t, y_t], 1)
        plot_var9 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t9), x_t, y_t], 1)
        plot_var10 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t10), x_t, y_t], 1)
        torch.set_printoptions(profile="full")

        # output1 = model(plot_var1)
        # output1 = torch.reshape(output1, [100, 100])
        # output1 = output1.detach().numpy()
        # output2 = model(plot_var2)
        # output2 = torch.reshape(output2, [100, 100])
        # output2 = output2.detach().numpy()
        # output3 = model(plot_var3)
        # output3 = torch.reshape(output3, [100, 100])
        # output3 = output3.detach().numpy()
        # output4 = model(plot_var4)
        # output4 = torch.reshape(output4, [100, 100])
        # output4 = output4.detach().numpy()
        output5 = model(plot_var5)
        output5 = torch.reshape(output5, [100, 100])
        output5 = output5.detach().numpy()
        # output6 = model(plot_var6)
        # output6 = torch.reshape(output6, [100, 100])
        # output6 = output6.detach().numpy()
        # output7 = model(plot_var7)
        # output7 = torch.reshape(output7, [100, 100])
        # output7 = output7.detach().numpy()
        # output8 = model(plot_var8)
        # output8 = torch.reshape(output8, [100, 100])
        # output8 = output8.detach().numpy()
        # output9 = model(plot_var9)
        # output9 = torch.reshape(output9, [100, 100])
        # output9 = output9.detach().numpy()
        # output10 = model(plot_var10)
        # output10 = torch.reshape(output10, [100, 100])
        # output10 = output10.detach().numpy()

        '''
        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
        # fig.subplots_adjust(wspace=0.04)
        sns.heatmap(output1, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2, shrink=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Time = 0.01s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.invert_yaxis()
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        fig.suptitle("PINN Prediction at t=0.01s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you have different number of points in PINN
        plt.setp(ax, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/PINN_Prediction_0.01s.png")
        plt.show()
        

        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
        # fig.subplots_adjust(wspace=0.04)
        sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2, shrink=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Time = 0.02s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.invert_yaxis()
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        fig.suptitle("PINN Prediction at 0.02s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you have different number of points in PINN
        plt.setp(ax, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/PINN_Prediction_0.02s.png")
        plt.show()
        '''
        '''
        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
        # fig.subplots_adjust(wspace=0.04)
        sns.heatmap(output3, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2, shrink=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Time = 0.03s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.invert_yaxis()
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        fig.suptitle("PINN Prediction at 0.03s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you have different number of points in PINN
        plt.setp(ax, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/PINN_Prediction_0.03s.png")
        plt.show()

        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
        # fig.subplots_adjust(wspace=0.04)
        sns.heatmap(output4, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2, shrink=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Time = 0.05s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.invert_yaxis()
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        fig.suptitle("PINN Prediction at 0.05s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you have different number of points in PINN
        plt.setp(ax, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/PINN_Prediction_0.05s.png")
        plt.show()
        '''

        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
        # fig.subplots_adjust(wspace=0.04)
        sns.heatmap(output5, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2, shrink=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Time = 0.09s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.invert_yaxis()
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        fig.suptitle("PINN Prediction at 0.09s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you have different number of points in PINN
        plt.setp(ax, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/PINN_Prediction_0.09s.png")
        plt.show()

        '''
        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
        # fig.subplots_adjust(wspace=0.04)
        sns.heatmap(output6, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2, shrink=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Time = 0.17s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.invert_yaxis()
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        fig.suptitle("PINN Prediction at 0.17s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you have different number of points in PINN
        plt.setp(ax, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/PINN_Prediction_0.17s.png")
        plt.show()

        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
        # fig.subplots_adjust(wspace=0.04)
        sns.heatmap(output7, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2, shrink=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Time = 0.33s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.invert_yaxis()
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        fig.suptitle("PINN Prediction at 0.33s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you have different number of points in PINN
        plt.setp(ax, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/PINN_Prediction_0.33s.png")
        plt.show()

        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
        # fig.subplots_adjust(wspace=0.04)
        sns.heatmap(output8, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2, shrink=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Time = 0.65s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.invert_yaxis()
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        fig.suptitle("PINN Prediction at 0.65s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you have different number of points in PINN
        plt.setp(ax, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/PINN_Prediction_0.65s.png")
        plt.show()

        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
        # fig.subplots_adjust(wspace=0.04)
        sns.heatmap(output9, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2, shrink=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Time = 0.97s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.invert_yaxis()
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        fig.suptitle("PINN Prediction at 0.97s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you have different number of points in PINN
        plt.setp(ax, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/PINN_Prediction_0.97s.png")
        plt.show()

        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
        # fig.subplots_adjust(wspace=0.04)
        sns.heatmap(output10, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2, shrink=0.6)
        # sns.heatmap(output2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2, shrink=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Time = 1s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.invert_yaxis()
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        fig.suptitle("PINN Prediction at 0.97s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you have different number of points in PINN
        plt.setp(ax, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 50, 100], xticklabels=[-1, 0, 1], yticks=[0, 50, 100], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/PINN_Prediction_1s.png")
        plt.show()
        '''

        temp = pd.read_csv(val_path + 'Temperature.csv', index_col='label')
        Exact = temp.to_numpy()
        # TODO: Change the lines below in case you solve a different model in Abaqus and
        #  want a different time value
        Temp_1 = Exact[:, 1].reshape(-1, 1)
        Temp_1 = torch.from_numpy(Temp_1)
        Temp_1 = torch.reshape(Temp_1, [41, 41])
        Temp_1 = Temp_1.detach().numpy()
        Temp_2 = Exact[:, 2].reshape(-1, 1)
        Temp_2 = torch.from_numpy(Temp_2)
        Temp_2 = torch.reshape(Temp_2, [41, 41])
        Temp_2 = Temp_2.detach().numpy()
        Temp_3 = Exact[:, 3].reshape(-1, 1)
        Temp_3 = torch.from_numpy(Temp_3)
        Temp_3 = torch.reshape(Temp_3, [41, 41])
        Temp_3 = Temp_3.detach().numpy()
        Temp_4 = Exact[:, 4].reshape(-1, 1)
        Temp_4 = torch.from_numpy(Temp_4)
        Temp_4 = torch.reshape(Temp_4, [41, 41])
        Temp_4 = Temp_4.detach().numpy()
        Temp_5 = Exact[:, 5].reshape(-1, 1)
        Temp_5 = torch.from_numpy(Temp_5)
        Temp_5 = torch.reshape(Temp_5, [41, 41])
        Temp_5 = Temp_5.detach().numpy()
        Temp_6 = Exact[:, 6].reshape(-1, 1)
        Temp_6 = torch.from_numpy(Temp_6)
        Temp_6 = torch.reshape(Temp_6, [41, 41])
        Temp_6 = Temp_6.detach().numpy()
        Temp_7 = Exact[:, 7].reshape(-1, 1)
        Temp_7 = torch.from_numpy(Temp_7)
        Temp_7 = torch.reshape(Temp_7, [41, 41])
        Temp_7 = Temp_7.detach().numpy()
        Temp_8 = Exact[:, 8].reshape(-1, 1)
        Temp_8 = torch.from_numpy(Temp_8)
        Temp_8 = torch.reshape(Temp_8, [41, 41])
        Temp_8 = Temp_8.detach().numpy()
        Temp_9 = Exact[:, 9].reshape(-1, 1)
        Temp_9 = torch.from_numpy(Temp_9)
        Temp_9 = torch.reshape(Temp_9, [41, 41])
        Temp_9 = Temp_9.detach().numpy()
        Temp_10 = Exact[:, 10].reshape(-1, 1)
        Temp_10 = torch.from_numpy(Temp_10)
        Temp_10 = torch.reshape(Temp_10, [41, 41])
        Temp_10 = Temp_10.detach().numpy()

        # fig, (ax, ax2) = plt.subplots(ncols=1, figsize=(20, 10))
        fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
        fig.subplots_adjust(wspace=0.04)
        sns.heatmap(Temp_5, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False, square=True)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
        # sns.heatmap(Temp_final, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False, square=True)
        # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
        # TODO: Change the name
        ax.set_xlabel('Time = 0.09s')
        # ax2.set_xlabel('Time = 1s')
        # ax2.yaxis.tick_right()
        # ax2.tick_params(rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        # TODO: Change the name
        fig.suptitle("FE Prediction at t=0.09s", fontsize=32)
        # TODO: Change the tick values in the lines below in case you solve a different model in Abaqus
        plt.setp(ax, xticks=[0, 20, 41], xticklabels=[-1, 0, 1], yticks=[0, 20, 41], yticklabels=[-1, 0, 1])
        # plt.setp(ax2, xticks=[0, 20, 41], xticklabels=[-1, 0, 1], yticks=[0, 20, 41], yticklabels=[-1, 0, 1])
        plt.savefig(images_path + "/FE_Prediction.png")
        plt.show()
