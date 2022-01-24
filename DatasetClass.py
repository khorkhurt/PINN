import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class DefineDataset:
    def __init__(self, Ec, n_collocation, n_boundary, n_initial, n_internal, batches, random_seed, shuffle=False):
        self.Ec = Ec
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        self.n_internal = n_internal
        self.batches = batches
        self.random_seed = random_seed
        self.shuffle = shuffle

        self.space_dimensions = self.Ec.space_dimensions
        self.time_dimensions = self.Ec.time_dimensions
        self.input_dimensions = self.Ec.space_dimensions + self.Ec.time_dimensions
        self.output_dimension = self.Ec.output_dimension
        self.n_samples = self.n_collocation + 2 * self.n_boundary * self.space_dimensions + self.n_initial * \
                         self.time_dimensions + self.n_internal
        self.BC = None
        self.data_coll = None
        self.data_boundary = None
        self.data_initial_internal = None

        if self.batches == "full":
            self.batches = int(self.n_samples)
        else:
            self.batches = int(self.batches)

    def assemble_dataset(self):

        fraction_coll = int(self.batches * self.n_collocation / self.n_samples)
        fraction_boundary = int(self.batches * 2 * self.n_boundary * self.space_dimensions / self.n_samples)
        fraction_initial = int(self.batches * self.n_initial / self.n_samples)
        fraction_internal = int(self.batches * self.n_internal / self.n_samples)

        x_coll, y_coll = self.Ec.add_collocation_points(self.n_collocation, self.random_seed)
        x_b, y_b = self.Ec.add_boundary_points(self.n_boundary, self.random_seed)

        x_time_internal, y_time_internal = self.Ec.add_initial_points(self.n_initial, self.random_seed)
        if self.n_internal != 0:
            x_internal, y_internal = self.Ec.add_internal_points(self.n_internal, self.random_seed)
            x_time_internal = torch.cat([x_time_internal, x_internal], 0)
            y_time_internal = torch.cat([y_time_internal, y_internal], 0)

        print("###################################")
        print(x_coll, x_coll.shape, y_coll.shape)
        np.savetxt("x_coll.csv", x_coll.detach().numpy())
        np.savetxt("y_coll.csv", y_coll.detach().numpy())

        '''
        import os
        images_path_xy = "C:/Users/yasa/Desktop/PinnsLV_v3/PinnsLV/" \
                      "HeatEqn_3D_ConductionOnly_Temp_Independent_SLM/" \
                      "Images/Collocation_Points/xy"
        images_path_xz = "C:/Users/yasa/Desktop/PinnsLV_v3/PinnsLV/" \
                      "HeatEqn_3D_ConductionOnly_Temp_Independent_SLM/" \
                      "Images/Collocation_Points/xz"
        images_path_yz = "C:/Users/yasa/Desktop/PinnsLV_v3/PinnsLV/" \
                      "HeatEqn_3D_ConductionOnly_Temp_Independent_SLM/" \
                      "Images/Collocation_Points/yz"
        iteration = 0.00001
        x = []
        y = []
        z = []
        for t in np.arange(0, 0.006, iteration):
            x = []
            y = []
            z = []
            for i in range(x_coll.shape[0]):
                # print(t)
                if t <= x_coll[i, 0] <= (t + iteration):
                    # print(t)
                    x.append(x_coll[i, 1])
                    y.append(x_coll[i, 2])
                    z.append(x_coll[i, 3])
            # Collocation_Points = images_path + "/Collocation_Points_%d" % ((t / 0.025) + 1)
            Collocation_Points_xy = images_path_xy
            if not os.path.isdir(Collocation_Points_xy):
                os.makedirs(Collocation_Points_xy)
            os.path.isdir(Collocation_Points_xy)
            Collocation_Points_xz = images_path_xz
            if not os.path.isdir(Collocation_Points_xz):
                os.makedirs(Collocation_Points_xz)
            os.path.isdir(Collocation_Points_xz)
            Collocation_Points_yz = images_path_yz
            if not os.path.isdir(Collocation_Points_yz):
                os.makedirs(Collocation_Points_yz)
            os.path.isdir(Collocation_Points_yz)
            fig, (axis) = plt.subplots(ncols=1, figsize=(20, 10))
            plt.scatter(x, y)
            axis.set_xlabel(r'x')
            axis.set_ylabel(r'y')
            plt.title("Collocation points for the time interval between %f and %f)" % (t, t + iteration),
                        fontsize=32)
            plt.savefig(Collocation_Points_xy + "/Collocation_Points_%f.png" % t)
            fig, (axis) = plt.subplots(ncols=1, figsize=(20, 10))
            plt.scatter(x, z)
            axis.set_xlabel(r'x')
            axis.set_ylabel(r'z')
            plt.title("Collocation points for the time interval between %f and %f)" % (t, t + iteration),
                      fontsize=32)
            plt.savefig(Collocation_Points_xz + "/Collocation_Points_%f.png" % t)
            fig, (axis) = plt.subplots(ncols=1, figsize=(20, 10))
            plt.scatter(y, z)
            axis.set_xlabel(r'y')
            axis.set_ylabel(r'z')
            plt.title("Collocation points for the time interval between %f and %f)" % (t, t + iteration),
                      fontsize=32)
            plt.savefig(Collocation_Points_yz + "/Collocation_Points_%f.png" % t)
        quit()
        '''
        if self.n_collocation == 0:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=1, shuffle=False)
        else:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=x_coll.shape[0],
                                        shuffle=self.shuffle)
        '''  
        else:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=fraction_coll,
                                        shuffle=self.shuffle)
        if self.n_collocation == 0:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(
                FE_outcome_all_data, Temp_all_data),
                                        batch_size=1, shuffle=False)
        else:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(
                FE_outcome_all_data, Temp_all_data),
                                        batch_size=fraction_coll,
                                        shuffle=self.shuffle)

        if self.n_collocation == 0:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(
                Supported_data_x, Supported_data_y),
                                        batch_size=1, shuffle=False)
        else:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(
                Supported_data_x, Supported_data_y),
                                        batch_size=fraction_coll+FE_outcome_all_data.shape[0],
                                        shuffle=self.shuffle)
        '''
        if self.n_boundary == 0:
            self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=1, shuffle=False)
        else:
            self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=fraction_boundary,
                                            shuffle=self.shuffle)

        if fraction_internal == 0 and fraction_initial == 0:
            self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal),
                                                    batch_size=1, shuffle=False)
        else:
            self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal),
                                                    batch_size=fraction_initial + fraction_internal,
                                                    shuffle=self.shuffle)
