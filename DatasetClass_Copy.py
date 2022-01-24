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
        print(x_coll)
        x_b, y_b = self.Ec.add_boundary_points(self.n_boundary, self.random_seed)

        x_time_internal, y_time_internal = self.Ec.add_initial_points(self.n_initial, self.random_seed)
        if self.n_internal != 0:
            x_internal, y_internal = self.Ec.add_internal_points(self.n_internal, self.random_seed)
            print(y_time_internal, y_internal)
            x_time_internal = torch.cat([x_time_internal, x_internal], 0)
            y_time_internal = torch.cat([y_time_internal, y_internal], 0)

        print("###################################")
        print(x_coll, x_coll.shape, y_coll.shape)
        print(x_time_internal, x_time_internal.shape, y_time_internal.shape)
        print(x_b, x_b.shape, y_b.shape)
        print("###################################")
        '''
        if self.n_collocation == 0:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=1, shuffle=False)
        else:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=fraction_coll,
                                        shuffle=self.shuffle)
        '''
        val_path = "C:/Users/yasa/Desktop/PinnsLV/PinnsLV/"
        node_coords = pd.read_csv(val_path + 'Nodes.csv', index_col='label')
        temp = pd.read_csv(val_path + 'Temperature.csv', index_col='label')
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
        FE_outcome_1 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t1),
                                  x_t, y_t], 1)
        FE_outcome_2 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t2),
                                  x_t, y_t], 1)
        FE_outcome_3 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t3),
                                  x_t, y_t], 1)
        FE_outcome_4 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t4),
                                  x_t, y_t], 1)
        FE_outcome_5 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t5),
                                  x_t, y_t], 1)
        FE_outcome_6 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t6),
                                  x_t, y_t], 1)
        FE_outcome_7 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t7),
                                  x_t, y_t], 1)
        FE_outcome_8 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t8),
                                  x_t, y_t], 1)
        FE_outcome_9 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t9),
                                  x_t, y_t], 1)
        FE_outcome_10 = torch.cat([torch.tensor(()).new_full(size=(node_coords_numpy.shape[0], 1), fill_value=t10),
                                  x_t, y_t], 1)
        Exact = temp.to_numpy()
        Temp_1 = Exact[:, 1].reshape(-1, 1)
        Temp_1 = torch.from_numpy(Temp_1)
        Temp_2 = Exact[:, 2].reshape(-1, 1)
        Temp_2 = torch.from_numpy(Temp_2)
        Temp_3 = Exact[:, 3].reshape(-1, 1)
        Temp_3 = torch.from_numpy(Temp_3)
        Temp_4 = Exact[:, 4].reshape(-1, 1)
        Temp_4 = torch.from_numpy(Temp_4)
        Temp_5 = Exact[:, 5].reshape(-1, 1)
        Temp_5 = torch.from_numpy(Temp_5)
        Temp_6 = Exact[:, 6].reshape(-1, 1)
        Temp_6 = torch.from_numpy(Temp_6)
        Temp_7 = Exact[:, 7].reshape(-1, 1)
        Temp_7 = torch.from_numpy(Temp_7)
        Temp_8 = Exact[:, 8].reshape(-1, 1)
        Temp_8 = torch.from_numpy(Temp_8)
        Temp_9 = Exact[:, 9].reshape(-1, 1)
        Temp_9 = torch.from_numpy(Temp_9)
        Temp_10 = Exact[:, 10].reshape(-1, 1)
        Temp_10 = torch.from_numpy(Temp_10)
        # TODO: Change the names
        plt.scatter(x_t, y_t, c=Temp_5)
        plt.colorbar()
        if self.n_collocation == 0:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(
                FE_outcome_5, Temp_5),
                                        batch_size=1, shuffle=False)
        else:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(
                FE_outcome_5, Temp_5),
                                        batch_size=fraction_coll,
                                        shuffle=self.shuffle)

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
