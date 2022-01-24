import torch
import numpy as np
import sobol_seq
from pyDOE import lhs
from scipy.stats import lognorm
import matplotlib.pyplot as plt
# from scipy.stats import qmc


def generator_points(samples, dim, random_seed, type_of_points, boundary):
    if type_of_points == "random":
        torch.random.manual_seed(random_seed)
        return torch.rand([samples, dim]).type(torch.FloatTensor)
    elif type_of_points == "lhs":
        return torch.from_numpy(lhs(dim, samples=samples, criterion='center')).type(torch.FloatTensor)
    elif type_of_points == "gauss":
        if samples != 0:

            x, _ = np.polynomial.legendre.leggauss(samples)
            x = 0.5 * (x.reshape(-1, 1) + 1)

            if dim == 1:
                return torch.from_numpy(x).type(torch.FloatTensor)
            if dim == 2:
                x = x.reshape(-1, )
                x = np.transpose([np.repeat(x, len(x)), np.tile(x, len(x))])
                return torch.from_numpy(x).type(torch.FloatTensor)
        else:
            return torch.zeros([0, dim])
    elif type_of_points == "grid":
        if samples != 0:

            x = np.linspace(0, 1, samples + 2)
            x = x[1:-1].reshape(-1, 1)
            if dim == 1:
                return torch.from_numpy(x).type(torch.FloatTensor)
            if dim == 2:
                x = x.reshape(-1, )
                if not boundary:
                    x = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
                else:
                    x = np.concatenate([x.reshape(-1, 1), x.reshape(-1, 1)], 1)
                print(x)
                return torch.from_numpy(x).type(torch.FloatTensor)
        else:
            return torch.zeros([0, dim])

    elif type_of_points == "sobol":
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
        return torch.from_numpy(data).type(torch.FloatTensor)

    elif type_of_points == "sobol_center":
        skip = random_seed
        data1 = np.full((samples, dim - 1), np.nan)
        data2 = np.full((samples, 1), np.nan)
        for j in range(samples):
            seed = j + skip
            data1[j, :], next_seed = sobol_seq.i4_sobol(dim - 1, seed)
            data2[j, :] = np.random.normal(0.5, 0.1, 1)
        data = np.concatenate((data1, data2), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)

    elif type_of_points == "sobol_center_2":
        skip = random_seed
        data1 = np.full((int(samples / 2), 1), np.nan) # t1
        data2 = np.full((int(samples / 2), 1), np.nan) # x1
        data3 = np.full((int(samples / 2), 1), np.nan) # y1
        data4 = np.full((int(samples / 2), 1), np.nan) # t2
        data5 = np.full((int(samples / 2), 1), np.nan) # x2
        data6 = np.full((int(samples / 2), 1), np.nan) # y2
        for j in range(int(samples / 2)):
            seed = j + skip
            rng = np.random.default_rng(seed=seed)
            data1 = rng.random((int(samples/2))).reshape(-1, 1)
            # data1 = qmc.Sobol(d=int(samples/2), seed=seed)
            data2[j, :], next_seed = sobol_seq.i4_sobol(1, seed)
            data3 = rng.random((int(samples/2))).reshape(-1, 1)
        data_first = np.concatenate((data1, data2, data3), axis=1)
        for k in range(int(samples / 2)):
            data4[k, :] = np.absolute(np.random.normal(0.025, 0.01, 1))
            data5[k, :] = np.random.normal(0, 1, 1)
            data6[k, :] = np.random.normal(0, 1, 1)
        # data4[:, :] = (data4[:, :] - np.min(data4[:, :]))
        data5[:, :] = (data5[:, :] - np.min(data5[:, :]))
        data6[:, :] = (data6[:, :] - np.min(data6[:, :]))
        # data4[:, :] = data4[:, :] / np.max(data4[:, :])
        data5[:, :] = data5[:, :] / np.max(data5[:, :])
        data6[:, :] = data6[:, :] / np.max(data6[:, :])
        data_second = np.concatenate((data4, data5, data6), axis=1)
        data = np.concatenate((data_first, data_second), axis=0)
        return torch.from_numpy(data).type(torch.FloatTensor)

    elif type_of_points == "sobol_center_3":
        skip = random_seed
        data1 = np.full((int(3 * samples / 4), 1), np.nan) # t1
        data2 = np.full((int(3 * samples / 4), 1), np.nan) # x1
        data3 = np.full((int(3 * samples / 4), 1), np.nan) # y1
        data4 = np.full((int(samples / 4), 1), np.nan) # t2
        data5 = np.full((int(samples / 4), 1), np.nan) # x2
        data6 = np.full((int(samples / 4), 1), np.nan) # y2
        for j in range(int(3 * samples / 4)):
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim, seed)
            data1[j, :] = rnd[0]
            data2[j, :] = rnd[1]
            data3[j, :] = rnd[2]
        data_first = np.concatenate((data1, data2, data3), axis=1)
        for k in range(int(samples / 4)):
            data4[k, :] = np.absolute(np.random.normal(0.025, 0.01, 1))
            data5[k, :] = np.random.normal(0.5, 0.02 + data4[k, :], 1)
            data6[k, :] = np.random.normal(0.5, 0.02 + data4[k, :], 1)
            # data5[k, :] = np.random.normal(0.5, 0.05, 1)
            # data6[k, :] = np.random.normal(0.5, 0.05, 1)
        '''
        data5[:, :] = (data5[:, :] - np.min(data5[:, :]))
        data6[:, :] = (data6[:, :] - np.min(data6[:, :]))
        data5[:, :] = data5[:, :] / np.max(data5[:, :])
        data6[:, :] = data6[:, :] / np.max(data6[:, :])
        '''
        data_second = np.concatenate((data4, data5, data6), axis=1)
        data = np.concatenate((data_first, data_second), axis=0)
        return torch.from_numpy(data).type(torch.FloatTensor)

    elif type_of_points == "sobol_center_4":
        skip = random_seed
        samples = samples * 10
        radius = np.full((int(samples), 1), np.nan)
        theta = np.full((int(samples), 1), np.nan)
        time = np.full((int(samples), 1), np.nan)
        x = np.full((int(samples), 1), np.nan)
        y = np.full((int(samples), 1), np.nan)
        # phi = np.full((int(samples), 1), np.nan)
        for j in range(samples):
            seed = j + skip
            a = 1
            rnd, next_seed = sobol_seq.i4_sobol(dim, seed)
            radius[j, :] = rnd[0]
            theta[j, :] = 2 * np.pi * rnd[1]
            time[j, :] = rnd[2]
            radius[j, :] = radius[j, :] ** (a * (1 - time[j, :]) + 0.5)
            x[j, :] = radius[j, :] * np.sqrt(2) * np.cos(theta[j, :])
            x[j, :] = x[j, :] + 0.5
            y[j, :] = radius[j, :] * np.sqrt(2) * np.sin(theta[j, :])
            y[j, :] = y[j, :] + 0.5
            # phi[j, :] = 0.5 * np.pi * rnd[3]
        data = np.concatenate((time, x, y), axis=1)
        data = data[(data[:, :] > 0).all(axis=1)].reshape(-1, dim)
        data = data[(data[:, :] < 1).all(axis=1)].reshape(-1, dim)
        print("tmax", np.max(data[:, 0]))
        print("tmin", np.min(data[:, 0]))
        print("xmax", np.max(data[:, 1]))
        print("xmin", np.min(data[:, 1]))
        print("ymax", np.max(data[:, 2]))
        print("ymin", np.min(data[:, 2]))
        print(data, data.shape)
        return torch.from_numpy(data).type(torch.FloatTensor)

    elif type_of_points == "sobol_center_3d":
        skip = random_seed
        samples = samples * 10
        radius = np.full((int(samples), 1), np.nan)
        theta = np.full((int(samples), 1), np.nan)
        phi = np.full((int(samples), 1), np.nan)
        time = np.full((int(samples), 1), np.nan)
        x = np.full((int(samples), 1), np.nan)
        y = np.full((int(samples), 1), np.nan)
        z = np.full((int(samples), 1), np.nan)
        norm_t = 0.002
        for j in range(samples):
            seed = j + skip
            a = 1
            rnd, next_seed = sobol_seq.i4_sobol(dim, seed)
            radius[j, :] = rnd[0]
            theta[j, :] = 2 * np.pi * rnd[1]
            time[j, :] = rnd[2] * norm_t
            phi[j, :] = 0.5 * np.pi * rnd[3]
            radius[j, :] = radius[j, :] ** (a * (1 - time[j, :]) + 0.5)
            x[j, :] = radius[j, :] * np.sqrt(2) * np.cos(phi[j, :]) * np.cos(theta[j, :])
            x[j, :] = x[j, :] + 0.03
            y[j, :] = radius[j, :] * np.sqrt(2) * np.cos(phi[j, :]) * np.sin(theta[j, :])
            y[j, :] = y[j, :] + 0.25
            z[j, :] = -radius[j, :] * np.sqrt(2) * np.sin(phi[j, :]) + 1
        data = np.concatenate((abs(time), abs(x), abs(y), z), axis=1)
        data = data[(data[:, :] > 0).all(axis=1)].reshape(-1, dim)
        data = data[(data[:, :] < 1).all(axis=1)].reshape(-1, dim)
        print("tmax :", np.max(data[:, 0]))
        print("tmin :", np.min(data[:, 0]))
        print("xmax :", np.max(data[:, 1]))
        print("xmin :", np.min(data[:, 1]))
        print("ymax :", np.max(data[:, 2]))
        print("ymin :", np.min(data[:, 2]))
        print("zmax :", np.max(data[:, 3]))
        print("zmin :", np.min(data[:, 3]))
        print(data, data.shape)
        return torch.from_numpy(data).type(torch.FloatTensor)

    elif type_of_points == "sobol_center_3d_moving":
        skip = random_seed
        samples = samples * 3
        radius = np.full((int(samples), 1), np.nan)
        theta = np.full((int(samples), 1), np.nan)
        phi = np.full((int(samples), 1), np.nan)
        time = np.full((int(samples), 1), np.nan)
        x = np.full((int(samples), 1), np.nan)
        y = np.full((int(samples), 1), np.nan)
        z = np.full((int(samples), 1), np.nan)
        norm_t = 0.002
        V = 2/0.002
        for j in range(samples):
            seed = j + skip
            a = 1
            rnd, next_seed = sobol_seq.i4_sobol(dim, seed)
            radius[j, :] = rnd[0]
            theta[j, :] = 2 * np.pi * rnd[1]
            time[j, :] = rnd[2] * norm_t
            phi[j, :] = 0.5 * np.pi * rnd[3]
            radius[j, :] = radius[j, :] ** (a * (1 - time[j, :]) + 0.5)
            x[j, :] = radius[j, :] * np.sqrt(2) * np.cos(phi[j, :]) * np.cos(theta[j, :]) + 0.03
            x[j, :] = x[j, :] + time[j, :] * V * 1/(2.14)
            y[j, :] = radius[j, :] * np.sqrt(2) * np.cos(phi[j, :]) * np.sin(theta[j, :])
            y[j, :] = y[j, :] + 0.25
            z[j, :] = -radius[j, :] * np.sqrt(2) * np.sin(phi[j, :]) + 1
            # z[j, :] = z[j, :] + 0.985
        data = np.concatenate((abs(time), abs(x), abs(y), z), axis=1)
        data = data[(data[:, :] > 0).all(axis=1)].reshape(-1, dim)
        data = data[(data[:, :] < 1).all(axis=1)].reshape(-1, dim)
        print("tmax :", np.max(data[:, 0]))
        print("tmin :", np.min(data[:, 0]))
        print("xmax :", np.max(data[:, 1]))
        print("xmin :", np.min(data[:, 1]))
        print("ymax :", np.max(data[:, 2]))
        print("ymin :", np.min(data[:, 2]))
        print("zmax :", np.max(data[:, 3]))
        print("zmin :", np.min(data[:, 3]))
        print(data, data.shape)
        return torch.from_numpy(data).type(torch.FloatTensor)

    elif type_of_points == "sobol_center_3d_moving_snake":
        skip = random_seed
        samples = samples * 5
        radius = np.full((int(samples), 1), np.nan)
        theta = np.full((int(samples), 1), np.nan)
        phi = np.full((int(samples), 1), np.nan)
        time = np.full((int(samples), 1), np.nan)
        x = np.full((int(samples), 1), np.nan)
        y = np.full((int(samples), 1), np.nan)
        z = np.full((int(samples), 1), np.nan)
        norm_t = 0.006
        V = 2/0.002
        for j in range(samples):
            seed = j + skip
            a = 1
            rnd, next_seed = sobol_seq.i4_sobol(dim, seed)
            radius[j, :] = rnd[0]
            theta[j, :] = 2 * np.pi * rnd[1]
            time[j, :] = rnd[2] * norm_t
            phi[j, :] = 0.5 * np.pi * rnd[3]
            radius[j, :] = radius[j, :] ** (a * (1 - time[j, :]) + 0.5)

            if time[j, :] <= 0.002:
                x[j, :] = radius[j, :] * np.sqrt(2) * np.cos(phi[j, :]) * np.cos(theta[j, :]) + 0.03
                x[j, :] = x[j, :] + time[j, :] * V * 1/(2.14)
                y[j, :] = radius[j, :] * np.sqrt(2) * np.cos(phi[j, :]) * np.sin(theta[j, :])
                y[j, :] = y[j, :] + 0.25
                z[j, :] = -radius[j, :] * np.sqrt(2) * np.sin(phi[j, :]) + 1
            elif 0.002 <= time[j, :] <= 0.004:
                x[j, :] = radius[j, :] * np.sqrt(2) * np.cos(phi[j, :]) * np.cos(theta[j, :]) + 0.97
                x[j, :] = x[j, :] - (time[j, :] - 0.002) * V * 1/(2.14)
                y[j, :] = radius[j, :] * np.sqrt(2) * np.cos(phi[j, :]) * np.sin(theta[j, :])
                y[j, :] = y[j, :] + 0.5
                z[j, :] = -radius[j, :] * np.sqrt(2) * np.sin(phi[j, :]) + 1
            elif 0.004 <= time[j, :] <= 0.006:
                x[j, :] = radius[j, :] * np.sqrt(2) * np.cos(phi[j, :]) * np.cos(theta[j, :]) + 0.03
                x[j, :] = x[j, :] + (time[j, :] - 0.004) * V * 1/(2.14)
                y[j, :] = radius[j, :] * np.sqrt(2) * np.cos(phi[j, :]) * np.sin(theta[j, :])
                y[j, :] = y[j, :] + 0.75
                z[j, :] = -radius[j, :] * np.sqrt(2) * np.sin(phi[j, :]) + 1

        data = np.concatenate((abs(time), abs(x), abs(y), z), axis=1)
        data = data[(data[:, :] > 0).all(axis=1)].reshape(-1, dim)
        data = data[(data[:, :] < 1).all(axis=1)].reshape(-1, dim)
        print("tmax :", np.max(data[:, 0]))
        print("tmin :", np.min(data[:, 0]))
        print("xmax :", np.max(data[:, 1]))
        print("xmin :", np.min(data[:, 1]))
        print("ymax :", np.max(data[:, 2]))
        print("ymin :", np.min(data[:, 2]))
        print("zmax :", np.max(data[:, 3]))
        print("zmin :", np.min(data[:, 3]))
        print(data, data.shape)
        np.savetxt("t_data.csv", data[:, 0])
        np.savetxt("x_data.csv", data[:, 1])
        np.savetxt("y_data.csv", data[:, 2])
        np.savetxt("z_data.csv", data[:, 3])
        return torch.from_numpy(data).type(torch.FloatTensor)

    elif type_of_points == "moving_center":
        print(type_of_points)
        skip = random_seed
        '''pts=1 #how many centered points per step
        step=1 #how many time steps in center'''
        prob = 0.1 # factor on how large the randomness of the center points is: 0= 100% centered, 1=completely random
        n = 10000 # number of total center-bias points

        x0 = 1 # start of the laser
        xspeed = 1.5 # speed of the laser in mm/t_max
        xsize = 2.8 # total length of domain
        x0 = x0/xsize # normed between one and 0
        xspeed = xspeed/xsize # normed to the size of the domain

        # CHANGE IF YOU CHANGE THE BUFFER Time (minimum extrema of t)!
        t0 = 0.1 # buffer time

        tlaser = 1 # time the laser is actually on
        sum = t0+tlaser
        t0 = t0/sum # normed for scaling
        tlaser = tlaser/sum # normed for scaling

        datat = np.full((samples, 1), np.nan)
        dataX = np.full((samples, 1), np.nan)
        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        dataP = np.full((samples, dim-4), np.nan)
        i = 0
        p = 1
        for j in range(samples-n-1): # majority of coll points sampled around laser
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim-3, seed)
            datat[j, :] = rnd[0]
            dataX[j, :] = np.random.triangular(0, x0+datat[j, :]*xspeed, 1)
            dataY[j, :] = np.random.normal(0.5, 0.1, 1)
            dataZ[j, :] = 1-abs(np.random.normal(0, 0.25, 1))
            if (dim > 4):
                dataP[j, :] = rnd[1:]
        for j in range(samples - n-1, samples): # Center-bias: some points directly in laser center
            seed = j + skip
            rnd, next_seed = sobol_seq.i4_sobol(dim, seed)
            datat[j, :] = rnd[0]
            dataX[j, :] = x0 + (datat[j, :] - t0) / tlaser * xspeed + (rnd[1]-0.5) * prob
            dataY[j, :] = 0.5 + (rnd[2]-0.5) * prob
            dataZ[j, :] = 1 - rnd[3] * prob/2
            if(dim > 4):
                dataP[j, :] = rnd[4:]
            '''if p==pts:
                i = i + 1
                p=1
            else:
                p=p+1'''
        if dim > 4:
            data = np.concatenate((datat, dataX, dataY, dataZ, dataP), axis=1)
        else:
            data = np.concatenate((datat, dataX, dataY, dataZ), axis=1)

        return torch.from_numpy(data).type(torch.FloatTensor)

    elif type_of_points == "initial_center":
        print(type_of_points)
        skip = random_seed
        dataX = np.full((samples, 1), np.nan)
        datat = np.zeros_like(dataX)
        dataP = np.full((samples, dim - 4), np.nan)
        x0 = 1
        xspeed = 1.5
        xsize = 2.8
        x0 = x0 / xsize
        xspeed = xspeed / xsize

        dataY = np.full((samples, 1), np.nan)
        dataZ = np.full((samples, 1), np.nan)
        for j in range(samples):
            seed = j + skip
            dataX[j, :] = np.random.triangular(0, x0 + datat[j, :] * xspeed, 1)
            dataY[j, :] = np.random.normal(0.5, 0.1, 1)
            dataZ[j, :] = np.random.triangular(0, 1, 1)
            if (dim > 4):
                dataP[j, :], next_seed = sobol_seq.i4_sobol(dim - 4, seed)
        if dim > 4:
            data = np.concatenate((datat, dataX, dataY, dataZ, dataP), axis=1)
        else:
            data = np.concatenate((datat, dataX, dataY, dataZ), axis=1)
        return torch.from_numpy(data).type(torch.FloatTensor)
