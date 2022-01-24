import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, Ec, network, x_u_train, u_train, x_b_train, u_b_train, x_f_train, dataclass):
        lambda_residual = network.lambda_residual
        lambda_reg = network.regularization_param
        order_regularizer = network.kernel_regularizer

        u_pred_var_list = list()
        u_train_var_list = list()

        if x_b_train.shape[0] != 0:
            Ec.apply_bc(network, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)
        if x_u_train.shape[0] != 0:
            Ec.apply_ic(network, x_u_train, u_train, u_pred_var_list, u_train_var_list)

        u_pred_tot_vars = torch.cat(u_pred_var_list, 0).to(Ec.device)
        u_train_tot_vars = torch.cat(u_train_var_list, 0).to(Ec.device)
        # TODO: what condition is compared atm?
        
        assert not torch.isnan(u_pred_tot_vars).any()

        res = Ec.compute_res(network, x_f_train, None).to(Ec.device)

        # TODO: I implemented so many new things here

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
        u_pred_coll = network(x_f_train.float())
        print(x_f_train)
        print(u_pred_coll, u_pred_coll.shape)
        # TODO: Change the name
        loss_new = (torch.mean(abs(Temp_5 - u_pred_coll) ** 2))

        loss_res = (torch.mean(abs(res) ** 2))
        print(loss_res)
        loss_vars = (torch.mean(abs(u_pred_tot_vars - u_train_tot_vars) ** 2))
        print(loss_vars)
        loss_reg = regularization(network, order_regularizer)

        loss_v = torch.log10(0*lambda_residual*loss_vars + 0*loss_res + loss_new + lambda_reg * loss_reg)
        print("Total Loss:", loss_v.detach().cpu().numpy().round(4), "| Function Loss:",
              torch.log10(loss_vars/(torch.mean(abs(u_train_tot_vars) ** 2))).detach().cpu().numpy().round(4),
              "| PDE Loss:", torch.log10(loss_new).detach().cpu().numpy().round(4), "\n")

        return loss_v, loss_vars, loss_res


def fit(Ec, model, training_set_class, verbose=False):
    num_epochs = model.num_epochs
    optimizer = model.optimizer

    train_losses = list([np.NAN, np.NAN, np.NAN])
    freq = 50

    training_coll = training_set_class.data_coll
    training_boundary = training_set_class.data_boundary
    training_initial_internal = training_set_class.data_initial_internal

    model.train()
    for epoch in range(num_epochs):
        if verbose and epoch % freq == 0:
            print("################################ ", epoch, " ################################")

        if len(training_boundary) != 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_), (x_u_train_, u_train_)) in \
                    enumerate(zip(training_coll, training_boundary, training_initial_internal)):
                if verbose and epoch % freq == 0:
                    print("Batch Number:", step)

                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                def closure():
                    optimizer.zero_grad()
                    loss_f, loss_vars, loss_pde = CustomLoss().forward(Ec, model, x_u_train_, u_train_, x_b_train_,
                                                                       u_b_train_, x_coll_train_, training_set_class)
                    loss_f.backward()
                    train_losses[0] = loss_f
                    train_losses[1] = loss_vars
                    train_losses[2] = loss_pde
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

        elif len(training_boundary) == 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_u_train_, u_train_)) in \
                    enumerate(zip(training_coll, training_initial_internal)):

                x_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)
                u_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)

                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                def closure():
                    optimizer.zero_grad()
                    loss_f, loss_vars, loss_pde = CustomLoss().forward(Ec, model, x_u_train_, u_train_, x_b_train_,
                                                                       u_b_train_, x_coll_train_, training_set_class)
                    loss_f.backward()
                    train_losses[0] = loss_f
                    train_losses[1] = loss_vars
                    train_losses[2] = loss_pde
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

        elif len(training_boundary) != 0 and len(training_initial_internal) == 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll,
                                                                                                  training_boundary)):
                print(step)
                x_u_train_ = torch.full((0, 1), 0)
                u_train_ = torch.full((0, 1), 0)

                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                def closure():
                    optimizer.zero_grad()
                    loss_f, loss_vars, loss_pde = CustomLoss().forward(Ec, model, x_u_train_, u_train_, x_b_train_,
                                                                       u_b_train_, x_coll_train_, training_set_class)

                    loss_f.backward()
                    train_losses[0] = loss_f
                    train_losses[1] = loss_vars
                    train_losses[2] = loss_pde
                    return loss_f

                optimizer.step(closure=closure)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

    return train_losses


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


class StandardLoss(torch.nn.Module):
    def __init__(self):
        super(StandardLoss, self).__init__()

    def forward(self, Ec, network, x_u_train, u_train):
        loss_reg = regularization(network, 2)
        lambda_reg = network.regularization_param
        u_pred = network(x_u_train)
        loss = torch.log10(torch.mean((u_train[:, 0] - u_pred[:, 0]) ** 2) + lambda_reg * loss_reg)
        del u_train, u_pred
        print(loss)
        return loss


def StandardFit(Ec, model, training_set_class, verbose=False):
    num_epochs = model.num_epochs
    optimizer = model.optimizer

    train_losses = list([np.nan, np.nan, np.nan])
    freq = 4

    model.train()
    training_initial_internal = training_set_class.data_initial_internal
    for epoch in range(num_epochs):
        if verbose and epoch % freq == 0:
            print("################################ ", epoch, " ################################")

        for step, (x_u_train_, u_train_) in enumerate(training_initial_internal):
            if verbose and epoch % freq == 0:
                print("Batch Number:", step)

            if torch.cuda.is_available():
                x_u_train_ = x_u_train_.cuda()
                u_train_ = u_train_.cuda()

            def closure():
                optimizer.zero_grad()
                loss_f = StandardLoss().forward(Ec, model, x_u_train_, u_train_)
                loss_f.backward()
                train_losses[0] = loss_f
                train_losses[1] = torch.tensor(0.)
                train_losses[2] = torch.tensor(0.)
                return loss_f
            print(step)
            optimizer.step(closure=closure)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del x_u_train_
            del u_train_

    return train_losses
