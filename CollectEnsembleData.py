from CollectUtils import *

np.random.seed(42)

base_path_list = ["HeatH1_50b", "HeatH1_100b", "HeatH1_200b",
                  "WaveH1_30", "WaveH1_60", "WaveH1_90", "WaveH1_120",
                  "WaveH1_30b", "WaveH1_60b", "WaveH1_90b", "WaveH1_120b",
                  "StokesH1_20", "StokesH1_40", "StokesH1_80", "StokesH1_160"]

# base_path_list = ["HeatH1_50b", "HeatH1_100b", "HeatH1_200b"]
base_path_list = ["Setups"]

for base_path in base_path_list:
    print("#################################################")
    print(base_path)

    b = False
    compute_std = False
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sensitivity_df = pd.DataFrame(columns=["batch_size",
                                           "regularization_parameter",
                                           "kernel_regularizer",
                                           "neurons",
                                           "hidden_layers",
                                           "residual_parameter",
                                           "mean_L2_test",
                                           "mean_rel_L2_test",
                                           "error_train",
                                           "error_val",
                                           "error_test"])
    # print(sensitivity_df)

    selection_criterion = "selection"
    eval_criterion = "selection"

    Nu_list = []
    Nf_list = []
    t_0 = 0
    t_f = 1
    x_0 = -1
    x_f = 1

    L2_norm = []
    criterion = []
    best_retrain_list = []
    list_models_setup = list()

    for subdirec in directories_model:
        model_path = base_path

        sample_path = model_path + "/" + subdirec
        retrainings_fold = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]

        retr_to_check_file = None
        for ret in retrainings_fold:
            if os.path.isfile(sample_path + "/" + ret + "/TrainedModel/Information.csv"):
                retr_to_check_file = ret
                break

        setup_num = int(subdirec.split("_")[1])
        if retr_to_check_file is not None:
            info_model = pd.read_csv(sample_path + "/" + retr_to_check_file + "/TrainedModel/Information.csv", header=0,
                                     sep=",")
            best_retrain = select_over_retrainings(sample_path, selection=selection_criterion, mode="mean",
                                                   compute_std=compute_std, compute_val=False, rs_val=0)
            info_model["error_train"] = best_retrain["error_train"]
            info_model["train_time"] = best_retrain["train_time"]
            info_model["selection"] = best_retrain["selection"]
            info_model["error_val"] = 0
            info_model["error_test"] = 0
            info_model["mean_L2_test"] = best_retrain["mean_L2_test"]
            info_model["mean_rel_L2_test"] = best_retrain["mean_rel_L2_test"]
            # info_model["a_rel"] = best_retrain["a_rel"]
            if os.path.isfile(sample_path + "/" + retr_to_check_file + "/Images/errors.txt"):
                # info_model["u_rel_err"] = best_retrain["u_rel_err"]
                # info_model["k_rel_err"] = best_retrain["k_rel_err"]
                info_model["err_1"] = best_retrain["err_1"]
                info_model["err_2"] = best_retrain["err_2"]
            if os.path.isfile(sample_path + "/" + retr_to_check_file + "/Images/errors_inv.txt"):
                info_model["l2_glob"] = best_retrain["l2_glob"]
                info_model["l2_glob_rel"] = best_retrain["l2_glob_rel"]
                info_model["l2_om_big"] = best_retrain["l2_om_big"]
                info_model["l2_om_big_rel"] = best_retrain["l2_om_big_rel"]
                info_model["h1_glob"] = best_retrain["h1_glob"]
                info_model["h1_glob_rel"] = best_retrain["h1_glob_rel"]
            info_model["setup"] = setup_num
            info_model["retraining"] = best_retrain["retraining"]

            if info_model["batch_size"].values[0] == "full":
                info_model["batch_size"] = best_retrain["Nu_train"] + best_retrain["Nf_train"]

            if info_model["regularization_parameter"].values[0] != 1e-2 and \
                    info_model["regularization_parameter"].values[0] != 1e-1:
                sensitivity_df = sensitivity_df.append(info_model, ignore_index=True)
        else:
            print(sample_path + "/TrainedModel/Information.csv not found")

    # min_idx = int(np.argmin(criterion))
    # min_idx_test = int(np.argmin(L2_norm))
    # min_test_according_val = L2_norm[min_idx]
    # min_test_according_test = L2_norm[min_idx_test]
    # best_setup = best_retrain_list[min_idx]
    # best = best_retrain_list[min_idx_test]
    # print(min_test_according_val/mean_val*100, best_setup)
    # print(min_test_according_test/mean_val*100, best)

    sensitivity_df = sensitivity_df.sort_values(selection_criterion)
    best_setup = sensitivity_df.iloc[0]
    best_setup.to_csv(base_path + "best.csv", header=0, index=False)
    # print(sensitivity_df)
    print("Best Setup:", best_setup["setup"])
    print(best_setup)

    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.scatter(sensitivity_df[selection_criterion], sensitivity_df[eval_criterion])
    plt.xlabel(r'$\varepsilon_T$')
    plt.ylabel(r'$\varepsilon_G$')
    # plt.show()
    # quit()
    plt.savefig(base_path + "/et_vs_eg.png", dpi=400)

    total_list = list()

    var_list = ["hidden_layers",
                "neurons",
                "residual_parameter",
                "kernel_regularizer",
                "regularization_parameter",
                "activation"]

    labels_list = ["hidden-layers",
                   "neurons",
                   "residual-parameter",
                   "kernel-regularizer",
                   "regularization-parameter",
                   "activation"]
    for var in var_list:
        print("=======================================================")
        print(var)
        params = sensitivity_df[var].values
        params = list(set(params))
        params.sort()
        df_param_list = list()
        for value in params:
            index_list_i = sensitivity_df.index[sensitivity_df[var] == value]
            new_df = sensitivity_df.loc[index_list_i]
            df_param_list.append(new_df)
            # print(new_df)
        total_list.append(df_param_list)

    if not b:
        out_var_vec = list()
        out_var_vec.append(eval_criterion)

        for out_var in out_var_vec:
            for j in range(len(total_list)):
                print("-------------------------------------------------------")
                var = var_list[j]
                lab = labels_list[j]
                print(var)
                print(lab)
                # name = name_list[j]
                sens_list = total_list[j]
                Nf_dep_fig = plt.figure()
                axes = plt.gca()
                max_val = 0
                plt.grid(True, which="both", ls=":")
                for i in range(len(sens_list)):
                    df = sens_list[i]
                    print(df)

                    value = df[var].values[0]
                    label = lab + " = " + str(value).replace("_", "-")

                    sns.distplot(df[out_var], label=label, kde=True, hist=True, norm_hist=False,
                                 kde_kws={'shade': True, 'linewidth': 2})
                plt.xlabel(r'$\varepsilon_G$')

                plt.legend(loc=1)
                plt.savefig(base_path + "/Sensitivity_" + var + ".png", dpi=500)
