import os
import sys
import itertools
import subprocess
from ImportFile import *





rs = 32
N_coll = int(sys.argv[1])
N_u = int(sys.argv[2])
N_int = int(sys.argv[3])
n_object = 0
ob = "None"
folder_name = sys.argv[4]
point = "sobol"
validation_size = 0.0
network_properties = {
    "hidden_layers": [6],
    "neurons": [20],
    "residual_parameter": [0.1, 1, 10, 100, 1000],
    "pde_parameter": [0.1, 1, 10, 100, 1000],
    "kernel_regularizer": [2],
    "regularization_parameter": [0, 1e-2, 1e-1, 1],
    "batch_size": [(N_coll + N_u + N_int)],
    "epochs": [1],
    "max_iter": [50000],
    "activation": ["sin"],
    "optimizer": ["LBFGS"]
}
shuffle = "false"
cluster = sys.argv[5]
GPU = "None"  # GPU="GeForceGTX1080"  # GPU = "GeForceGTX1080Ti"  # GPU = "TeslaV100_SXM2_32GB"
n_retrain = 1

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
settings = list(itertools.product(*network_properties.values()))

i = 0
for setup in settings:
    print('setup:  ', setup)

    folder_path = folder_name + "/Setup_" + str(i)
    print("###################################")
    setup_properties = {
        "hidden_layers": setup[0],
        "neurons": setup[1],
        "residual_parameter": setup[2],
        "pde_parameter": setup[3],
        "kernel_regularizer": setup[4],
        "regularization_parameter": setup[5],
        "batch_size": setup[6],
        "epochs": setup[7],
        "max_iter": setup[8],
        "activation": setup[9],
        "optimizer": setup[10]
    }
    setup_properties = json.dumps(setup_properties)
    arguments = list()
    arguments.append(str(rs))
    arguments.append(str(N_coll))
    arguments.append(str(N_u))
    arguments.append(str(N_int))
    arguments.append(str(n_object))
    arguments.append(str(ob))
    arguments.append(str(folder_path))
    arguments.append(str(point))
    arguments.append(str(validation_size))
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        arguments.append("\'" + str(setup_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append("\""+"\'" + str(setup_properties).replace("\'", "\"") + "\'" + "\"")
        # arguments.append(str(setup_properties).replace("\'", "\""))
    arguments.append(str(shuffle))
    arguments.append(str(cluster))
    arguments.append(str(GPU))
    arguments.append(str(n_retrain))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            string_to_exec = "bsub python3 single_retraining.py "
        else:
            string_to_exec = "python3 single_retraining.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        print(string_to_exec)
        os.system(string_to_exec)
    else:
        string_to_exec = "python single_retraining.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        print('string to exec:   ', string_to_exec)
        os.system(string_to_exec)

    i = i + 1
