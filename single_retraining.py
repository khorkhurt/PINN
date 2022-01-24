import os
import sys
import json
import random
import subprocess

random.seed(42)

print("Start Retrainings")

sampling_seed = int(sys.argv[1])
n_coll = int(sys.argv[2])
n_u = int(sys.argv[3])
n_int = int(sys.argv[4])
n_object = int(sys.argv[5])
ob = sys.argv[6]
folder_path = sys.argv[7]
point = sys.argv[8]
validation_size = float(sys.argv[9])

'''argume=list()
for a in range(11, 30): argume.append(str(sys.argv[a]))
argum = str(sys.argv[10])
for arg in argume:
    argum = argum + " " + arg
print('handover:  ', argum)'''
json_string = str(sys.argv[10])
if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
    print()
else:
    json_string = json_string.replace(', ', '\", \"')
    json_string = json_string.replace(': ', '\" :\"')
    json_string = json_string.replace('{', '{\"')
    json_string = json_string.replace('}', '\"}')
    json_string = json_string.replace('\'', '')

network_properties = json.loads(json_string)
shuffle = sys.argv[11]
cluster = sys.argv[12]
GPU = str(sys.argv[13])
n_retrain = int(sys.argv[14])

seeds = list()
for i in range(n_retrain):
    seeds.append(random.randint(1, 10000))
print(seeds)
os.mkdir(folder_path)

for retrain in range(len(seeds)):
    folder_path_retraining = folder_path + "/Retrain_" + str(retrain)
    arguments = list()
    arguments.append(str(sampling_seed))
    arguments.append(str(n_coll))
    arguments.append(str(n_u))
    arguments.append(str(n_int))
    arguments.append(str(n_object))
    arguments.append(str(ob))
    arguments.append(str(folder_path_retraining))
    arguments.append(str(point))
    arguments.append(str(validation_size))
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        arguments.append("\'" + str(network_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append("\""+"\'" + str(network_properties).replace("\'", "\"") + "\'" + "\"")
    arguments.append(str(seeds[retrain]))
    arguments.append(shuffle)

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            if GPU != "None":
                string_to_exec = "bsub -W 24:00 -R \"rusage[mem=16384,ngpus_excl_p=1]\" -R \"select[gpu_model0==" + GPU \
                                 + "]\" python3 PINNS2.py  "
                print(string_to_exec)
            else:
                string_to_exec = "bsub -W 24:00 -R \"rusage[mem=8192]\" python3 PINNS2.py  "
        else:
            string_to_exec = "python3 PINNS2.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        os.system(string_to_exec)
    else:
        '''python = os.environ['PYTHON']
        p = subprocess.Popen([python, "PINNS2.py"] + arguments)
        p.wait()'''
        string_to_exec = "python PINNS2.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        print(string_to_exec)
        os.system(string_to_exec)
