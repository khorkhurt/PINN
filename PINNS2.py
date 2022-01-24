from ImportFile import *

torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def dump_to_file():
    torch.save(model, model_path + "/model3.pkl")
    torch.save(model.state_dict(), model_path + "/model4.pkl")
    with open(model_path + os.sep + "Information.csv", "w") as w:
        keys = list(network_properties.keys())
        vals = list(network_properties.values())
        w.write(keys[0])
        for i in range(1, len(keys)):
            w.write("," + keys[i])
        w.write("\n")
        w.write(str(vals[0]))
        for i in range(1, len(vals)):
            w.write("," + str(vals[i]))

    with open(folder_path + '/InfoModel.txt', 'w') as file:
        file.write("Nu_train,"
                   "Nf_train,"
                   "Nint_train,"
                   "validation_size,"
                   "train_time,"
                   "error_train,"
                   "error_vars,"
                   "error_pde\n")
        file.write(str(N_u_train) + "," +
                   str(N_coll_train) + "," +
                   str(N_int_train) + "," +
                   str(validation_size) + "," +
                   str(end) + "," +
                   str(final_error_train) + "," +
                   str(error_vars) + "," +
                   str(error_pde))


def initialize_inputs(len_sys_argv):
    if len_sys_argv == 1:

        # Random Seed for sampling the dataset
        sampling_seed_ = 128

        # Number of training+validation points
        n_coll_ = 2 ** 19  # 2 ** 15
        n_u_ = 2 ** 18 # 2 ** 14
        # n_int_ = 5151 * 156  # internal points = data support
        n_int_ = 0  # internal points = data support

        # Additional Info
        folder_path_ = "Paper_Model"
        point_ = "moving_center"  # useless
        validation_size_ = 0.0  # useless
        network_properties_ = {
            "hidden_layers": 6,
            "neurons": 24,
            "residual_parameter": 1,  # weight of the initial and boundary function in Loss function
            "pde_parameter": 1,  # weight of the initial and boundary function in Loss function
            "kernel_regularizer": 2,  # what kind of regularization L#
            "regularization_parameter": 0,  # how strong is regularization
            "batch_size": (n_coll_ + n_u_ + n_int_),
            "epochs": 1,
            "max_iter": 50000,
            "activation": "sin",  # tanh,#sin... needs to be 2 times differentiable
            "optimizer": "LBFGS"  # ADAM
        }
        retrain_ = 32

        shuffle_ = False

    elif len_sys_argv == 13:
        print(sys.argv)
        # Random Seed for sampling the dataset
        sampling_seed_ = int(sys.argv[1])

        # Number of training+validation points
        n_coll_ = int(sys.argv[2])
        n_u_ = int(sys.argv[3])
        n_int_ = int(sys.argv[4])

        # Additional Info
        folder_path_ = sys.argv[7]
        point_ = sys.argv[8]
        print(sys.argv[9])
        validation_size_ = float(sys.argv[9])

        json_string = str(sys.argv[10])
        if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
            print()
        else:
            json_string = json_string.replace(', ', '\", \"')
            json_string = json_string.replace(': ', '\" :\"')
            json_string = json_string.replace('{', '{\"')
            json_string = json_string.replace('}', '\"}')
            json_string = json_string.replace('\'', '')
        network_properties_ = json.loads(json_string)
        network_properties_["hidden_layers"] = int(network_properties_["hidden_layers"])
        network_properties_["neurons"] = int(network_properties_["neurons"])
        network_properties_["residual_parameter"] = float(network_properties_["residual_parameter"])
        network_properties_["pde_parameter"] = float(network_properties_["pde_parameter"])
        network_properties_["kernel_regularizer"] = int(network_properties_["kernel_regularizer"])
        network_properties_["regularization_parameter"] = float(network_properties_["regularization_parameter"])
        network_properties_["batch_size"] = int(network_properties_["batch_size"])
        network_properties_["epochs"] = int(network_properties_["epochs"])
        network_properties_["max_iter"] = int(network_properties_["max_iter"])

        retrain_ = sys.argv[11]
        if sys.argv[12] == "false":
            shuffle_ = False
        else:
            shuffle_ = True
    else:
        raise ValueError("One input is missing, I only have ", len_sys_argv)
    print(network_properties_)
    return sampling_seed_, n_coll_, n_u_, n_int_, folder_path_, point_, validation_size_, network_properties_, \
           retrain_, shuffle_


sampling_seed, N_coll, N_u, N_int, folder_path, point, validation_size, network_properties, retrain, \
    shuffle = initialize_inputs(len(sys.argv))

Ec = EquationClass()
Ec.type_of_points = point
if Ec.extrema_values is not None:
    extrema = Ec.extrema_values
    space_dimensions = Ec.space_dimensions
    time_dimension = Ec.time_dimensions
    parameter_dimensions = Ec.parameter_dimensions

    print(space_dimensions, time_dimension, parameter_dimensions)
else:
    print("Using free shape. Make sure you have the functions:")
    print("     - add_boundary(n_samples)")
    print("     - add_collocation(n_samples)")
    print("in the Equation file")

    extrema = None
    space_dimensions = Ec.space_dimensions
    time_dimension = Ec.Etime_dimensions
try:
    parameters_values = Ec.parameters_values
    parameter_dimensions = parameters_values.shape[0]
except AttributeError:
    print("No additional parameter found")
    parameters_values = None
    parameter_dimensions = 0

input_dimensions = parameter_dimensions + time_dimension + space_dimensions
output_dimension = Ec.output_dimension
mode = "none"
if network_properties["epochs"] != 1:
    max_iter = 1
else:
    max_iter = network_properties["max_iter"]
print(max_iter)
N_u_train = int(N_u * (1 - validation_size))
N_coll_train = int(N_coll * (1 - validation_size))
N_int_train = int(N_int * (1 - validation_size))
N_train = N_u_train + N_coll_train + N_int_train

if space_dimensions > 0:
    N_b_train = int(N_u_train / (4 * space_dimensions))
    # N_b_train = int(N_u_train / (1 + 2 * space_dimensions))
else:
    N_b_train = 0
if time_dimension == 1:
    N_i_train = N_u_train - 2 * space_dimensions * N_b_train
    # N_i_train = N_u_train - N_b_train*(2 * space_dimensions)
elif time_dimension == 0:
    N_b_train = int(N_u_train / (2 * space_dimensions))
    N_i_train = 0
else:
    raise ValueError()

print("\n######################################")
print("*******Domain Properties********")
print(extrema)

print("\n######################################")
print("*******Info Training Points********")
print("Number of train collocation points: ", N_coll_train)
print("Number of initial and boundary points: ", N_u_train, N_i_train, N_b_train)
print("Number of internal points: ", N_int_train)
print("Total number of training points: ", N_train)

print("\n######################################")
print("*******Network Properties********")
pprint.pprint(network_properties)
batch_dim = network_properties["batch_size"]

print("\n######################################")
print("*******Dimensions********")
print("Space Dimensions", space_dimensions)
print("Time Dimension", time_dimension)
print("Parameter Dimensions", parameter_dimensions)
print("\n######################################")

if network_properties["optimizer"] == "LBFGS" and network_properties["epochs"] != 1 and network_properties["max_iter"] \
        == 1 and (batch_dim == "full" or batch_dim == N_train):
    print(bcolors.WARNING + "WARNING: you set max_iter=1 and epochs=" + str(network_properties["epochs"]) +
          " with a LBFGS optimizer.\n"
          "This will work but it is not efficient in full batch mode. Set max_iter = "
          + str(network_properties["epochs"]) + " and epochs=1. instead" + bcolors.ENDC)

if batch_dim == "full":
    batch_dim = N_train

# #############################################################################################################################################################
# Dataset Creation
training_set_class = DefineDataset(Ec, N_coll_train, N_b_train, N_i_train, N_int_train, batches=batch_dim,
                                   random_seed=sampling_seed, shuffle=shuffle)
training_set_class.assemble_dataset()

# #############################################################################################################################################################
# Model Creation
additional_models = None
model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension,
              network_properties=network_properties)

# #############################################################################################################################################################
# Weights Initialization
torch.manual_seed(retrain)
init_xavier(model)

model_path = folder_path + "/TrainedModel/"
weights = torch.load(model_path + "/model3.pkl", map_location=torch.device('cpu'))
model.load_state_dict(weights.state_dict(), strict=False)
print(model)
quit()

# print(model.coeff_list)
# #############################################################################################################################################################
# Model Training
start = time.time()
print("Fitting Model")
model.to(Ec.device)
model.train()
optimizer_LBFGS = optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter, max_eval=50000, history_size=100,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
optimizer_ADAM = optim.Adam(model.parameters(), lr=0.0005)

if network_properties["optimizer"] == "LBFGS":
    model.optimizer = optimizer_LBFGS
elif network_properties["optimizer"] == "ADAM":
    model.optimizer = optimizer_ADAM
else:
    raise ValueError()

if N_coll_train != 0:
    errors = fit(Ec, model, training_set_class, verbose=True)
else:
    errors = StandardFit(Ec, model, training_set_class, verbose=True)
end = time.time() - start
print("\nTraining Time: ", end)

model = model.eval()
final_error_train = float(((10 ** errors[0]) ** 0.5).detach().cpu().numpy())
error_vars = float((errors[1]).detach().cpu().numpy())
error_pde = float((errors[2]).detach().cpu().numpy())
print("\n################################################")
print("Final Training Loss:", final_error_train)
print("################################################")

# #############################################################################################################################################################
# Plotting ang Assessing Performance
images_path = folder_path + "/Images"
model_path = folder_path + "/TrainedModel"
val_path = "C:/Users/sarve/Desktop/PinnsLV_v3/PinnsLV/"

if (not os.path.exists(folder_path)):
    os.mkdir(folder_path)
    os.mkdir(images_path)
    os.mkdir(model_path)

# Ec.plotting(model, val_path, images_path, extrema)

# Mean_L2_test, max_L2_test, Mean_rel_L2_test, max_rel_L2_test = Ec.compute_generalization_error(model, val_path,
#                                                                                              images_path)

dump_to_file()
