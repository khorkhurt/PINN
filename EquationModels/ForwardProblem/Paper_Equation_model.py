from ImportFile import *
from EquationBaseClass import EquationBaseClass
from GeneratorPoints import generator_points
from SquareDomain import SquareDomain
from BoundaryConditions import PeriodicBC, DirichletBC, AbsorbingBC, NeumannBC


class EquationClass(EquationBaseClass):

    def __init__(self, ):
        EquationBaseClass.__init__(self)

        self.type_of_points = "placeholder" #is assigned by PINN.py
        self.output_dimension = 1 #Output is temperature
        self.space_dimensions = 3 #x,y,z
        self.time_dimensions = 1
        self.parameters_values = None #look at the other file for parametric "master model" creation
        self.parameter_dimensions = 0
        self.extrema_values = torch.tensor([[-0.1, 1],   #time domain
                                            [-1, 1.8],   #x-domain
                                            [-1, 1],     #y-domain
                                            [-1, 0.03]]) #z-domain
        self.umax = 3000 #normalization temperature as output is best in [0,1]
        self.norm_time=0.0015 #normalization time, as input is best in [0,1]
        self.c=0.1 #constant material conductivity

        self.tmax = self.norm_time/self.extrema_values[0,1].item() #was used for trials with other time domain ranges
        self.list_of_BC = list([[self.ub1, self.ub1],[self.ub1, self.ub1],[self.ub1, self.ub1]]) #boundary conditions
        self.extrema_values = self.extrema_values if self.parameters_values is None else torch.cat(
            [self.extrema_values, self.parameters_values], 0)
        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          self.list_of_BC,
                                          self.extrema_values,
                                          self.type_of_points,
                                          vel_wave=self.a)
        #select the ressource folder depending on system
        if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
            self.val_path= "/cluster/scratch/syardim/PinnsLV_v3/PinnsLV/"
        else:
            self.val_path = "C:/Users/sarve/Desktop/PinnsLV_v3/PinnsLV/"

        ###########

        self.a_type = "constant"



    def add_collocation_points(self, n_coll, random_seed):

        self.square_domain.type_of_points=self.type_of_points
        #Plotting x-t collocation points at middle surface
        '''xcoll,ycoll=self.square_domain.add_collocation_points(n_coll, random_seed)
        xcoll=xcoll.detach().numpy()
        mask = xcoll[:,2]>-0.05
        mask1= xcoll[:,2]<0.05
        mask2 = xcoll[:,3]>0
        mask3=mask& mask1 & mask2
        plt.scatter(xcoll[mask3,0], xcoll[mask3,1], s=1)
        plt.show()'''
        #plotting x-y collocation points at surface over time steps for gif
        '''
        xcoll,ycoll=self.square_domain.add_collocation_points(n_coll, random_seed)
        xcoll=xcoll.detach().numpy()'''
        #xcoll=xcoll[-10000:,:] #activate for seeing the center bias
        ''' path="D:\polybox\Masterthesis\PINN\Taniya\Paper_Model_y_norm_6_24\\Images/"
        num_img=20
        for i in range(0,num_img):
            time=(i+1)/num_img
            mask1 = xcoll[:, 0] < time
            mask2= xcoll[:, 0] > (i)/num_img
            mask=mask1&mask2
            fig, (ax) = plt.subplots(ncols=1)
            fig.subplots_adjust(wspace=0.01)
            plt.scatter(xcoll[mask,1], xcoll[mask,2], s=1)

            plt.scatter(0+time*1.5, 0, s=10)
            plt.xlim(-1,1.8)
            plt.ylim(-1,1)
            ax.set(xticklabels=[])
            ax.set(yticklabels=[])
            ax.set_title('Sampled Collocation Points')
            plt.savefig(path + "/collocation_all_" + str(i) + ".png", dpi=500)
            print("plotting collocation",i)
            plt.close()'''

        return self.square_domain.add_collocation_points(n_coll, random_seed)

    def add_boundary_points(self, n_boundary, random_seed):
        return self.square_domain.add_boundary_points(n_boundary, random_seed)

    def add_initial_points(self, n_initial, random_seed):
        extrema_0 = self.extrema_values[:, 0] #minimum values
        extrema_f = self.extrema_values[:, 1] #maximum values
        x_time_0 = generator_points(n_initial, self.time_dimensions + self.space_dimensions + self.parameter_dimensions, random_seed,  "initial_center", True)
        x_time_0[:, 0] = torch.full(size=(n_initial,), fill_value=0.0)
        x_time_0 = x_time_0 * (extrema_f - extrema_0) + extrema_0

        y_time_0 = self.u0(x_time_0)

        return x_time_0, y_time_0
    def a(self,x):
        return 1

    def add_internal_points(self, n_internal, random_seed):
        set=1 #1=Rosenthal, 2=FEM
        if set==1:
            print("Rosenthal Support")
            extrema_0 = self.extrema_values[:, 0] #minimum values
            extrema_f = self.extrema_values[:, 1] #maximum values
            x_intern = generator_points(n_internal, self.time_dimensions + self.space_dimensions + self.parameter_dimensions, random_seed, self.type_of_points, True)
            x_intern[:, 0] = torch.full(size=(n_internal,), fill_value=0.0)
            x_intern = x_intern * (extrema_f - extrema_0) + extrema_0
            y_intern = self.rosenthal(x_intern)
        else:
            print("FEM Support")
            #load data
            node_coord_df = pd.read_csv(self.val_path + 'nodes.csv', index_col='label')
            temp_df = pd.read_csv(self.val_path + 'temp.csv', index_col='label')
            times = [0,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.0011,0.0012,0.0013,0.0014,0.0015]
            test_inp = node_coord_df.to_numpy()
            Exact = temp_df.to_numpy()
            # create dataset with different times, mostly shaping into correct form
            for t in times:
                t_i=np.ones(test_inp.shape[0])*t
                wanted = (t_i * 10000)
                test_time = t_i / self.tmax
                temp_inp_i = np.column_stack([test_time, test_inp])
                temp_out_i=Exact[np.arange(len(Exact)), wanted.astype(int)]
                temp_out_i=temp_out_i[:, None] / self.umax
                if t == 0:
                    test_input = temp_inp_i
                    output = temp_out_i
                else:
                    test_input=np.row_stack((test_input,temp_inp_i))
                    output =np.row_stack((output, temp_out_i))

            complete=np.column_stack((test_input,output))
            np.random.seed(random_seed)
            np.random.shuffle(complete)
            assert (n_internal<np.shape(complete)[0]), "Too many support points chosen, dataset too small!"

            selection=complete[:n_internal,:]
            test_inp=selection[:,0:4]
            output=selection[:,4]
            test_inp = test_inp.astype(np.float32)
            test_inp_tens = torch.from_numpy(test_inp)
            x_intern = test_inp_tens

            y_intern = torch.from_numpy(output.astype(np.float32))
            y_intern = torch.reshape(y_intern, (-1,))
        return x_intern, y_intern

    def apply_bc(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):

        self.square_domain.apply_boundary_conditions(model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)

    def apply_ic(self, model, x_u_train, u_train, u_pred_var_list, u_train_var_list):
        for j in range(self.output_dimension):
            if x_u_train.shape[0] != 0:
                u = model(x_u_train)[:, j]
                u_pred_var_list.append(u)
                u_train_var_list.append(u_train[:])
                print(torch.mean((u-u_train[:])**2)/torch.mean((u_train[:])**2))


    def compute_res(self, network, x_f_train, solid_object):
        self.network=network
        x_f_train.requires_grad = True
        u = network(x_f_train)[:, 0].reshape(-1, )
        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u).to(self.device), create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_y = grad_u[:, 2]
        grad_u_z = grad_u[:, 3]

        FV_0=self.FV_0(x_f_train,1,1,1,1)
        c=self.conductivity(x_f_train,u,FV_0)
        rho = 8220e-9
        # cp = 605
        cp = self.heat_capacity(u)
        #c=self.c
        grad_u_xx = torch.autograd.grad(c*grad_u_x, x_f_train, grad_outputs=torch.ones_like(u).to(self.device), create_graph=True)[0][:, 1]
        grad_u_yy = torch.autograd.grad(c*grad_u_y, x_f_train, grad_outputs=torch.ones_like(u).to(self.device), create_graph=True)[0][:, 2]
        grad_u_zz = torch.autograd.grad(c*grad_u_z, x_f_train, grad_outputs=torch.ones_like(u).to(self.device), create_graph=True)[0][:, 3]
        time = self.tmax
        q = self.source(x_f_train)*time/(rho*cp*self.umax)
        res2=(grad_u_xx.reshape(-1, ) + grad_u_yy.reshape(-1, )+ grad_u_zz.reshape(-1, )) * time / (rho*cp)
        residual = (grad_u_t.reshape(-1, )- res2 - q)

        #enforce zero-gradient in the region before laser gradient
        mask_init = torch.le(x_f_train[:,0], 0)
        residual[mask_init]=grad_u_t.reshape(-1, )[mask_init]
        #enforce temperatures above 0
        mask_temp=torch.le(u, 0)
        residual[mask_temp]=u[mask_temp]*self.umax
        #debugging printout
        print("Max du/dt",torch.max(abs(grad_u_t.reshape(-1, ))).detach().cpu().numpy().round(4),"; mean: ",torch.mean(grad_u_t.reshape(-1, )).detach().cpu().numpy().round(4))
        print("Max d/dx(c*du/dx)",torch.max(abs(res2)).detach().cpu().numpy().round(4), "; mean: ",torch.mean(res2).detach().cpu().numpy().round(4))
        print("Max source",(torch.max(abs(q))).detach().cpu().numpy().round(4), "; mean: ",(torch.mean(q)).detach().cpu().numpy().round(4))
        print("max predicted temp: ", (torch.max((u*self.umax))).detach().cpu().numpy().round(4),"min temp: ", (torch.min((u*self.umax))).detach().cpu().numpy().round(4))
        print("Max residual: ", torch.max(abs(residual)).detach().cpu().numpy().round(4),"; Mean: ", torch.mean(residual).detach().cpu().numpy().round(4))
        return residual

    def v0(self, x):
        return torch.zeros((x.shape[0], 1))

    def ub0(self, t):
        type_BC = [DirichletBC()]
        u0= 25 / self.umax
        u = torch.full(size=(t.shape[0], 1), fill_value=u0)
        return u, type_BC

    def ub1(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub2(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0],1), fill_value=0.0)
        return u, type_BC
    def u0(self, x):
        self.ini=25/self.umax
        u0= torch.ones(x.shape[0])*self.ini
        return u0
    def rosenthal(self, x):
        #c=conductivity, a=diffusivity
        sig = 0.05
        q = 150
        n = 0.5
        T0 = 25
        # make all x physical
        c = self.c*1000
        rho = 8220
        cp = 605
        a=c/(rho*cp)
        x_phys = x[:, 1] /1000 # *self.xmax-1
        y_phys = x[:, 2]/1000  # *self.ymax-1
        z_phys = x[:, 3]/1000
        norm = self.norm_time / self.tmax
        v = 1.5/self.tmax/1000
        timestep = 0/1000 + x[:, 0] / norm * 1.5/1000
        r = ((x_phys-timestep)**2+y_phys**2+(z_phys-0.03)**2)**0.5
        T = T0 + (q*n)/(2*pi*c*r) * torch.exp(-(v*(x_phys-timestep+r))/(2*a))
        # heat=heat/torch.max(heat)
        return T/self.umax
    def source(self, x):
        sig=0.05
        q=150
        n=0.5
        x_phys = x[:, 1]
        y_phys = x[:, 2]
        z_phys = x[:, 3]
        norm=self.norm_time/self.tmax
        v=1.5/norm
        timestep=0+x[:,0]*v
        mask=1-1./(1+torch.exp((x[:,0]-0.03)*400))
        heat=6*np.sqrt(3)*n*q/(pi*np.sqrt(pi)*sig**3)* torch.exp(-3* ( (x_phys-timestep)**2 + (y_phys)**2 + (z_phys-0.03)**2 ) /sig**2) *mask

        #heat=heat/torch.max(heat)
        return heat

    def FV_0(self,input, x, y, xsize, ysize):
        FV=torch.ones_like(input[:,1])
        X=[x,y,xsize,ysize]
        mask = torch.gt(input[:,1], X[0] - X[2] / 2) & ~torch.ge(input[:,1],X[0] + X[2] / 2) & torch.gt(input[:, 2],X[1] - X[3] / 2) & ~torch.ge(input[:, 2], X[1] + X[3] / 2)
        FV[mask]=0

        FV=torch.ones_like(input[:,1])
        '''plt.scatter(input[:,1].cpu().detach(),input[:,2].cpu().detach(),c=FV.cpu().detach())
        plt.show()'''
        return FV
    def FV(self, input, T, FV_0):
        t=input[:,0]
        x=input[:,1]
        y=input[:,2]
        T_melt=1260
        T_sint=25

        #sintering is not implemented
        FV_sint=(T-T_sint)/(T_melt-T_sint)


        mask=(T<T_melt-460)
        #FV_0[mask]=1/(1+torch.exp(-0.035*(T[mask]-T_melt+260)))
        FV_new=FV_0
        return FV_new

    def conductivity(self, x, u, FV_0):
        T=u*self.umax
        k = torch.empty(T.shape)
        k = torch.tanh((T-1475)/200)*200 + 210 + 0.02815*T
        return k

    def heat_capacity(self, u):
        T = u * self.umax
        cp = torch.empty(u.shape)
        cp = 428.5 + T * 0.226 + \
             (276000 / (100 * ((2 * 3.141592654) ** 0.5))) * torch.exp(-0.5 * ((T - 1360) ** 2) / (100 ** 2))
        return cp

    def compute_generalization_error(self, model, extrema, val_path=None, images_path=None):
        node_coord_df = pd.read_csv(self.val_path + 'nodes.csv', index_col='label')
        temp_df = pd.read_csv(self.val_path + 'temp.csv', index_col='label')
        times = [0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.0011,0.0012,0.0013,0.0014]
        test_inp = node_coord_df.to_numpy()
        Exact = temp_df.to_numpy()
        i=0
        # create dataset with different times, mostly shaping into correct form
        for t in times:
            t_i = np.ones(test_inp.shape[0]) * t
            wanted = np.round(t_i * 10000).astype(int)
            test_time = t_i / self.tmax
            temp_inp_i = np.column_stack([test_time, test_inp])
            temp_out_i = Exact[np.arange(len(Exact)), wanted]
            temp_out_i = temp_out_i[:, None] / self.umax
            if i == 0:
                test_input = temp_inp_i
                output = temp_out_i
            else:
                test_input = np.row_stack((test_input, temp_inp_i))
                output = np.row_stack((output, temp_out_i))
            i=i+1
        test_inp = test_input.astype(np.float32)
        test_inp_tens = torch.from_numpy(test_inp)
        Exact=output*self.umax
        test_out = model(test_inp_tens.to('cuda'))[:, 0].cpu().detach().numpy().reshape(-1, 1)*self.umax
        assert (Exact.shape[1] == test_out.shape[1])
        L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
        print("Error Test:", L2_test)
        rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
        print("Relative Error Test:", rel_L2_test)
        '''if images_path is not None:
            fig, (ax) = plt.subplots(ncols=1)
            fig.subplots_adjust(wspace=0.01)
            ax.grid(True, which="both", ls=":")
            plt.xlabel(r'Exact Values')
            plt.ylabel(r'Predicted Values')
            scatter = ax.scatter(Exact, test_out, c= test_input[:,0], s=1,cmap='viridis')
            legend1 = ax.legend(*scatter.legend_elements(), bbox_to_anchor=(1.05, 1), loc='upper left', title="t=")
            ax.add_artist(legend1)
            plt.savefig(images_path + "/Score.png", dpi=500)'''

        return L2_test, rel_L2_test

