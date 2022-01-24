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
        self.list_of_BC = list([[self.ub0, self.ub1], [self.ub0, self.ub2]])
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
        self.sigma = 1 / 8 #cp
        self.k = 1         #thermal conductivity
        self.rho=1         #density
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
    def a(self,x):
        return 1

    def apply_bc(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):

        self.square_domain.apply_boundary_conditions(model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)

    def apply_ic(self, model, x_u_train, u_train, u_pred_var_list, u_train_var_list):
        for j in range(self.output_dimension):
            if x_u_train.shape[0] != 0:
                x_u_train.requires_grad = True
                u = model(x_u_train)[:, j]
           #     v = self.v0(x_u_train)[:, j]
                grad_u = torch.autograd.grad(u, x_u_train, grad_outputs=torch.ones_like(u).to(self.device), create_graph=True)[0]
                grad_u_t = grad_u[:, 0]
                u_pred_var_list.append(u)
                u_train_var_list.append(u_train[:])
         #       u_pred_var_list.append(grad_u_t)
          #      u_train_var_list.append(v)

    def compute_res(self, network, x_f_train, solid_object):
        x_f_train.requires_grad = True
        u = network(x_f_train)[:, 0].reshape(-1, )
        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                     create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_y = grad_u[:, 2]

        #grad_u_tt = torch.autograd.grad(grad_u_t, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
        # create_graph=True)[0][:, 0]
        grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                        create_graph=True)[0][:, 1]
        grad_u_yy = torch.autograd.grad(grad_u_y, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                        create_graph=True)[0][:, 2]

        residual = grad_u_t.reshape(-1, ) - 1/9 *( grad_u_xx.reshape(-1, )+ grad_u_yy.reshape(-1, ))
        return residual

    ''' def exact(self, x):
        #lambda=pi/6*sqrt(m**2+n**2)
        u = 0
        for m in range(1,50):
            for n in range(1,50):
                u=u+ 4/pi**2 *(( 1+(-1)**(m+1)) *(1-np.cos(n*pi/2))/(m*n)* torch.sin(m*pi/2*(x[:,1]+1))*torch.sin(n*pi/2*(x[:,2]+1))*torch.exp(-pi**2*(m**2+n**2)*x[:,0]/36)  )
        return u.reshape(-1, 1) '''

    def v0(self, x):
        return torch.zeros((x.shape[0], 1))

    def ub0(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub1(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub2(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def u0(self, x):
        #50 for y<0, 0 else
        u0 = (1/(1+torch.exp(2000*x[:,2])))
        print(u0)
        print(u0.shape)
        quit()
        return u0

    def compute_generalization_error(self, model, extrema,val_path=None, images_path=None):
        model=model.cpu()
        model.eval()
        test_inp = self.convert(torch.rand([100000, extrema.shape[0]]), extrema)
        Exact = (self.exact(test_inp)[:, 0]).numpy().reshape(-1, 1)
        test_out = model(test_inp)[:, 0].detach().numpy().reshape(-1, 1)
        assert (Exact.shape[1] == test_out.shape[1])
        L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
        print("Error Test:", L2_test)

        rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
        print("Relative Error Test:", rel_L2_test)

        if images_path is not None:
            plt.figure()
            plt.grid(True, which="both", ls=":")
            plt.scatter(Exact, test_out)
            plt.xlabel(r'Exact Values')
            plt.ylabel(r'Predicted Values')
            plt.savefig(images_path + "/Score.png", dpi=400)
        return L2_test, rel_L2_test


    def plotting(self, model, images_path, extrema, solid):
        model.cpu()
        model = model.eval()
        x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], 100), [100, 1])
        y = torch.reshape(torch.linspace(extrema[2, 0], extrema[2, 1], 100), [100, 1])
        time_steps = [0.0, 0.25, 0.5, 0.75, 1]
        scale_vec = np.linspace(0.65, 1.55, len(time_steps))
        t1=0.1
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        X=x.numpy()
        Y=y.numpy()
        X, Y = np.meshgrid(X,Y)
        x_t=torch.reshape(torch.from_numpy(X),[10000,1])
        y_t = torch.reshape(torch.from_numpy(Y),[10000,1])
        plot_var1 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=t1), x_t, y_t], 1)
        Z = self.exact(plot_var1)[:, 0]
        Z = torch.reshape(Z, [100, 100])
        Z = Z.numpy()
        plot_var2 = torch.cat([torch.tensor(()).new_full(size=(10000, 1), fill_value=0.6), x_t, y_t], 1)
        Z2 = self.exact(plot_var2)[:, 0]
        Z2 = torch.reshape(Z2, [100, 100])
        Z2 = Z2.numpy()
        fig, (ax, ax2) = plt.subplots(ncols=2)
        fig.subplots_adjust(wspace=0.01)
        sns.heatmap(Z, cmap="rocket",vmin=0, vmax=1, ax=ax, cbar=False)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
        sns.heatmap(Z2, cmap="rocket",vmin=0, vmax=1, ax=ax2, cbar=False)
        fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
        ax2.yaxis.tick_right()
        ax2.tick_params(rotation=0)
        plt.show()
        plt.savefig(images_path + "/exact.png", dpi=500)

        Z = model(plot_var1)[:, 0]
        Z = torch.reshape(Z, [100, 100])
        Z = Z.detach().numpy()
        Z2 = model(plot_var2)[:, 0]
        Z2 = torch.reshape(Z2, [100, 100])
        Z2 = Z2.detach().numpy()
        fig, (ax, ax2) = plt.subplots(ncols=2)
        fig.subplots_adjust(wspace=0.01)
        sns.heatmap(Z, cmap="rocket", vmin=0, vmax=1, ax=ax, cbar=False)
        fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
        sns.heatmap(Z2, cmap="rocket", vmin=0, vmax=1, ax=ax2, cbar=False)
        fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
        ax2.yaxis.tick_right()
        ax2.tick_params(rotation=0)
        plt.show()

        plt.savefig(images_path + "/prediction.png", dpi=500)
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

        plt.figure()
        plt.grid(True, which="both", ls=":")
        for val, scale in zip(time_steps, scale_vec):
            plot_var = torch.cat([torch.tensor(()).new_full(size=(100, 1), fill_value=val), x], 1)
            plt.plot(x, self.exact(plot_var)[:, 0], 'b-', linewidth=2, label='Exact, t=' + str(val) + 's', color=self.lighten_color('grey', scale), zorder=0)
            plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var)[:, 0].detach().numpy(), label='Predicted, t=' + str(val) + 's', marker="o", s=14,
                        color=self.lighten_color('C0', scale), zorder=10)

        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()

