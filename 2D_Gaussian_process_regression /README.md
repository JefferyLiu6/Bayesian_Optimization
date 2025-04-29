# Documentation for translating 2D GPRegression to GPyTorch

The aim of this work is to understand  for the research project on multi-fidelity Bayesian optimization (MFBO)

The 2D quadratic function used in . 

## Part1: importing libraries

## Part2: Data Generation

I didn't change the way to generate data in order to compare the correctness of my work comparing to the original sk aproach. 

## Part3: Data Splitting and Normalization

The sk-learn approcah called the function "GaussianProcessRegressor()"


## Part4: Define the GP Model and Kernel

The sk-learn approcah. I will break it down line by line explaining how to achieve it in gpytorch

- The function "guess_l = (2., 1.)" a tuple defines with initial guesses for the length scale parameters of the Radial Basis Function (RBF) kernel. 2.0 for the x-dim, 1.0 for the y-dim.




- bounds_l = ((1e-1,100.),) * 2

- guess_n = 1.

- bounds_n = (1e-20, 10.)

- kernel = (RBF(length_scale=guess_l, length_scale_bounds=bounds_l) + WhiteKernel(noise_level=guess_n, noise_level_bounds=bounds_n))

- gpr = GaussianProcessRegressor(kernel, normalize_y=True)

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)