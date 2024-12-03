#!/usr/bin/env python

""" 
    GPyTorch Implementation of the GP Regression model for the data-augmented MPC.
"""

import torch
import gpytorch
from torch.distributions import Normal
import torch.distributions as base_distributions
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        # Make predictions using fast predictive variances
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            predictive_distribution = self.forward(x)
            return predictive_distribution.mean, predictive_distribution.variance

    def get_cache(self):
        # Implement a method to compute the cache (if necessary)
        pass
# class ApproximateGPModel(gpytorch.models.ApproximateGP):
#     def __init__(self, inducing_points):
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
#             inducing_points.size(-1))
#         variational_strategy = gpytorch.variational.VariationalStrategy(
#             self, inducing_points, variational_distribution, 
#             learn_inducing_locations=True
#         )
#         super().__init__(variational_strategy)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ApproximateGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(ApproximateGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=18))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class MeanVarModelWrapper(torch.nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean #, output_dist.variance
    
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class CustomGaussianLikelihood(gpytorch.likelihoods.GaussianLikelihood):
    def __init__(self, noise=0.1):
        super().__init__(noise=noise)

    def forward(self, function_samples, noise=None):
        if noise is None:
            noise = self.noise
        return super().forward(function_samples, noise=noise)
    
class CustomLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self):
        super().__init__()

    def forward(self, function_samples: torch.Tensor) -> torch.distributions.Normal:
        # Define a custom noise model if needed
        noise = torch.ones_like(function_samples)  # Example: constant noise
        return gpytorch.distributions.Normal(function_samples, noise)
