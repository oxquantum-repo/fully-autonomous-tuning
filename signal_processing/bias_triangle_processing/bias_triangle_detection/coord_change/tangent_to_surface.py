from typing import Callable, Dict, List, Optional, Union

import gpytorch
import numpy as np
import torch
from torch import nn

NamedFunc = Callable[[Dict[str, float]], Dict[str, float]]
ArrayFunc = Callable[[np.ndarray], np.ndarray]


def is_orthonormal(matrix: Union[np.ndarray, torch.Tensor], atol: float = 1e-5) -> bool:
    framework = torch if isinstance(matrix, torch.Tensor) else np
    return framework.allclose(
        matrix @ matrix.T,
        framework.eye(len(matrix), dtype=framework.float64),
        atol=atol,
    )


class EuclideanTransformation:
    """Simple euclidean transformation class (rotation + translation))"""

    def __init__(
        self,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
        name_orig_coords: Optional[List[str]] = None,
        name_new_coords: Optional[List[str]] = None,
    ):
        """Euclidean transform with optinal named coordinates.

        Args:
            rotation_matrix (np.ndarray): rotation from original to target system.
            translation_vector (np.ndarray): translation to target system.
            name_orig_coords (Optional[List[str]]): names of original coordinates
            name_new_coords (Optional[List[str]]): names of target coordinates
        """
        # assert rotation_matrix is orthonormal
        assert is_orthonormal(rotation_matrix), "rotation matrix is not orthonormal"
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.name_orig_coords = name_orig_coords
        self.name_new_coords = name_new_coords

    def named_transform(
        self,
        func: ArrayFunc,
        from_names: Optional[List[str]] = None,
        to_names: Optional[List[str]] = None,
    ) -> Union[ArrayFunc, NamedFunc]:
        if from_names is None and to_names is None:
            return func
        if from_names is not None and to_names is not None:

            def named_func(x: Dict[str, float]) -> Dict[str, float]:
                arr = np.array([x[name] for name in from_names])
                return {name: val for name, val in zip(to_names, func(arr))}

            return named_func
        raise ValueError("either both or none of from_names and to_names must be given")

    def __call__(
        self, point: Union[np.ndarray, Dict[str, float]]
    ) -> Union[np.ndarray, Dict[str, float]]:
        return self.named_transform(
            self._call, self.name_orig_coords, self.name_new_coords
        )(point)

    def _call(self, point: np.ndarray) -> np.ndarray:
        return (
            self.rotation_matrix @ (point - self.translation_vector).reshape(-1, 1)
        ).squeeze()

    def inverse(
        self, point: Union[np.ndarray, Dict[str, float]]
    ) -> Union[np.ndarray, Dict[str, float]]:
        return self.named_transform(
            self._inverse, self.name_new_coords, self.name_orig_coords
        )(point)

    def _inverse(self, point: np.ndarray) -> np.ndarray:
        return point @ self.rotation_matrix + self.translation_vector


def surface_points_to_rotation_matrix(surface_points: np.ndarray) -> np.ndarray:
    """Get rotation matrix from standard basis to basis whose z-axis (last coord) is normal to the surface at the first point.
    The rotation matrix is obtained by first fitting a GP to the surface, using the normal as new z axis and tangent plane as the rest of the coordinates.
    """
    # Fit surface with GP
    train_x, train_y = surface_points[:, :-1], surface_points[:, -1]
    train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    model = get_model(train_x, train_y)
    model = train_model(model, train_x, train_y).pred_mean
    input_vect = train_x[0].unsqueeze(0)
    # get gradient of surface (avg of GP) at first point
    grad = get_grad_with_respect_to_input(model, input_vect)[0]
    # get tangent vectors to surface
    tangent_vectors = get_tangent_vectors_from_gradient(grad)
    # rotation matrix from orthonormal basis of tangent space and normal vector
    rotation_matrix = get_rotation_matrix_from_tangent_vectors(tangent_vectors)
    return rotation_matrix.cpu().numpy()


def get_grad_with_respect_to_input(
    model: nn.Module, input_vect: torch.Tensor
) -> torch.Tensor:
    """computes the gradient of the model with respect to the input"""
    input_vect.requires_grad = True
    output = model(input_vect)
    output.backward()
    grad = input_vect.grad
    input_vect.requires_grad = False
    return grad


def get_tangent_vectors_from_gradient(grad: torch.Tensor) -> torch.Tensor:
    """returns the tangent vectors from the gradient"""
    diagonal = torch.eye(grad.shape[-1])
    return torch.concat([diagonal, grad.reshape(-1, 1)], dim=-1)


def orth(vectors: torch.Tensor) -> torch.Tensor:
    # Gram-Schmidt process
    q, _ = torch.qr(vectors.T)
    return q.T


def normal_to_space(orthonormal_matrix: torch.Tensor) -> torch.Tensor:
    # assert orthonormal_matrix is orthonormal
    assert is_orthonormal(orthonormal_matrix), "matrix is not orthonormal"

    # Create a random vector n with the same number of dimensions as orthonormal_matrix
    orthonormal_matrix = orthonormal_matrix.float()
    random_vector = torch.randn(orthonormal_matrix.size(1), dtype=torch.float32)

    # Project n onto the subspace orthogonal to the column space of O
    perpendicular_vector = (
        random_vector
        - (orthonormal_matrix * random_vector).sum(dim=1) @ orthonormal_matrix
    )

    # Normalize n to make it a unit vector
    normal_vector = perpendicular_vector / torch.norm(perpendicular_vector)

    return normal_vector


def get_rotation_matrix_from_tangent_vectors(
    tangent_vectors: torch.Tensor,
) -> torch.Tensor:
    # orthonormal basis of tangent space plus normal vector
    orth_vects = orth(tangent_vectors)
    _normal = normal_to_space(orth_vects).unsqueeze(0)
    return torch.cat([orth_vects, _normal], dim=0)


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def pred_mean(self, x):
        self.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self(x))
        return observed_pred.mean


def get_model(train_x: torch.Tensor, train_y: torch.Tensor) -> ExactGPModel:
    """returns a GP model"""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    return model


def train_model(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor):
    """Basic training loop obtained from GPytorch docs. Hardcoded for now."""
    # Find optimal model hyperparameters
    model.train()
    model.likelihood.train()
    training_iter = 50

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print(
            "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
            % (
                i + 1,
                training_iter,
                loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item(),
            )
        )
        optimizer.step()
    return model
