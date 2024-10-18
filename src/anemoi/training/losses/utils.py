# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import logging
import operator
import uuid
from collections.abc import Sequence

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


def grad_scaler(
    module: nn.Module,
    grad_in: tuple[torch.Tensor, ...],
    grad_out: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...] | None:
    """Scales the loss gradients.

    Uses the formula in https://arxiv.org/pdf/2306.06079.pdf, section 4.3.2

    Use <module>.register_full_backward_hook(grad_scaler, prepend=False) to register this hook.

    Parameters
    ----------
    module : nn.Module
        Loss object (not used)
    grad_in : tuple[torch.Tensor, ...]
        Loss gradients
    grad_out : tuple[torch.Tensor, ...]
        Output gradients (not used)

    Returns
    -------
    tuple[torch.Tensor, ...]
        Re-scaled input gradients

    """
    del module, grad_out
    # first grad_input is that of the predicted state and the second is that of the "ground truth" (== zero)
    channels = grad_in[0].shape[-1]  # number of channels
    channel_weights = torch.reciprocal(torch.sum(torch.abs(grad_in[0]), dim=1, keepdim=True))  # channel-wise weights
    new_grad_in = (
        (channels * channel_weights) / torch.sum(channel_weights, dim=-1, keepdim=True) * grad_in[0]
    )  # rescaled gradient
    return new_grad_in, grad_in[1]


TENSOR_SPEC = tuple[int | tuple[int], torch.Tensor]


class Shape:
    """Shape resolving object"""

    def __init__(self, func):
        self.func = func

    def __getitem__(self, dimension: int) -> int:
        return self.func(dimension)


class ScaleTensor:
    """Dynamically resolved tensor scaling class.

    Allows a user to specify a scalar and the dimensions it should be applied to.
    The class will then enforce that additional scalars are compatible with the specified dimensions.

    When `get_scalar` or `scale` is called, the class will return the product of all scalars, resolved
    to the dimensional size of the input tensor.

    Additionally, the class can be subsetted to only return a subset of the scalars, but
    only from those given names.

    Examples
    --------
    >>> tensor = torch.randn(3, 4, 5)
    >>> scalars = ScaleTensor((0, torch.randn(3)), (1, torch.randn(4)))
    >>> scaled_tensor = scalars.scale(tensor)
    >>> scalars.get_scalar(tensor.shape).shape
    torch.Size([3, 4, 1])
    """

    tensors: dict[str, TENSOR_SPEC]
    _specified_dimensions: list[tuple[int]]

    def __init__(
        self,
        scalars: dict[str, TENSOR_SPEC] | TENSOR_SPEC | None = None,
        *tensors: TENSOR_SPEC,
        **named_tensors: dict[str, TENSOR_SPEC],
    ):
        """ScaleTensor constructor.

        Parameters
        ----------
        scalars : dict[str, TENSOR_SPEC] | TENSOR_SPEC | None, optional
            Scalars to initalise with, by default None
        tensors : TENSOR_SPEC
            Args form of (dimension, tensor) to add to the scalars
            Will be given a random uuid name
        named_tensors : dict[str, TENSOR_SPEC]
            Kwargs form of {name: (dimension, tensor)} to add to the scalars
        """
        self.tensors = {}
        self._specified_dimensions = []

        scalars = scalars or {}
        scalars.update(named_tensors)

        for name, tensor_spec in scalars.items():
            self.add_scalar(*tensor_spec, name=name)

        for tensor_spec in tensors:
            self.add_scalar(*tensor_spec)

    @property
    def shape(self) -> Shape:
        """Get the shape of the scale tensor.

        Returns a Shape object to be indexed,
        Will only resolve those dimensions specified in the tensors.
        """

        def get_dim_shape(dimension: int) -> int:
            for dim_assign, tensor in self.tensors.values():
                if isinstance(dim_assign, tuple) and dimension in dim_assign:
                    return tensor.shape[list(dim_assign).index(dimension)]

            error_msg = (
                f"Could not find shape of dimension {dimension} with tensors in dims {list(self.tensors.keys())}"
            )
            raise IndexError(error_msg)

        return Shape(get_dim_shape)

    def subset(self, scalars: str | Sequence[str]) -> ScaleTensor:
        """Get subset of the scalars.

        Parameters
        ----------
        scalars : str | Sequence[str]
            Name/s of the scalars to get

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        if isinstance(scalars, str):
            scalars = [scalars]
        return ScaleTensor(**{name: self.tensors[name] for name in scalars})

    def validate_scaler(self, dimension: int | tuple[int], scalar: torch.Tensor) -> None:
        """Check if the scalar is compatible with the given dimension.

        Parameters
        ----------
        dimension : int | tuple[int]
            Dimensions to check `scalar` against
        scalar : torch.Tensor
            Scalar tensor to check

        Raises
        ------
        ValueError
            If the scalar is not compatible with the given dimension
        """
        if isinstance(dimension, int):
            dimension = [dimension]

        for scalar_dim, dim in enumerate(dimension):
            if dim not in self or scalar.shape[scalar_dim] == 1 or self.shape[dim] == 1:
                continue

            if self.shape[dim] != scalar.shape[scalar_dim]:
                raise ValueError(
                    f"Scaler shape {scalar.shape} at dimension {scalar_dim} does not match shape of scalar at dimension {dim}. Expected {self.shape[dim]}",
                )

    def add_scalar(
        self,
        dimension: int | tuple[int],
        scalar: torch.Tensor,
        *,
        name: str | None = None,
        join_operation: str = "multiply",
    ) -> None:
        """Add new scalar to be applied along `dimension`.

        Parameters
        ----------
        dimension : int | tuple[int]
            Dimension/s to apply the scalar to
        scalar : torch.Tensor
            Scalar tensor to apply
        name : str | None, optional
            Name of the scalar, by default None
        """
        if isinstance(dimension, int):
            if len(scalar.shape) == 1:
                dimension = (dimension,)
            else:
                dimension = tuple(dimension + i for i in range(len(scalar.shape)))

        self.validate_scaler(dimension, scalar)
        if name is None:
            name = str(uuid.uuid4())

        if dimension in self.tensors:
            self.tensors[name] = (
                dimension,
                getattr(operator, join_operation)(self.tensors[dimension], scalar),
            )
        else:
            self.tensors[name] = (dimension, scalar)
            self._specified_dimensions.append(dimension)

    def scale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scale a given tensor by the scalars.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to scale

        Returns
        -------
        torch.Tensor
            Scaled tensor
        """
        return tensor * self.get_scalar(tensor.shape)

    def get_scalar(self, shape: Sequence[int]) -> torch.Tensor:
        """Get completely resolved scalar tensor.

        Parameters
        ----------
        shape : Sequence[int]
            Shape of the tensor to resolve the scalars to
            Used to resolve relative indices, and add singleton dimensions

        Returns
        -------
        torch.Tensor
            Scalar tensor

        Raises
        ------
        ValueError
            If resolving relative indices is invalid
        """
        complete_scalar = torch.ones(1)

        for dims, scalar in self.tensors.values():
            if any(d < 0 for d in dims):
                absolute_dims = [d if d >= 0 else len(shape) + d for d in dims]
                try:
                    self.validate_scaler(
                        absolute_dims,
                        scalar,
                    )  # Validate tensor in case of resolution of relative indexing
                    dims = absolute_dims
                except ValueError as e:
                    raise ValueError(f"Resolving relative indices of {dims} was invalid.") from e

            missing_dims = list(d for d in range(len(shape)) if d not in dims)
            reshape = [1] * len(missing_dims)
            reshape.extend(scalar.shape)

            reshaped_scalar = scalar.view(reshape)
            reshaped_scalar = torch.moveaxis(reshaped_scalar, list(range(len(shape))), (*missing_dims, *dims))

            if complete_scalar is None:
                complete_scalar = reshaped_scalar
            else:
                complete_scalar = complete_scalar * reshaped_scalar
        return complete_scalar

    def __repr__(self):
        return "ScalarTensor:\n" f"With {list(self.tensors.keys())}\nWith dims: {self._specified_dimensions}"

    def __contains__(self, dimension: int | tuple[int]) -> bool:
        if isinstance(dimension, tuple):
            return any(x in self._specified_dimensions for x in dimension)

        result = False
        for dim_assign, _ in self.tensors.values():
            result = dimension in dim_assign or result
        return result

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def __len__(self):
        return len(self.tensors)
