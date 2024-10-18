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
import uuid
from typing import TYPE_CHECKING
from typing import Callable

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)


def grad_scalar(
    module: nn.Module,
    grad_in: tuple[torch.Tensor, ...],
    grad_out: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...] | None:
    """Scales the loss gradients.

    Uses the formula in https://arxiv.org/pdf/2306.06079.pdf, section 4.3.2

    Use <module>.register_full_backward_hook(grad_scalar, prepend=False) to register this hook.

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
    """Shape resolving object."""

    def __init__(self, func: Callable[[int], int]):
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
    >>> scalars.get_scalar(tensor.ndim).shape
    torch.Size([3, 4, 1])
    >>> scalars.add_scalar(-1, torch.randn(5))
    >>> scalars.get_scalar(tensor.ndim).shape
    torch.Size([3, 4, 5])
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

    def validate_scalar(self, dimension: int | tuple[int], scalar: torch.Tensor) -> None:
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
                error_msg = (
                    f"Scalar shape {scalar.shape} at dimension {scalar_dim}"
                    f"does not match shape of scalar at dimension {dim}. Expected {self.shape[dim]}",
                )
                raise ValueError(error_msg)

    def add_scalar(
        self,
        dimension: int | tuple[int],
        scalar: torch.Tensor,
        *,
        name: str | None = None,
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

        try:
            self.validate_scalar(dimension, scalar)
        except ValueError as e:
            error_msg = f"Validating tensor {name!r} raised an invalidation."
            raise ValueError(error_msg) from e

        if name is None:
            name = str(uuid.uuid4())

        if name in self.tensors:
            self.tensors[name] = (
                dimension,
                self.tensors[name] * scalar,
            )
        else:
            self.tensors[name] = (dimension, scalar)
            self._specified_dimensions.append(dimension)

    def subset(self, scalars: str | Sequence[str]) -> ScaleTensor:
        """Get subset of the scalars, filtering by name.

        See `.subset_by_dim` for subsetting by affected dimensions.

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

    def subset_by_dim(self, dimensions: int | Sequence[int]) -> ScaleTensor:
        """Get subset of the scalars, filtering by dimension.

        See `.subset` for subsetting by name.

        Parameters
        ----------
        dimensions : int | Sequence[int]
            Dimensions to get scalars of

        Returns
        -------
        ScaleTensor
            Subset of self
        """
        subset_scalars: dict[str, TENSOR_SPEC] = {}

        if isinstance(dimensions, int):
            dimensions = (dimensions,)

        for name, (dim, scalar) in self.tensors.items():
            if isinstance(dim, int):
                dim = (dim,)
            if len(set(dimensions).intersection(dim)) > 0:
                subset_scalars[name] = (dim, scalar)

        return ScaleTensor(**subset_scalars)

    def resolve(self, ndim: int) -> ScaleTensor:
        """Resolve relative indexes in scalars by associating against ndim.

        i.e. if a scalar was given as effecting dimension -1,
        and `ndim` was provided as 4, the scalar will be fixed
        to effect dimension 3.

        Parameters
        ----------
        ndim : int
            Number of dimensions to resolve relative indexing against

        Returns
        -------
        ScaleTensor
            ScaleTensor with all relative indexes resolved
        """
        resolved_scalars: dict[str, TENSOR_SPEC] = {}

        for name, (dims, scalar) in self.tensors.items():
            if any(d < 0 for d in dims):
                dims = [d if d >= 0 else ndim + d for d in dims]
            resolved_scalars[name] = (dims, scalar)
        return ScaleTensor(**resolved_scalars)

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
        return tensor * self.get_scalar(tensor.ndim)

    def get_scalar(self, ndim: int) -> torch.Tensor:
        """Get completely resolved scalar tensor.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the tensor to resolve the scalars to
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

        tensors = self.resolve(ndim).tensors

        for dims, scalar in tensors.values():
            missing_dims = [d for d in range(ndim) if d not in dims]
            reshape = [1] * len(missing_dims)
            reshape.extend(scalar.shape)

            reshaped_scalar = scalar.view(reshape)
            reshaped_scalar = torch.moveaxis(reshaped_scalar, list(range(ndim)), (*missing_dims, *dims))

            complete_scalar = complete_scalar if complete_scalar is None else complete_scalar * reshaped_scalar

        return complete_scalar

    def __repr__(self):
        return f"ScalarTensor:\n - With {list(self.tensors.keys())}\n - With dims: {self._specified_dimensions}"

    def __contains__(self, dimension: int | tuple[int] | str) -> bool:
        """Check if either scalar by name or dimension by int is being scaled."""
        if isinstance(dimension, tuple):
            return any(x in self._specified_dimensions for x in dimension)
        if isinstance(dimension, str):
            return dimension in self.tensors

        result = False
        for dim_assign, _ in self.tensors.values():
            result = dimension in dim_assign or result
        return result

    def __len__(self):
        return len(self.tensors)
