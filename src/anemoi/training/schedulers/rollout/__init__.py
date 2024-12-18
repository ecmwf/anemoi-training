# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class RolloutScheduler(ABC):
    """
    `RolloutScheduler` is an abstract base class for rollout schedulers.

    A rollout scheduler is an object that manages the rollout of a training loop.

    ```python
    RollSched = RolloutScheduler()

    for epoch in range(20):
        for step in range(100):
            y = model(x, rollout = RollSched.rollout)

            RollSched.step()
        RollSched.step_epoch()
    ```
    """

    _epoch: int = 0
    _step: int = 0

    @property
    @abstractmethod
    def rollout(self) -> int:
        """Get the current rollout value."""
        error_msg = "`rollout` property not implemented by parent class."
        raise NotImplementedError(error_msg)

    @property
    @abstractmethod
    def maximum_rollout(self) -> int:
        """Get maximum rollout possible."""
        error_msg = "`maximum_rollout` property not implemented by parent class."
        raise NotImplementedError(error_msg)

    @property
    def current_maximum(self) -> int:
        """Get the current maximum rollout value."""
        return self.rollout

    def __int__(self) -> int:
        return int(self.rollout)

    def rollout_at(self, step: int | None = None, epoch: int | None = None) -> int:
        """
        Get the rollout at a specific step and epoch.

        Parameters
        ----------
        step : int, optional
            Step value to override with, by default None
        epoch : int, optional
            Epoch value to override with, by default None

        Returns
        -------
        int
            Rollout value at the specified step and epoch.
        """
        step_ = self._step
        epoch_ = self._epoch

        self._step = step if step is not None else step_
        self._epoch = epoch if epoch is not None else epoch_

        rollout = self.rollout

        self._step = step_
        self._epoch = epoch_

        return rollout

    def step(self, count: int = 1, /) -> None:
        """Step the scheduler by a count."""
        self._step += count

    def step_epoch(self, count: int = 1, /) -> None:
        """Step the scheduler by a count of epochs."""
        self._epoch += count

    def count(self, n_epochs: int | None = None, n_steps: int | None = None) -> int:
        """
        Get the count of steps or epochs.

        Parameters
        ----------
        n_epochs : int | None, optional
            Number of epochs to count, by default None
        n_steps : int | None, optional
            Number of steps to count, by default None

        Returns
        -------
        int
            Count of steps or epochs.

        Raises
        ------
        ValueError
            If both `n_epochs` and `n_steps` are given, or if neither are given.
        """
        if (n_epochs is not None and n_steps is not None) or (n_epochs is None and n_steps is None):
            error_msg = "Only one of `n_epochs` or `n_steps` can be given."
            raise ValueError(error_msg)

        if n_epochs is not None:
            return self._epoch // n_epochs
        return self._step // n_steps

    @abstractmethod
    def description(self) -> str:
        """Description of the rollout scheduler."""
        error_msg = "`description` method not implemented by parent class."
        raise NotImplementedError(error_msg)


class Static(RolloutScheduler):
    """`Static` is a rollout scheduler that always returns the same rollout value."""

    def __init__(self, rollout_value: int):
        """
        `Static` is a rollout scheduler that always returns the same rollout value.

        Parameters
        ----------
        rollout_value : int
            Rollout value to return.

        Example
        -------
        ```python
        from anemoi.training.schedulers.rollout import Static
        RollSched = Static(rollout_value = 5)
        RollSched.rollout_at(epoch = 1)
        # 5
        RollSched.rollout_at(epoch = 5)
        # 5
        ```
        """
        self._rollout_value = rollout_value

    @property
    def rollout(self) -> int:
        return self._rollout_value

    @property
    def maximum_rollout(self) -> int:
        return self._rollout_value

    def description(self) -> str:
        return f"Static rollout value of {self._rollout_value}."
