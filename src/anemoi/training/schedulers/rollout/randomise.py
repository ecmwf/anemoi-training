# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: S608

from __future__ import annotations

from typing import Literal

import numpy as np
import pytorch_lightning as pl

from anemoi.training.schedulers.rollout import RolloutScheduler
from anemoi.training.schedulers.rollout.indexed import get_closest_key
from anemoi.training.utils.seeding import get_base_seed


class BaseRandom(RolloutScheduler):
    """BaseRandom Scheduler."""

    def __init__(self):
        """
        Initialise the base random rollout scheduler.

        Set the seed with the environment variable `ANEMOI_BASE_SEED` if it exists,
        """
        super().__init__()

        try:
            seed = get_base_seed()
        except AssertionError:
            seed = 42

        rnd_seed = pl.seed_everything(seed, workers=True)
        self.rng = np.random.default_rng(rnd_seed)

    def broadcast(self, value: int) -> None:
        """
        Broadcast the rollout value to all processes.

        Parameters
        ----------
        value : int
            Value to broadcast.
        """
        # TODO(Harrison Cook): Need to broadcast the rollout to all processes

    def _randomly_pick(self, rollouts: list[int]) -> int:
        """
        Randomly pick from a list of rollouts.

        Parameters
        ----------
        rollouts : list[int]
            s to choose from.

        Returns
        -------
        int
            Randomly selected rollout.
        """
        rollout = self.rng.choice(rollouts)
        self.broadcast(rollout)
        return rollout


class RandomList(BaseRandom):
    """`RandomList` is a rollout scheduler that randomly selects a rollout from a list of values."""

    def __init__(self, rollouts: list[int]):
        """
        RandomList is a rollout scheduler that randomly selects a rollout from a list of values.

        Parameters
        ----------
            rollouts : list[int]
                List of rollouts to choose from.

            Example
            -------
            ```python
            from anemoi.training.schedulers.rollout import RandomList

            RollSched = RandomList(rollouts = [1, 2, 3, 4, 5])
            RollSched.rollout_at(epoch = 1)
            # any value in the list
            RollSched.rollout_at(epoch = 2)
            # any value in the list
            ```
        """
        super().__init__()
        self._rollouts = rollouts

    @property
    def rollout(self) -> int:
        return self._randomly_pick(self._rollouts)

    @property
    def maximum_rollout(self) -> int:
        return max(self._rollouts)

    @property
    def current_maximum(self) -> int:
        return self.maximum_rollout

    def description(self) -> str:
        return f"Randomly select a rollout from {self._rollouts}"


class RandomRange(RandomList):
    """`RandomRange` is a rollout scheduler that randomly selects a rollout from a range of values."""

    def __init__(self, minimum: int = 1, maximum: int = 1, step: int = 1):
        """
        RandomRange is a rollout scheduler that randomly selects a rollout from a range of values.

        Parameters
        ----------
        minimum : int, optional
            Minimum rollout to choose from, by default 1
        maximum : int, optional
            Maximum rollout to choose from, by default 1
        step : int, optional
            Step size for the range, by default 1

        Example
        -------
        ```python
        from anemoi.training.schedulers.rollout import RandomRange

        RollSched = RandomRange(minimum = 1, maximum = 5)
        RollSched.rollout_at(epoch = 1)
        # any value between 1 and 5
        RollSched.rollout_at(epoch = 2)
        # any value between 1 and 5
        ```
        """
        super().__init__(list(range(minimum, maximum + 1, step)))

    def description(self) -> str:
        return (
            "Randomly select a rollout from the "
            f"{range(min(self._rollouts), max(self._rollouts) + 1, np.diff(self._rollouts)[0])}"
        )


class IncreasingRandom(BaseRandom):
    """IncreasingRandom is a rollout scheduler that randomly selects a rollout from an increasing range of values."""

    def __init__(
        self,
        minimum: int = 1,
        maximum: int = 1,
        range_step: int = 1,
        every_n: int = 1,
        increment: int | dict[int, int] = 1,
        step_type: Literal["step", "epoch"] = "epoch",
    ):
        """
        `IncreasingRandom` is a rollout scheduler that randomly selects a rollout from an increasing range of values.

        Parameters
        ----------
        minimum : int, optional
            Minimum rollout to choose from, by default 1
        maximum : int, optional
            Maximum rollout to choose from, can be -1 for no maximum,
            by default 1.
        range_step : int, optional
            Step size for the range, by default 1
        every_n : int, optional
            Number of steps or epochs to step the rollout value.
            If `every_n` is 0, the rollout will stay at `minimum`.
        increment : int | dict[int, int], optional
            Value to increment the rollout by `every_n_epochs`, by default 1
        step_type : Literal['step', 'epoch'], optional
            Type of step, either 'epoch' or 'batch'.
            by default 'epoch'.

        Example
        -------
        ```python
        from anemoi.training.schedulers.rollout import IncreasingRandom

        RollSched = IncreasingRandom(minimum = 1, maximum = 10, step = 1, every_n_epochs = 1)
        RollSched.rollout_at(epoch = 1)
        # any value between 1 and 1
        RollSched.rollout_at(epoch = 2)
        # any value between 1 and 2
        ```
        """
        super().__init__()

        if maximum <= -1:
            maximum = float("inf")

        self._minimum = minimum
        self._maximum = maximum
        self._range_step = range_step
        self._every_n = every_n
        self._increment = increment
        self._step_type = step_type

    @property
    def rollout(self) -> int:
        if self._every_n == 0:
            return self._minimum

        count_of_n = self.count(self._every_n, self._step_type)

        if isinstance(self._increment, int):
            maximum_value = self._minimum + self._increment * count_of_n
        else:
            sum_of_increments = [
                self._increment.get(get_closest_key(self._increment, i + 1)) for i in range(count_of_n)
            ]
            maximum_value = self._minimum + sum(sum_of_increments)

        rollouts = range(self._minimum, maximum_value + 1, self._range_step)

        return self._randomly_pick(rollouts)

    @property
    def maximum_rollout(self) -> int:
        return self._maximum

    @property
    def current_maximum(self) -> int:
        return self._minimum + ((self._epoch // self._every_n_epochs) * self._step)

    def description(self) -> str:
        return (
            f"Randomly select a rollout from the increasing range "
            f"{range(self._minimum, self._maximum, self._step)}"
            f"with the upper bound increasing by {self._step} every {self._every_n} {self._step_type}"
        )


class EpochIncreasingRandom(IncreasingRandom):
    """
    `EpochIncreasingRandom` is a rollout scheduler that randomly selects a rollout from an increasing range of values.

    The maximum is incremented every n epochs.
    """

    def __init__(
        self,
        minimum: int = 1,
        maximum: int = 1,
        range_step: int = 1,
        every_n_epochs: int = 1,
        increment: int | dict[int, int] = 1,
    ):
        """
        EpochIncreasingRandom is a rollout scheduler that randomly selects a rollout from an increasing range of values.

        The maximum is incremented every n epochs.

        Parameters
        ----------
        minimum : int, optional
            Minimum rollout to choose from, by default 1
        maximum : int, optional
            Maximum rollout to choose from, can be -1 for no maximum,
            by default 1.
        range_step : int, optional
            Step size for the range, by default 1
        every_n_epochs : int, optional
            Number of epochs to step the rollout value.
            If `every_n_epochs` is 0, the rollout will stay at `minimum`.
        increment : int | dict[int, int], optional
            Value to increment the rollout by `every_n_epochs`, by default 1

        Example
        -------
        ```python
        from anemoi.training.schedulers.rollout import EpochIncreasingRandom

        RollSched = EpochIncreasingRandom(minimum = 1, maximum = 10, range_step = 1, every_n_epochs = 1, increment = 1)
        RollSched.rollout_at(epoch = 1)
        # any value between 1 and 1
        RollSched.rollout_at(epoch = 2)
        # any value between 1 and 2

        RollSched = EpochIncreasingRandom(
            minimum = 1, maximum = 10, range_step = 1,
            every_n_epochs = 1, increment = {0: 0, 10: 1}
        )
        RollSched.rollout_at(epoch = 1)
        # any value between 1 and 1
        RollSched.rollout_at(epoch = 9)
        # any value between 1 and 1
        RollSched.rollout_at(epoch = 10)
        # any value between 1 and 2, and then increments of 1
        ```
        """
        super().__init__(minimum, maximum, range_step, every_n_epochs, increment, step_type="epoch")


class StepIncreasingRandom(IncreasingRandom):
    """
    `StepIncreasingRandom` is a rollout scheduler that randomly selects a rollout from an increasing range of values.

    The maximum is incremented every n steps.
    """

    def __init__(
        self,
        minimum: int = 1,
        maximum: int = 1,
        range_step: int = 1,
        every_n_steps: int = 1,
        increment: int | dict[int, int] = 1,
    ):
        """
        StepIncreasingRandom` is a rollout scheduler that randomly selects a rollout from an increasing range of values.

        The maximum is incremented every n steps.

        Parameters
        ----------
        minimum : int, optional
            Minimum rollout to choose from, by default 1
        maximum : int, optional
            Maximum rollout to choose from, can be -1 for no maximum,
            by default 1.
        range_step : int, optional
            Step size for the range, by default 1
        every_n_steps : int, optional
            Number of steps to step the rollout value.
            If `every_n_steps` is 0, the rollout will stay at `minimum`.
        increment : int | dict[int, int], optional
            Value to increment the rollout by `every_n_epochs`, by default 1

        Example
        -------
        ```python
        from anemoi.training.schedulers.rollout import StepIncreasingRandom

        RollSched = StepIncreasingRandom(minimum = 1, maximum = 10, range_step = 1, every_n_steps = 1, increment = 1)
        RollSched.rollout_at(step = 1)
        # any value between 1 and 1
        RollSched.rollout_at(step = 2)
        # any value between 1 and 2

        RollSched = StepIncreasingRandom(
            minimum = 1, maximum = 10, range_step = 1,
            every_n_steps = 1, increment = {0: 0, 10: 1}
        )
        RollSched.rollout_at(step = 1)
        # any value between 1 and 1
        RollSched.rollout_at(step = 9)
        # any value between 1 and 1
        RollSched.rollout_at(step = 10)
        # any value between 1 and 2, and then increments of 1
        ```
        """
        super().__init__(minimum, maximum, range_step, every_n_steps, increment, step_type="step")
