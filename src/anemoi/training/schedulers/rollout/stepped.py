# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

from typing import Literal

from anemoi.training.schedulers.rollout import RolloutScheduler
from anemoi.training.schedulers.rollout.indexed import get_closest_key


class Stepped(RolloutScheduler):
    """`Stepped` is a base rollout scheduler that steps the rollout value at the end of each n steps or epochs."""

    def __init__(
        self,
        minimum: int,
        maximum: int,
        every_n: int,
        increment: int | dict[int, int],
        step_type: Literal["step", "epoch"] = "epoch",
    ):
        """
        `SteppedRollout` is a base rollout scheduler that steps the rollout value at the end of each n steps or epochs.

        Parameters
        ----------
        minimum : int
            Minimum rollout value.
        maximum : int
            Maximum rollout value.
            Can be -1 for no maximum.
        every_n : int
            Number of steps or epochs to step the rollout value.
            If `every_n` is 0, the rollout will stay at `minimum`.
        increment : int | dict[int, int], optional
            Value to increment the rollout by.
            Can be an int or dictionary, where the keys represent the value of `step_type`
            and the values represent the increment.
            Will round down to the closest key.
            i.e. {0: 1, 10: 2} will increment by 1 until 10, then by 2.
            by default 1.
        step_type : Literal['step', 'epoch'], optional
            Type of step, either 'epoch' or 'step'.
            by default 'epoch'.

        Example
        -------
        ```python
        from anemoi.training.schedulers.rollout.stepped import Stepped

        RollSched = Stepped(minimum = 1, maximum = 10, every_n = 5, increment = 1)
        RollSched.rollout_at(epoch = 2)
        # 1
        RollSched.rollout_at(epoch = 5)
        # 2

        RollSched = Stepped(minimum = 1, maximum = 10, every_n = 5, increment = 2)
        RollSched.rollout_at(epoch = 2)
        # 1
        RollSched.rollout_at(epoch = 5)
        # 3

        RollSched = Stepped(minimum = 1, maximum = 10, every_n = 1, increment = {0: 0, 10: 1})
        RollSched.rollout_at(epoch = 2)
        # 1
        RollSched.rollout_at(epoch = 9)
        # 1
        RollSched.rollout_at(epoch = 10)
        # 2, and then increments of 1
        ```
        """
        super().__init__()

        if maximum <= -1:
            maximum = float("inf")

        self._minimum = minimum
        self._maximum = maximum
        self._every_n = every_n
        self._increment = increment
        self._step_type = step_type

    @property
    def rollout(self) -> int:
        if self._every_n == 0:
            return self._minimum

        count_of_n = self.count(self._every_n, self._step_type)

        if isinstance(self._increment, int):
            return min(self._maximum, self._minimum + self._increment * count_of_n)

        sum_of_increments = [
            self._increment.get(get_closest_key(self._increment, i + 1 if self._step_type == "epoch" else i))
            for i in range(count_of_n)
        ]
        return min(self._maximum, self._minimum + sum(sum_of_increments))

    @property
    def maximum_rollout(self) -> int:
        return self._maximum

    def description(self) -> str:
        return (
            "Stepped rollout scheduler stepping between"
            f"{self._minimum} and {self._maximum} by {self._increment} for {self._every_n} {self._step_type}s."
        )


class EpochStepped(Stepped):
    """`EpochStepped` is a rollout scheduler that steps the rollout value at the end of each n epochs."""

    def __init__(self, minimum: int, maximum: int, every_n_epochs: int = 1, increment: int = 1):
        """
        `EpochStepped` is a rollout scheduler that steps the rollout value at the end of each n epochs.

        Parameters
        ----------
        minimum : int
            The minimum value for the scheduler.
        maximum : int
            The maximum value for the scheduler.
        every_n_epochs : int, optional
            The number of epochs after which the value is incremented, by default 1.
        increment : int, optional
            The amount by which the value is incremented, by default 1.
        """
        super().__init__(minimum, maximum, every_n_epochs, increment, step_type="epoch")


class StepStepped(Stepped):
    """`StepStepped` is a rollout scheduler that steps the rollout value at the end of each n steps."""

    def __init__(self, minimum: int, maximum: int, every_n_steps: int = 1000, increment: int = 1):
        """
        `StepStepped` is a rollout scheduler that steps the rollout value at the end of each n steps.

        Parameters
        ----------
        minimum : int
            The minimum value for the scheduler.
        maximum : int
            The maximum value for the scheduler.
        every_n_steps : int, optional
            The number of steps after which the value is incremented, by default 1000.
        increment : int, optional
            The amount by which the value is incremented, by default 1.
        """
        super().__init__(minimum, maximum, every_n_steps, increment, step_type="step")
