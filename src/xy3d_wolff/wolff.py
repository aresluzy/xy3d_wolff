from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import wolff_update, wolff_update_with_estimator


@dataclass
class WolffParameters:
    J: float
    T: float


class WolffClusterUpdater:
    def __init__(self, params: WolffParameters):
        self.params = params

    @property
    def J(self) -> float:
        return self.params.J

    @property
    def T(self) -> float:
        return self.params.T

    def step(self, spins: np.ndarray) -> int:
        return wolff_update(spins, self.J, self.T)

    def step_with_estimator(self, spins: np.ndarray):
        return wolff_update_with_estimator(spins, self.J, self.T)
