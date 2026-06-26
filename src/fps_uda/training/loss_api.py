from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, MutableMapping, Protocol

import torch


LossContext = MutableMapping[str, Any]


@dataclass
class LossOutput:
    name: str
    value: torch.Tensor
    logs: Dict[str, float] = field(default_factory=dict)


class LossTerm(Protocol):
    def __call__(self, ctx: LossContext) -> LossOutput:
        ...


LossCallable = Callable[[LossContext], LossOutput]
