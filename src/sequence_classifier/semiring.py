from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Final, cast

import torch


class Semiring(metaclass=ABCMeta):
    zero: Final[float] = -5e3
    one: Final[float] = 0.0

    @classmethod
    def eye(cls, n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        eye = torch.full(size=(n, n), fill_value=cls.zero, dtype=dtype, device=device)
        return torch.diagonal_scatter(
            eye, torch.full(size=(n,), fill_value=cls.one, dtype=dtype, device=device)
        )

    @staticmethod
    @abstractmethod
    def sum(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def prod(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.sum(tensor, dim=dim)

    @staticmethod
    @abstractmethod
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, x + y)

    @classmethod
    def bmm(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return cls.sum(cls.mul(x.unsqueeze(-1), y.unsqueeze(-3)), dim=-2)


class LogSemiring(Semiring):
    @staticmethod
    def sum(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(tensor, dim=dim)

    @staticmethod
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.logaddexp(x, y)


class MaxSemiring(Semiring):
    @staticmethod
    def sum(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.max(tensor, dim=dim).values

    @staticmethod
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, y)
