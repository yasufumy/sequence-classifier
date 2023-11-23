from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import cast, final

import torch
from torch import nn

from sequence_classifier.semiring import LogSemiring, MaxSemiring, Semiring


def reduce(semiring: type[Semiring], potentials: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, num_tags, _ = potentials.size()

    n = sequence_length.bit_length()
    padding_length = (1 << n) - sequence_length
    padding_value = semiring.eye(
        n=num_tags, dtype=potentials.dtype, device=potentials.device
    )[None, None]

    potentials = torch.cat(
        (
            potentials,
            padding_value.repeat(batch_size, padding_length, 1, 1),
        ),
        dim=1,
    )

    for _ in range(n):
        potentials = semiring.bmm(potentials[:, 0::2], potentials[:, 1::2])

    return cast(torch.Tensor, potentials.squeeze(dim=1))


class BaseLogPartitions(metaclass=ABCMeta):
    @property
    @abstractmethod
    def value(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def marginals(self) -> torch.Tensor | None:
        raise NotImplementedError()


class UnitLogPartitions(BaseLogPartitions):
    def __init__(self, logits: torch.Tensor):
        self.__logits = logits

    @property
    def value(self) -> torch.Tensor:
        return cast(
            torch.Tensor, LogSemiring.sum(self.__logits, dim=-1).squeeze(dim=-1)
        )

    @property
    def marginals(self) -> torch.Tensor | None:
        return None


class LogPartitions(BaseLogPartitions):
    def __init__(
        self,
        start_potentials: torch.Tensor,
        potentials: torch.Tensor,
        mask: torch.Tensor,
    ):
        transitions = reduce(semiring=LogSemiring, potentials=potentials)
        transitions = transitions + start_potentials[..., None]
        log_partitions = LogSemiring.sum(LogSemiring.sum(transitions, dim=-1), dim=-1)

        self.__potentials = potentials
        self.__log_partitions = log_partitions
        self.__mask = mask

    @property
    def value(self) -> torch.Tensor:
        return self.__log_partitions

    @property
    def marginals(self) -> torch.Tensor | None:
        (marginals,) = torch.autograd.grad(
            self.__log_partitions.sum(), self.__potentials, create_graph=True
        )
        return cast(torch.Tensor, marginals * self.__mask[:, 1:, None, None])


class BaseCrfDistribution(metaclass=ABCMeta):
    @final
    def log_likelihood(self, tag_indices: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.log_scores(tag_indices=tag_indices) - self.log_partitions.value,
        )

    @final
    def marginal_log_likelihood(self, tag_bitmap: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.log_multitag_scores(tag_bitmap=tag_bitmap) - self.log_partitions.value,
        )

    @property
    def marginals(self) -> torch.Tensor | None:
        return self.log_partitions.marginals

    @abstractmethod
    def log_scores(self, tag_indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def log_multitag_scores(self, tag_bitmap: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def log_partitions(self) -> BaseLogPartitions:
        raise NotImplementedError()

    @property
    @abstractmethod
    def max(sefl) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def argmax(self) -> torch.Tensor:
        raise NotImplementedError()


class CrfUnitDistribution(BaseCrfDistribution):
    def __init__(self, logits: torch.Tensor):
        self.__logits = logits

    def log_scores(self, tag_indices: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.__logits.squeeze(dim=1)
            .gather(dim=-1, index=tag_indices)
            .squeeze(dim=-1),
        )

    def log_multitag_scores(self, tag_bitmap: torch.Tensor) -> torch.Tensor:
        logits = self.__logits.masked_fill(~tag_bitmap, Semiring.zero)
        return cast(torch.Tensor, LogSemiring.sum(logits, dim=-1).squeeze(dim=-1))

    @property
    def log_partitions(self) -> UnitLogPartitions:
        return UnitLogPartitions(logits=self.__logits)

    @property
    def max(self) -> torch.Tensor:
        return cast(
            torch.Tensor, MaxSemiring.sum(self.__logits, dim=-1).squeeze(dim=-1)
        )

    @property
    def argmax(self) -> torch.Tensor:
        return torch.max(self.__logits, dim=-1).indices


class CrfDistribution(BaseCrfDistribution):
    def __init__(
        self,
        start_potentials: torch.Tensor,
        potentials: torch.Tensor,
        mask: torch.Tensor,
        padding_index: int,
    ):
        self.__start_potentials = start_potentials
        self.__potentials = potentials
        self.__mask = mask
        self.__padding_index = padding_index

    def log_scores(self, tag_indices: torch.Tensor) -> torch.Tensor:
        log_scores = self.__start_potentials.gather(
            index=tag_indices[:, [0]], dim=-1
        ).squeeze(dim=-1)
        log_scores += (
            self.__potentials.take_along_dim(
                indices=tag_indices[:, 1:, None, None], dim=-1
            )
            .take_along_dim(indices=tag_indices[:, :-1, None, None], dim=-2)
            .squeeze(dim=(-1, -2))
            * self.__mask[:, 1:]
        ).sum(dim=-1)
        return cast(torch.Tensor, log_scores)

    def log_multitag_scores(self, tag_bitmap: torch.Tensor) -> torch.Tensor:
        # Make sure to deactivate masked indices
        tag_bitmap = tag_bitmap & self.__mask[..., None]
        # Create transition mask
        mask = tag_bitmap[:, :-1, :, None] & tag_bitmap[:, 1:, None, :]
        # Flip masked indices. No need to touch them.
        mask |= (~self.__mask)[:, 1:, None, None]
        potentials = self.__potentials * mask + Semiring.zero * ~mask
        # Same as log_partitions
        start_potentials = self.__start_potentials.masked_fill(
            ~tag_bitmap[:, 0], Semiring.zero
        )
        return LogPartitions(
            start_potentials=start_potentials, potentials=potentials, mask=mask
        ).value

    @property
    def log_partitions(self) -> LogPartitions:
        return LogPartitions(
            start_potentials=self.__start_potentials,
            potentials=self.__potentials,
            mask=self.__mask,
        )

    @property
    def max(self) -> torch.Tensor:
        transitions = reduce(semiring=MaxSemiring, potentials=self.__potentials)
        transitions = transitions + self.__start_potentials[..., None]
        return MaxSemiring.sum(MaxSemiring.sum(transitions, dim=-1), dim=-1)

    @property
    def argmax(self) -> torch.Tensor:
        (transition_sequence,) = torch.autograd.grad(
            self.max.sum(),
            self.__potentials,
            create_graph=True,
        )
        transition_sequence = transition_sequence.long()

        tag_bitmap = transition_sequence.sum(dim=-2)
        tag_indices = tag_bitmap.argmax(dim=-1)

        start = transition_sequence[:, 0].sum(dim=-1).argmax(dim=-1, keepdim=True)

        return cast(
            torch.Tensor,
            torch.cat([start, tag_indices], dim=-1) * self.__mask
            + (~self.__mask) * self.__padding_index,
        )


class Crf(nn.Module):
    def __init__(
        self,
        num_tags: int,
        include_start: bool = False,
        include_end: bool = False,
        padding_index: int = -1,
    ) -> None:
        super().__init__()

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.xavier_normal_(self.transitions)

        self.start_states: torch.Tensor | None
        self.end_states: torch.Tensor | None

        if include_start:
            self.start_states = nn.Parameter(torch.empty(num_tags))
            nn.init.normal_(self.start_states)
        else:
            self.start_states = None

        if include_end:
            self.end_states = nn.Parameter(torch.empty(num_tags))
            nn.init.normal_(self.end_states)
        else:
            self.end_states = None

        self.__padding_index = padding_index

    def forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_constraints: torch.Tensor | None = None,
        end_constraints: torch.Tensor | None = None,
        transition_constraints: torch.Tensor | None = None,
    ) -> BaseCrfDistribution:
        if mask is None:
            mask = logits.new_ones(logits.shape[:-1], dtype=torch.bool)

        batch_size, sequence_length, num_tags = logits.size()
        end_indices = mask.sum(dim=-1) - 1

        if self.start_states is not None:
            logits = logits.select_scatter(
                src=logits[:, 0] + self.start_states, dim=1, index=0
            )

        if self.end_states is not None:
            logits = logits.scatter_add(
                dim=1,
                index=end_indices[:, None, None].expand(-1, -1, num_tags),
                src=self.end_states[None, None, :].expand(batch_size, -1, -1),
            )

        # Apply constrains
        if start_constraints is not None:
            logits = logits.select_scatter(
                src=logits[:, 0].masked_fill(start_constraints, Semiring.zero),
                dim=1,
                index=0,
            )

        if end_constraints is not None:
            batch_indices = torch.arange(batch_size, device=logits.device)
            end_logits = logits[batch_indices, end_indices].masked_fill(
                end_constraints, Semiring.zero
            )
            logits = logits.scatter(
                dim=1,
                index=end_indices[:, None, None].expand(-1, -1, num_tags),
                src=end_logits[:, None, :],
            )

        if sequence_length == 1:
            return CrfUnitDistribution(logits=logits)

        if transition_constraints is not None:
            transitions = self.transitions.masked_fill(
                transition_constraints, Semiring.zero
            )
        else:
            transitions = self.transitions

        logits = logits * mask[..., None]  # + Semiring.one * ~mask[..., None]
        mask_expanded = mask[:, 1:, None, None]
        transitions = transitions[None, None] * mask_expanded  # + Semiring.one * ~mask
        potentials = Semiring.mul(logits[:, 1:, None, :], transitions)

        # Masking
        mask_value = Semiring.eye(n=num_tags, dtype=logits.dtype, device=logits.device)
        potentials = potentials * mask_expanded + mask_value * (~mask_expanded)

        return CrfDistribution(
            start_potentials=logits[:, 0],
            potentials=potentials,
            mask=mask,
            padding_index=self.__padding_index,
        )
