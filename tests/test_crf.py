from functools import reduce
from typing import cast

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from sequence_label import LabelSet
from torch.nn import functional as F
from torch.testing import assert_close
from torch_struct import LinearChainCRF
from torchcrf import CRF

from sequence_classifier.crf import BaseCrfDistribution, Crf


@pytest.fixture(scope="session")
def label_set() -> LabelSet:
    return LabelSet({"ORG", "PER"})


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    sequence_length=st.integers(min_value=2, max_value=15),
    num_tags=st.integers(min_value=2, max_value=7),
)
def test_crf_maches_torch_struct_results(
    batch_size: int, sequence_length: int, num_tags: int
) -> None:
    logits = torch.randn(batch_size, sequence_length, num_tags, requires_grad=True)
    lengths = torch.arange(sequence_length, sequence_length - batch_size, -1)
    lengths = torch.maximum(lengths, torch.full((batch_size,), 2))
    mask = torch.arange(sequence_length) < lengths[..., None]
    tag_indices = torch.randint(0, num_tags, (batch_size, sequence_length))

    crf = Crf(num_tags=num_tags, padding_index=0)

    dist = cast(BaseCrfDistribution, crf(logits, mask))

    log_potentials = logits[:, 1:, ..., None] + crf.transitions.T[None, None]
    log_potentials[:, 0] += logits[:, [0]]
    expected = LinearChainCRF(log_potentials.contiguous(), lengths)

    assert_close(dist.log_partitions.value, expected.partition)
    assert_close(dist.marginals, expected.marginals.transpose(3, 2))
    assert_close(dist.argmax, expected.from_event(expected.argmax)[0])

    # lengths don't take into account in a log_prob method
    dist2 = cast(BaseCrfDistribution, crf(logits))
    assert_close(
        dist2.log_scores(tag_indices) - dist.log_partitions.value,
        expected.log_prob(expected.to_event(tag_indices, num_tags)),
    )
    assert_close(
        dist2.log_multitag_scores(F.one_hot(tag_indices, num_tags).bool())
        - dist.log_partitions.value,
        expected.log_prob(expected.to_event(tag_indices, num_tags)),
    )


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    sequence_length=st.integers(min_value=1, max_value=15),
    num_tags=st.integers(min_value=2, max_value=7),
)
def test_crf_maches_pytorch_crf_results(
    batch_size: int, sequence_length: int, num_tags: int
) -> None:
    logits = torch.randn(batch_size, sequence_length, num_tags, requires_grad=True)
    lengths = torch.arange(sequence_length, sequence_length - batch_size, -1)
    lengths = torch.maximum(lengths, torch.full((batch_size,), 2))
    mask = torch.arange(sequence_length) < lengths[..., None]
    tag_indices = torch.randint(0, num_tags, (batch_size, sequence_length))

    target = CRF(num_tags=num_tags, batch_first=True)

    crf = Crf(num_tags=num_tags, include_start=True, include_end=True)
    crf.transitions.data = target.transitions.data
    cast(torch.Tensor, crf.start_states).data = target.start_transitions.data
    cast(torch.Tensor, crf.end_states).data = target.end_transitions.data

    dist = crf(logits=logits, mask=mask)

    actual = dist.log_likelihood(tag_indices)
    expected = target(emissions=logits, tags=tag_indices, mask=mask, reduction="none")

    assert_close(actual, expected)

    assert dist.argmax.masked_select(mask).tolist() == reduce(
        lambda a, b: a + b, target.decode(emissions=logits, mask=mask)
    )

    # Check gradients
    actual.sum().neg().backward()
    expected.sum().neg().backward()

    assert_close(
        cast(torch.Tensor, crf.start_states).grad, target.start_transitions.grad
    )
    assert_close(cast(torch.Tensor, crf.end_states).grad, target.end_transitions.grad)
    if target.transitions.grad is not None:
        assert_close(crf.transitions.grad, target.transitions.grad)


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    sequence_length=st.integers(min_value=1, max_value=15),
)
def test_constraints(
    label_set: LabelSet, batch_size: int, sequence_length: int
) -> None:
    crf = Crf(num_tags=label_set.state_size)

    start_constraints = ~torch.tensor(label_set.start_states)
    end_constraints = ~torch.tensor(label_set.end_states)
    transition_constraints = ~torch.tensor(label_set.transitions)

    logits = torch.randn(batch_size, sequence_length, label_set.state_size)

    dist = crf(
        logits,
        start_constraints=start_constraints,
        end_constraints=end_constraints,
        transition_constraints=transition_constraints,
    )

    tag_indices = dist.argmax

    label_set.decode(tag_indices.tolist())
