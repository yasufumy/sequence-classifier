import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from sequence_label import LabelSet

from sequence_classifier.crf import Crf


@pytest.fixture(scope="session")
def label_set() -> LabelSet:
    return LabelSet({"ORG", "PER"})


def test_crf() -> None:
    logits1 = torch.randn(3, 15, 5)
    logits2 = torch.randn(3, 1, 5)

    crf = Crf(num_tags=5)

    dist1 = crf(logits1)
    assert dist1.log_partitions.value.size() == torch.Size([3])

    dist2 = crf(logits2)
    assert dist2.log_partitions.value.size() == torch.Size([3])


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
