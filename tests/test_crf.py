import torch

from sequence_classifier.crf import Crf


def test_crf() -> None:
    logits1 = torch.randn(3, 15, 5)
    logits2 = torch.randn(3, 1, 5)

    crf = Crf(num_tags=5)

    dist1 = crf(logits1)
    assert dist1.log_partitions.value.size() == torch.Size([3])

    dist2 = crf(logits2)
    assert dist2.log_partitions.value.size() == torch.Size([3])
