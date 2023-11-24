# sequence-classifier

`sequence-classifier` is an open-source library designed for sequence classification in PyTorch. It provides utilities for sequence classifiers, particularly Conditional random fields (CRFs), and it can calculate the log likelihood of tag sequences and retrieve the best label sequences. `sequence-classifier` also offers the capability to compute the marginal log likelihood and the marginal probability. These features are handy when working with partially annotated datasets.

If you are searching for libraries to handle sequence labeling tasks such as named-entity recognition or part-of-speech tagging combined with the use of foundation models like BERT, RoBERTa, or DeBERTa, you will find `sequence-classifier` to be helpful.

## Prerequisites

- Python 3.8+

## Installation

You can install sequence-classifier via pip:

```bash
pip install sequence-classifier
```

## Basic Usage

```python
import torch
from sequence_classifier.crf import Crf

batch_size = 3
sequence_length = 15
num_tags = 6

logits = torch.randn(batch_size, sequence_length, num_tags)
tag_indices = torch.randint(0, num_tags, (batch_size, sequence_length))

model = Crf(num_tags)

dist = model(logits)

nll_loss = dist.log_likelihood(tag_indices).sum().neg()
best_sequence = dist.argmax
```

## References

- Alexander Rush. 2020. [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/). In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations_, pages 335–342, Online. Association for Computational Linguistics.
- Yuta Tsuboi, Hisashi Kashima, Shinsuke Mori, Hiroki Oda, and Yuji Matsumoto. 2008. [Training Conditional Random Fields Using Incomplete Annotations](https://aclanthology.org/C08-1113/). In _Proceedings of the 22nd International Conference on Computational Linguistics (Coling 2008)_, pages 897–904, Manchester, UK. Coling 2008 Organizing Committee.
