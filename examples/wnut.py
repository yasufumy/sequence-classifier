# Most of the code here is borrowed from the following page:
# https://huggingface.co/docs/transformers/tasks/token_classification

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Any, Callable, Final, Literal, cast

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sequence_classifier.crf import Crf

if TYPE_CHECKING:
    from datasets.formatting.formatting import LazyBatch
    from transformers import (
        BatchEncoding,
        EvalPrediction,
        PretrainedConfig,
        PreTrainedTokenizerFast,
    )
    from transformers.modeling_outputs import TokenClassifierOutput


IGNORE_INDEX: Final[int] = -100


class SequenceClassifier(PreTrainedModel):  # type:ignore
    config_class = AutoConfig
    base_model_prefix = "foundation_model"
    supports_gradient_checkpointing = True

    def __init__(self, config: PretrainedConfig, *inputs: Any, **kwargs: Any):
        super().__init__(config, *inputs, **kwargs)

        self.foundation_model = AutoModelForTokenClassification.from_config(config)

        self.crf = Crf(
            num_tags=self.foundation_model.classifier.out_features, padding_index=0
        )

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> TokenClassifierOutput:
        outputs = self.foundation_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        mask = cast(torch.BoolTensor, labels != IGNORE_INDEX)

        dist = self.crf(logits=outputs.logits, mask=mask)

        loss = None
        if labels is not None:
            loss = dist.log_likelihood(labels).mean().neg()

        logits = nn.functional.one_hot(dist.argmax, num_classes=outputs.logits.size(-1))
        logits *= mask[..., None]

        return replace(outputs, loss=loss, logits=logits)


ClassifierType = Literal["token", "sequence"]


@dataclass(frozen=True)
class HuggingFaceRepository:
    model_name: str

    def load_tokenizer(self, **tokenizer_args: Any) -> PreTrainedTokenizerFast:
        return AutoTokenizer.from_pretrained(self.model_name, **tokenizer_args)

    def load_classifier(
        self, classifier_type: ClassifierType, **model_args: Any
    ) -> AutoModelForTokenClassification | SequenceClassifier:
        match classifier_type:
            case "token":
                return AutoModelForTokenClassification.from_pretrained(
                    self.model_name, **model_args
                )
            case "sequence":
                return SequenceClassifier.from_pretrained(self.model_name, **model_args)
            case _:
                raise ValueError(
                    f"An invalid classifier type is specified: {classifier_type}"
                )


@dataclass(frozen=True)
class TrainingJob:
    hf_repo: HuggingFaceRepository
    tokenizer_args: dict[str, Any]
    classifier_type: ClassifierType
    output_directory: str
    metric_file: str = "metrics.json"
    seed: int = 0

    @staticmethod
    def __get_preprocessor(
        tokenizer: PreTrainedTokenizerFast,
    ) -> Callable[[LazyBatch], BatchEncoding]:
        def tokenize_and_align_labels(examples: LazyBatch) -> BatchEncoding:
            tokenized_inputs = tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )

            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(
                    batch_index=i
                )  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(IGNORE_INDEX)
                    elif (
                        word_idx != previous_word_idx
                    ):  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(IGNORE_INDEX)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        return tokenize_and_align_labels

    @staticmethod
    def __get_compute_metrics(
        id2label: dict[int, str]
    ) -> Callable[[EvalPrediction], dict[str, float]]:
        seqeval = evaluate.load("seqeval")

        def compute_metrics(p: EvalPrediction) -> dict[str, float]:
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [id2label[x] for (x, y) in zip(prediction, label) if y != IGNORE_INDEX]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [id2label[y] for (x, y) in zip(prediction, label) if y != IGNORE_INDEX]
                for prediction, label in zip(predictions, labels)
            ]

            results = seqeval.compute(
                predictions=true_predictions, references=true_labels
            )
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1_score": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        return compute_metrics

    def __save_metrics(self, metrics: dict[str, float]) -> None:
        with open(f"{self.output_directory}/{self.metric_file}", "w") as f:
            json.dump(metrics, f, indent=2)

    def start(self) -> None:
        set_seed(self.seed)

        dataset = load_dataset("wnut_17")

        label_list = dataset["train"].features["ner_tags"].feature.names
        id2label = dict(enumerate(label_list))
        label2id = {label: i for i, label in id2label.items()}

        tokenizer = self.hf_repo.load_tokenizer(**self.tokenizer_args)

        tokenized_dataset = dataset.map(
            self.__get_preprocessor(tokenizer=tokenizer), batched=True
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        model = self.hf_repo.load_classifier(
            classifier_type=self.classifier_type,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )

        training_args = TrainingArguments(
            output_dir=self.output_directory,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=20,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            greater_is_better=True,
            metric_for_best_model="eval_f1_score",
            load_best_model_at_end=True,
            save_total_limit=1,
            seed=self.seed,
            data_seed=self.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.__get_compute_metrics(id2label=id2label),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()

        metrics = trainer.evaluate(
            eval_dataset=tokenized_dataset["test"], metric_key_prefix="test"
        )
        self.__save_metrics(metrics)


def execute(job: TrainingJob, available_devices: Queue[int]) -> None:
    device = available_devices.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    job.start()
    available_devices.put(device)


@dataclass(frozen=True)
class JobRunner:
    devices: tuple[int, ...]
    jobs: tuple[TrainingJob, ...]

    def __call__(self) -> None:
        available_devices: Queue[int] = Queue()
        for device in self.devices:
            available_devices.put(device)

        processes = []
        for job in self.jobs:
            process = Process(target=execute, args=(job, available_devices))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
            process.close()

        available_devices.close()


if __name__ == "__main__":
    hf_repo = HuggingFaceRepository(model_name="distilroberta-base")
    tokenizer_args = {"add_prefix_space": True}
    jobs = []
    for classifier_type in ("token", "sequence"):
        jobs.append(
            TrainingJob(
                hf_repo=hf_repo,
                tokenizer_args=tokenizer_args,
                classifier_type=cast(ClassifierType, classifier_type),
                output_directory=f"output-{classifier_type}",
            )
        )

    runner = JobRunner(devices=(0,), jobs=tuple(jobs))
    runner()
