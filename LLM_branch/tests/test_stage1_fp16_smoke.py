"""Short CUDA/FP16 optimizer smoke test for the Stage-1 training loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from transformers import TrainingArguments

from LLM_branch.train.train_SFT_xattn_new import LengthGroupedTrainer


class _RepeatedDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, index):
        del index
        return {
            "features": torch.tensor([0.25, -0.5, 0.75, 1.0]),
            "labels": torch.tensor([0.5]),
        }


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 1)

    def forward(self, features, labels=None):
        prediction = self.projection(features)
        loss = (prediction - labels).square().mean()
        return SimpleNamespace(loss=loss, logits=prediction)


class _TinyFP16Trainer(LengthGroupedTrainer):
    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        return self.optimizer

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def _collate(rows):
    return {
        key: torch.stack([row[key] for row in rows])
        for key in rows[0]
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v100_fp16_four_optimizer_steps(tmp_path):
    major, _ = torch.cuda.get_device_capability(0)
    if major != 7:
        pytest.skip("The locked smoke test targets a V100-class CUDA device")

    model = _TinyModel()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    stage1_trainer = _TinyFP16Trainer(
        model=model,
        args=args,
        train_dataset=_RepeatedDataset(),
        data_collator=_collate,
        family_sampling_power=0.0,
        xattn_diagnostic_steps=0,
    )

    result = stage1_trainer.train()

    assert result.global_step == 4
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
