"""Stage 5 severity-margin DPO loss tests."""

from __future__ import annotations

import unittest

import torch

from hsa_dpo.trainer.base_dpo_trainer import BaseDPOTrainer


def _trainer(*, loss_type: str, margin_scale: float = 0.5) -> BaseDPOTrainer:
    trainer = BaseDPOTrainer.__new__(BaseDPOTrainer)
    trainer.beta = 0.1
    trainer.use_chosen_score = False
    trainer.use_rejected_score = True
    trainer.dpo_loss_type = loss_type
    trainer.severity_margin_scale = margin_scale
    trainer.severity_score_normalizer = 3.0
    return trainer


class SeverityMarginLossTests(unittest.TestCase):
    def test_zero_margin_matches_standard_dpo(self) -> None:
        policy_chosen = torch.tensor([3.0])
        policy_rejected = torch.tensor([1.0])
        ref_chosen = torch.tensor([2.0])
        ref_rejected = torch.tensor([1.5])
        chosen_scores = torch.tensor([1.0])
        rejected_scores = torch.tensor([3.0])

        standard_loss, _, _ = _trainer(loss_type="standard").dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            chosen_scores,
            rejected_scores,
        )
        margin_loss, _, _ = _trainer(loss_type="severity_margin", margin_scale=0.0).dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            chosen_scores,
            rejected_scores,
        )

        self.assertTrue(torch.allclose(standard_loss, margin_loss))

    def test_higher_severity_increases_loss_for_same_logits(self) -> None:
        policy_chosen = torch.tensor([3.0, 3.0])
        policy_rejected = torch.tensor([1.0, 1.0])
        ref_chosen = torch.tensor([2.0, 2.0])
        ref_rejected = torch.tensor([1.5, 1.5])
        chosen_scores = torch.tensor([1.0, 1.0])
        rejected_scores = torch.tensor([1.0, 3.0])

        losses, _, _ = _trainer(loss_type="severity_margin", margin_scale=0.5).dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            chosen_scores,
            rejected_scores,
        )

        self.assertGreater(losses[1].item(), losses[0].item())

    def test_hsa_weighted_keeps_rejected_score_weighting(self) -> None:
        policy_chosen = torch.tensor([3.0])
        policy_rejected = torch.tensor([1.0])
        ref_chosen = torch.tensor([2.0])
        ref_rejected = torch.tensor([1.5])
        chosen_scores = torch.tensor([1.0])
        rejected_scores = torch.tensor([3.0])

        weighted_loss, _, _ = _trainer(loss_type="hsa_weighted").dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            chosen_scores,
            rejected_scores,
        )
        standard_loss, _, _ = _trainer(loss_type="standard").dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            chosen_scores,
            rejected_scores,
        )

        self.assertFalse(torch.allclose(weighted_loss, standard_loss))


if __name__ == "__main__":
    unittest.main()
