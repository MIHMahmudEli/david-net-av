"""The floor detector has to fire on the real Kaggle readings and stay quiet on healthy
ones -- it is what stops a QACP run burning epochs the objective can no longer teach.
"""
import math

import pytest

from src.training.losses import at_loss_floor, supcon_floors, supcon_loss

import torch


def test_floors_match_the_analytic_values():
    assert supcon_floors(4) == pytest.approx((0.0, math.log(2), math.log(3)))
    assert supcon_floors(8) == pytest.approx((0.0, math.log(2), math.log(7)))
    # B=2 must not ask for log(1)=0 twice or log(0)
    assert all(f >= 0 for f in supcon_floors(2))


def test_fires_on_the_real_saturated_readings():
    """Kaggle epoch 12-14: every term parked, gnorm 0.00, seven epochs wasted."""
    assert at_loss_floor({"qacp_v": 0.6932, "qacp_a": 0.6932, "qacp_c": 1.0986}, 4)
    # epoch 9, the other degenerate mix seen in the same run
    assert at_loss_floor({"qacp_v": -0.0000, "qacp_a": 0.6932, "qacp_c": 1.0986}, 4)


def test_quiet_while_the_run_is_still_learning():
    """Kaggle epoch 0 -- all three terms away from any floor."""
    assert not at_loss_floor({"qacp_v": 1.1057, "qacp_a": 1.1195, "qacp_c": 1.0991}, 4)
    assert not at_loss_floor({"qacp_v": 0.3124, "qacp_a": 0.3285, "qacp_c": 0.9}, 4)


def test_one_live_term_is_enough_to_stay_quiet():
    """If any term still has something to learn the run should continue."""
    assert not at_loss_floor({"qacp_v": 0.6932, "qacp_a": 0.3000, "qacp_c": 1.0986}, 4)


def test_floor_is_batch_size_aware():
    """1.0986 is a floor at B=4 but an ordinary value at B=8, so the guard must not
    fire on it there -- otherwise raising batch_size would trip its own alarm."""
    reading = {"qacp_v": 0.0, "qacp_a": 0.6932, "qacp_c": 1.0986}
    assert at_loss_floor(reading, 4)
    assert not at_loss_floor(reading, 8)


def test_supcon_actually_bottoms_out_where_supcon_floors_says():
    """Ground the constants in the real loss rather than in arithmetic."""
    torch.manual_seed(0)
    for B in (4, 8):
        emb = torch.nn.Parameter(torch.randn(B, 32))
        opt = torch.optim.Adam([emb], lr=5e-2)
        labels = torch.zeros(B, dtype=torch.long)      # no negatives at all
        for _ in range(800):
            loss = supcon_loss(emb, labels, temperature=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert float(loss.detach()) == pytest.approx(math.log(B - 1), abs=1e-3), B
