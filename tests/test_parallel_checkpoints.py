"""DataParallel must not change what a checkpoint looks like.

`nn.DataParallel` prefixes every state_dict key with `module.`. If a checkpoint written by
a 2-GPU run cannot be read by a 1-GPU run (or vice versa), `init_from` reports a wall of
unexpected keys -- or worse, `strict=False` swallows it and Stage 1 trains from noise while
printing a reassuring "0 missing, 0 unexpected". Every artefact already on HF (QACP v3/v4)
was written unwrapped, so unwrapped is the format these tests pin down.
"""
import torch
import torch.nn as nn

from src.utils.parallel import maybe_parallel, parallel_batch_warning, unwrap


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.head = nn.Linear(3, 2)

    def forward(self, x):
        return self.head(torch.relu(self.fc(x)))


def test_unwrap_is_identity_without_a_wrapper():
    m = Tiny()
    assert unwrap(m) is m


def test_unwrap_returns_the_real_module_when_wrapped():
    m = Tiny()
    wrapped = nn.DataParallel(m)
    assert unwrap(wrapped) is m


def test_wrapped_save_has_no_module_prefix():
    """The whole point: a 2-GPU run writes the same keys a 1-GPU run does."""
    m = Tiny()
    wrapped = nn.DataParallel(m)
    assert any(k.startswith("module.") for k in wrapped.state_dict()), \
        "sanity: DataParallel really does add the prefix"
    saved = unwrap(wrapped).state_dict()
    assert not any(k.startswith("module.") for k in saved)
    assert set(saved) == set(Tiny().state_dict())


def test_checkpoint_round_trips_wrapped_to_unwrapped():
    src = Tiny()
    with torch.no_grad():
        src.fc.weight.fill_(0.5)
    ckpt = unwrap(nn.DataParallel(src)).state_dict()

    dst = Tiny()
    missing, unexpected = dst.load_state_dict(ckpt, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    assert torch.allclose(dst.fc.weight, torch.full_like(dst.fc.weight, 0.5))


def test_checkpoint_round_trips_unwrapped_to_wrapped():
    """The direction that matters for init_from: QACP's weights were saved on 1 GPU."""
    src = Tiny()
    with torch.no_grad():
        src.head.bias.fill_(-1.25)
    ckpt = src.state_dict()

    dst = nn.DataParallel(Tiny())
    missing, unexpected = unwrap(dst).load_state_dict(ckpt, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    assert torch.allclose(unwrap(dst).head.bias, torch.full_like(src.head.bias, -1.25))


def test_loading_a_wrapped_dict_into_a_bare_model_would_have_failed():
    """Documents the bug being prevented, so nobody 'simplifies' unwrap away."""
    wrapped_keys = nn.DataParallel(Tiny()).state_dict()
    missing, unexpected = Tiny().load_state_dict(wrapped_keys, strict=False)
    assert missing and unexpected, "if this ever passes cleanly, unwrap is unnecessary"


def test_maybe_parallel_is_a_noop_without_multiple_gpus():
    m = Tiny()
    assert maybe_parallel(m, device="cpu") is m
    assert maybe_parallel(m, device="cuda", enabled=False) is m


def test_batch_warning_is_silent_for_an_unwrapped_model(capsys):
    parallel_batch_warning(4, Tiny())
    assert capsys.readouterr().out == ""
