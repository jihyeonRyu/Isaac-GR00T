import types

from gr00t.experiment.trainer import TrainableModelEmaCallback
import pytest
import torch


def test_ema_updates_trainable_parameters_and_restores_raw_weights():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model[1].weight.requires_grad_(False)
    callback = TrainableModelEmaCallback(decay=0.5)
    state = types.SimpleNamespace(global_step=0)

    callback.on_train_begin(None, state, None, model=model)
    raw_before = [p.detach().clone() for p in callback.params]
    with torch.no_grad():
        for param in callback.params:
            param.add_(2.0)
    raw_after = [p.detach().clone() for p in callback.params]

    state.global_step = 1
    callback.on_step_end(None, state, None)
    for shadow, before in zip(callback.shadows, raw_before):
        assert torch.allclose(shadow, before.float() + 1.0)

    callback.swap_in()
    for param, before in zip(callback.params, raw_before):
        assert torch.allclose(param.float(), before.float() + 1.0)
    callback.restore()
    for param, after in zip(callback.params, raw_after):
        assert torch.equal(param, after)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ema_decay": 1.0}, "ema_decay"),
        ({"ema_update_after_step": -1}, "ema_update_after_step"),
        ({"ema_update_every": 0}, "ema_update_every"),
    ],
)
def test_training_config_rejects_invalid_ema_settings(kwargs, message):
    from gr00t.configs.training.training_config import TrainingConfig

    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)
