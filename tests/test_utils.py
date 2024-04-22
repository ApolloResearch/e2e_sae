import pytest

from e2e_sae.utils import get_linear_lr_schedule


class TestLinearLRSchedule:
    @pytest.mark.parametrize(
        "warmup_samples, cooldown_samples, n_samples, effective_batch_size, expected_error",
        [
            (100, 50, None, 10, AssertionError),  # Cooldown requested without setting n_samples
            (100, 100, 150, 10, AssertionError),  # Cooldown starts before warmup ends
        ],
    )
    def test_value_errors(
        self,
        warmup_samples: int,
        cooldown_samples: int,
        n_samples: int | None,
        effective_batch_size: int,
        expected_error: type[BaseException],
    ):
        with pytest.raises(expected_error):
            get_linear_lr_schedule(
                warmup_samples, cooldown_samples, n_samples, effective_batch_size
            )

    def test_constant_lr(self):
        lr_schedule = get_linear_lr_schedule(
            warmup_samples=0, cooldown_samples=0, n_samples=None, effective_batch_size=10
        )
        for step in range(0, 100):
            assert lr_schedule(step) == 1.0, "Learning rate should be constant at 1.0"

    @pytest.mark.parametrize(
        "warmup_samples, cooldown_samples, n_samples, effective_batch_size, step, expected_lr",
        [
            (100, 0, None, 10, 5, 0.6),  # During warmup
            (100, 200, 1000, 10, 50, 1.0),  # After warmup, before cooldown
            (100, 100, 1000, 10, 92, 0.9),  # During cooldown
            (100, 100, 1000, 10, 101, 0.0),  # After cooldown
        ],
    )
    def test_learning_rate_transitions(
        self,
        warmup_samples: int,
        cooldown_samples: int,
        n_samples: int,
        effective_batch_size: int,
        step: int,
        expected_lr: float,
    ):
        # Note that a `step` corresponds to the upcoming sample.
        lr_schedule = get_linear_lr_schedule(
            warmup_samples, cooldown_samples, n_samples, effective_batch_size
        )
        assert lr_schedule(step) == pytest.approx(
            expected_lr
        ), f"Learning rate at step {step} should be {expected_lr}"
