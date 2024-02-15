import pytest
import torch

from sparsify.metrics import calc_batch_dict_el_frequencies, update_dict_el_frequencies


class TestFrequencyCalculations:
    @pytest.mark.parametrize(
        "c, expected_frequency",
        [
            (torch.tensor([[1, 0], [1, 1]], dtype=torch.float), [1.0, 0.5]),  # No pos
            (torch.tensor([[1, 1], [0, 0], [1, 0]], dtype=torch.float), [2 / 3, 1 / 3]),  # No pos
            (
                torch.tensor([[[1, 0], [1, 1]], [[1, 0], [0, 0]]], dtype=torch.float),
                [3 / 4, 1 / 4],
            ),  # 2 batches, 2 pos, 2 dict els
            (
                torch.tensor(
                    [[[1, 1], [0, 0], [1, 0]], [[1, 0], [1, 1], [0, 0]]], dtype=torch.float
                ),
                [4 / 6, 2 / 6],
            ),  # 2 batches, 3 pos, 2 dict els
        ],
    )
    def test_calc_batch_dict_el_frequencies(
        self, c: torch.Tensor, expected_frequency: float
    ) -> None:
        frequency = calc_batch_dict_el_frequencies({"hook1": {"c": c}})["hook1"]
        assert torch.allclose(torch.tensor(frequency), torch.tensor(expected_frequency))


class TestUpdateFrequencies:
    @pytest.mark.parametrize(
        "initial_frequencies, batch_frequencies, total_tokens, batch_tokens, expected_frequencies",
        [
            # Case 1: Initial frequencies are empty
            ({}, {"hook1": [1.0, 0.5]}, 0, 2, {"hook1": [1.0, 0.5]}),
            # Case 2: Updating existing frequencies
            ({"hook1": [0.5, 0.25]}, {"hook1": [1.0, 0.75]}, 4, 2, {"hook1": [4 / 6, 2.5 / 6]}),
            # Case 3: Multiple hooks
            (
                {"hook1": [0.5, 0.25], "hook2": [0.3, 0.7]},
                {"hook1": [1.0, 0.5], "hook2": [0.4, 0.6]},
                2,
                1,
                {"hook1": [2 / 3, 1 / 3], "hook2": [1 / 3, 2 / 3]},
            ),
        ],
    )
    def test_update_dict_el_frequencies(
        self,
        initial_frequencies: dict[str, list[float]],
        batch_frequencies: dict[str, list[float]],
        total_tokens: int,
        batch_tokens: int,
        expected_frequencies: dict[str, list[float]],
    ):
        update_dict_el_frequencies(
            initial_frequencies, batch_frequencies, total_tokens, batch_tokens
        )
        for hook_name, freqs in expected_frequencies.items():
            assert torch.allclose(
                torch.tensor(initial_frequencies[hook_name]), torch.tensor(freqs)
            ), f"Mismatch in frequencies for {hook_name}"
