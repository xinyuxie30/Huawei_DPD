import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "steps" / "aclr_active_bin_table_optimization.py"
SPEC = importlib.util.spec_from_file_location("aclr_active_bin_table_optimization", SCRIPT)
ACTIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTIVE)
TOPOLOGY_SCRIPT = PROJECT_ROOT / "steps" / "aclr_lag_topology_search.py"
TOPOLOGY_SPEC = importlib.util.spec_from_file_location("aclr_lag_topology_search", TOPOLOGY_SCRIPT)
TOPOLOGY = importlib.util.module_from_spec(TOPOLOGY_SPEC)
TOPOLOGY_SPEC.loader.exec_module(TOPOLOGY)


class ActiveBinOptimizerTest(unittest.TestCase):
    def test_grid_24_to_48_warm_start_preserves_old_knots(self):
        generator = torch.Generator().manual_seed(7)
        tables = torch.randn(5, 2, 25, generator=generator, dtype=torch.float64)
        state = {
            "corr.tables": tables,
            "unchanged": torch.tensor([1.0, 2.0], dtype=torch.float64),
        }

        resized = ACTIVE.resample_table_grid(state, 48)

        self.assertEqual(tuple(resized["corr.tables"].shape), (5, 2, 49))
        torch.testing.assert_close(resized["corr.tables"][..., ::2], tables, rtol=0.0, atol=0.0)
        torch.testing.assert_close(resized["unchanged"], state["unchanged"], rtol=0.0, atol=0.0)

    def test_torch_welch_matches_scipy_complex_defaults(self):
        rng = np.random.default_rng(20260714)
        fs = 491.52
        for nperseg, length in [(9, 45), (4096, 12288)]:
            with self.subTest(nperseg=nperseg):
                signal = rng.standard_normal(length) + 1j * rng.standard_normal(length)
                helper = ACTIVE.TorchWelch(nperseg, fs, torch.device("cpu"))
                tensor = torch.as_tensor(signal, dtype=torch.complex128).unsqueeze(0)

                actual = helper.power(tensor).numpy()
                freq, expected = welch(
                    signal,
                    fs=fs,
                    nperseg=nperseg,
                    return_onesided=False,
                )
                scale = 1.0 / (fs * np.sum(np.hanning(nperseg + 1)[:-1] ** 2))

                np.testing.assert_allclose(helper.freq_mhz, freq, rtol=0.0, atol=1e-15)
                np.testing.assert_allclose(actual * scale, expected, rtol=2e-12, atol=1e-15)

    def test_lag_remap_copies_semantic_rows_and_zeros_new_lags(self):
        source_offsets = [-2, -1, 0, 1, 2]
        target_offsets = [-2, -1, 0, 2, 3]
        tables = torch.stack(
            [torch.full((2, 4), float(row), dtype=torch.float64) for row in range(9)]
        )
        state = {
            "base.in_conv.weight": torch.zeros(1, dtype=torch.float64),
            "corr.tables": tables,
            "corr.gmin": -torch.arange(1, 10, dtype=torch.float64),
            "corr.gmax": torch.arange(1, 10, dtype=torch.float64),
            "fir_real": torch.zeros(3, dtype=torch.float64),
            "fir_imag": torch.zeros(3, dtype=torch.float64),
        }

        remapped, copied, initialized = TOPOLOGY.remap_table_rows(
            state, source_offsets, target_offsets
        )

        # amplitude, -2 and -1 stay in place; source +2 rows 7/8 move to 5/6.
        for target_row, source_row in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 7), (6, 8)]:
            torch.testing.assert_close(remapped["corr.tables"][target_row], tables[source_row])
            self.assertEqual(remapped["corr.gmin"][target_row], state["corr.gmin"][source_row])
            self.assertEqual(remapped["corr.gmax"][target_row], state["corr.gmax"][source_row])
        torch.testing.assert_close(remapped["corr.tables"][7:], torch.zeros_like(tables[7:]))
        self.assertEqual(set(initialized), {("real", 3), ("imag", 3)})
        self.assertIn(("amplitude", 0), copied)


if __name__ == "__main__":
    unittest.main()
