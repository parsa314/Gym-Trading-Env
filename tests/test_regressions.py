import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from gym_trading_env.environments import TradingEnv, MultiDatasetTradingEnv


def prices(value=100):
    return pd.DataFrame(
        {"open": value, "high": value + 10, "low": value - 10,
         "close": value, "feature_price": np.arange(100, dtype=float)},
        index=pd.date_range("2024-01-01", periods=100, freq="4h"),
    )


class RegressionTests(unittest.TestCase):
    def test_seed_repeats_start_and_position(self):
        env = TradingEnv(prices(), positions=[-1, 0, 1], max_episode_duration=20, verbose=0)
        first, info = env.reset(seed=42)
        np.random.seed(999)
        np.random.random(50)
        second, again = env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(info["idx"], again["idx"])
        self.assertEqual(info["position"], again["position"])

    def test_filled_nonpersistent_order_is_removed(self):
        env = TradingEnv(prices(), initial_position=0, verbose=0)
        env.reset()
        env.add_limit_order(1, limit=100)
        env.step(0)
        self.assertEqual(env._position, 1)
        self.assertEqual(env._limit_orders, {})

    def test_persistent_order_remains(self):
        env = TradingEnv(prices(), initial_position=0, verbose=0)
        env.reset()
        env.add_limit_order(1, limit=100, persistent=True)
        env.step(0)
        self.assertIn(1, env._limit_orders)

    def test_dataset_selection_uses_least_used_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(3):
                prices(100+i).to_pickle(Path(folder) / f"{i}.pkl")
            env = MultiDatasetTradingEnv(folder+"/*.pkl", verbose=0)
            env.dataset_nb_uses[:] = [5, 5, 0]
            chosen = env.next_dataset()
            self.assertEqual(chosen["close"].iloc[0], 102)
            np.testing.assert_array_equal(env.dataset_nb_uses, [5, 5, 1])
            env.reset(seed=42, options={})

    def test_missing_dataset_has_clear_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                MultiDatasetTradingEnv(folder+"/*.pkl")


if __name__ == "__main__":
    unittest.main()
