# Research environment reliability review

This fork supports the simulation component of the separate modular-crypto-trading-bot project. Original project: https://github.com/ClementPerroud/Gym-Trading-Env. Original authorship and license remain unchanged.

## Upstream review

Compared fork main with upstream main at a86fc03d29759374fae2db7df4d878485808a730. The fork was six commits behind and zero ahead before this work.

- aba29e7 / e156771: correct least-used dataset indexing. Incorporated the same correction here, using the environment RNG.
- 77d8c0f / f7e167: downloader event-loop changes, registration changes, example updates, missing-data guard, reset options forwarding. Incorporated the missing-data guard and reset options support. Other changes need separate compatibility review; this branch is not a full upstream synchronization.
- 311495f / a86fc03: documentation typo correction. Deferred as unrelated to these behavior fixes.

## Changes

- TradingEnv uses its Gymnasium RNG for random initial positions and episode starting indices.
- MultiDatasetTradingEnv sorts dataset paths, uses its RNG, correctly maps least-used candidate indices, and accepts reset options. An explicit reset seed initializes its RNG before dataset selection.
- Filled nonpersistent orders are removed while iterating a snapshot, avoiding dictionary-size mutation errors.

## Validation

Five unittest regression tests passed locally with Gymnasium 0.28.1:

1. Repeating a seed reproduces the single-dataset initial observation, index and position despite unrelated global NumPy draws.
2. Filled nonpersistent limit orders execute and are removed.
3. Persistent orders remain registered.
4. The unique least-used dataset is selected and its count updated; reset accepts options.
5. An empty dataset glob raises FileNotFoundError.

Run from the repository root after installing project dependencies:

    PYTHONPATH=src python -m unittest discover -s tests -v

## Scope and follow-up

This is a focused regression suite, not a financial-performance validation or a full compatibility matrix. No live exchange calls or orders were made. Multi-dataset replay also depends on accumulated dataset-use counts and episode counters; reset(seed) does not clear them.

Before research experiments: validate execution timing and slippage assumptions, borrowing-interest accounting, terminal rewards, episode boundary handling, and dynamic feature history across resets. Review README/package version discrepancies. Add chronological train/validation/test splits, Ichimoku features without future-data leakage, baseline comparisons and recorded experiment configurations in the main bot repository. No profitability results are claimed.
