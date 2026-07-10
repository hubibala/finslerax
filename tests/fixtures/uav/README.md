# UAV test fixtures

Frozen inputs so `tests/test_uav.py` runs offline and deterministic.

## What lives here

- **`real_segments.csv`** — preprocessed segment tables of **5 real PX4 flights**
  (3 physical vehicles), produced by `experiments/uav/ingest.py` (`REAL_CFG`)
  from the 2026-07 corpus pull and frozen for `test_u4_real_fixture_ledger`:
  the Stage-A gates on real data (every flight ledger-conditioned, climb costs
  more in all, deterministic k̂ reproduced to rtol 1e-3). CSV, not parquet, so
  the suite stays hermetic without `pyarrow`.
- The U0/U1 ingest rungs additionally drive the pipeline with **synthetic
  PX4-convention raw flights** generated in-process by `simulate_raw_flight`
  (seeded — no files needed).

## Attribution (CC-BY)

The fixture flights come from the public **PX4 Flight Review** database
(<https://review.px4.io>), licensed **CC-BY**. Original logs (uploader
attribution available on each page):

| log_id | vehicle (sys_uuid prefix) |
|---|---|
| [04fc37c2-aa7c-4e3f-aa13-3cfe93114329](https://review.px4.io/plot_app?log=04fc37c2-aa7c-4e3f-aa13-3cfe93114329) | 003D003E34345117… |
| [0f773ce7-2ae1-4b1b-a68e-2ad53ee25795](https://review.px4.io/plot_app?log=0f773ce7-2ae1-4b1b-a68e-2ad53ee25795) | 003D003E34345117… |
| [0193dffe-5962-4b8b-829a-efd9dfad1fd4](https://review.px4.io/plot_app?log=0193dffe-5962-4b8b-829a-efd9dfad1fd4) | 0040001C31375105… |
| [02f37fcd-0bdf-4701-ac7e-ff26698c91f8](https://review.px4.io/plot_app?log=02f37fcd-0bdf-4701-ac7e-ff26698c91f8) | 0040001C31375105… |
| [12a68b66-dc0b-409f-b40c-b8d46e4b2eba](https://review.px4.io/plot_app?log=12a68b66-dc0b-409f-b40c-b8d46e4b2eba) | 00390038313751055… |

Only derived per-segment aggregates (displacement, duration, ∫V·I dt energy,
segment-mean wind) are committed — no raw telemetry.
