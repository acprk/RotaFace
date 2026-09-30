# RotaFace

Key rotation for TFHE-encrypted biometric databases.

RotaFace lets a cloud server re-key a TFHE-encrypted 1:N face gallery in place, from a small per-epoch token, without
decrypting it. The repository contains the Rust implementation (on the `core_crypto` layer of TFHE-rs 0.6.4), the
experiment drivers, the analysis/plotting scripts and the raw CSV/log outputs of all experiments.

## Build

```bash
cd code && cargo build --release        # Rust >= 1.75; binaries in code/target/release/
python3 -m venv .venv && .venv/bin/pip install numpy scipy scikit-learn matplotlib
```
Embeddings are not redistributed; see `data/README.md`. All binaries read `RF_DATA` (default `data/lfw_emb.f32`).
Parameter set: `RF_PARAMS=p128` (k=1, N=2048, q=2^64, sigma=3.2*2^24; main set) or unset (legacy k=3, N=512).

## Main building blocks (`code/src/lib.rs`)
| function | role |
|---|---|
| `enc_template_b`, `enc_query_b`, `pack`, `twist`, `coef_score` | slot-packed GLWE templates, GGSW queries, matching |
| `rotate_ggsw`, `ggsw_row_noise` | GGSW-at-rest layout (comparison) |
| `ksk_gen_seeded`, `NativeKsk::expand`, `native_ks`, `truncate_split` | seeded mask-only key switch, gadget-aligned storage |
| `pk_gen`, `rerandomise` | public-key re-randomisation |
| `pad_key`, `upgrade_glwe`, `upgrade_ggsw` | in-place upgrade to a larger GLWE dimension |
| `owner_reencrypt_fft` | owner-side decrypt + re-encrypt baseline |

## Experiments
Single-core timings are sensitive to host load; run them on an idle machine.

| experiment | command (from repo root, `B=code/target/release`, `export RF_PARAMS=p128`) | output |
|---|---|---|
| parameter security | `sage results/e0_security/est.sage`, `est_full.sage` (set `LATTICE_ESTIMATOR`) | `results/e0_security/*.log` |
| 1:N accuracy along the rotation chain | `RF_RERAND=1 RF_COMPACT=1 $B/ident B 16 2 4 9 0,1024 24 <prefix>` then `python results/analyze_e2.py <dir> <out.csv>` | `results/p128/e2_*metrics.csv` |
| noise model vs. measurement | `$B/chain_compact <logB> <l> <T> 16 16 out.csv`; `python results/noise_model_compact.py` | `results/p128/rev/chain_*.csv` |
| predicted vs. measured decision flips | `results/p128/flip_prediction.csv` | same |
| ablation of the rotation pipeline | `$B/final_bench 9 100 out.csv` (time), `$B/compact 1024 16 16 out.csv` (sizes, error) | `results/p128/final/final_bench.csv`, `results/p128/e11_compact/compact.csv` |
| per-operation costs, incl. standard seeded key switch + re-randomisation | `taskset -c 4 $B/final_bench 9 100 out.csv` | `results/p128/rev2/final_bench.csv` (earlier session: `results/p128/final/`) |
| key leakage to holders of the next epoch key | `$B/leak 200 out.csv` | `results/p128/rev2/leak.csv` |
| token choice vs. rotation budget | `$B/budget_bench out.csv` | `results/p128/final/budget_bench.csv` |
| multi-core scaling (pinned; NUMA first-touch `RF_FIRST_TOUCH=1`; permuted output `RF_PERM=1`) | `results/p128/rev2/run_scale.sh` | `results/p128/rev2/scale.csv` |
| multi-node shard scheduling (simulation) | `python results/e5_shard_sim.py 151.41 4 out.csv 1024` | `results/p128/rev2/shard/shard.csv` |
| multi-node makespan with I/O and chunk-size sweep | `python results/e5_shard_io.py 151.41 0.70 17408 10 out.csv` | `results/p128/rev2/shard/shard_io.csv` |
| GLWE vs. GGSW storage layout | `RF_CFG=B_r4x15 $B/cost 9 100 out.csv`, `RF_CFG=A_q4x8_r4x15 ...` | `results/p128/final/layout/` |
| in-place upgrade | `$B/upgrade B 16 2 4 15 64 16 16 out.csv` (and `A 4 8 4 15 16 8 8`) | `results/e6_upgrade/upgrade.csv` |
| plots | `python results/make_figs.py` | `figures/*.pdf` |

The scaling run builds large galleries by replicating the encrypted LFW templates; rotation cost does not depend on
ciphertext content. The multi-node experiment is a simulation from the measured per-ciphertext cost and excludes
network and storage I/O.

Other drivers: `correct` (sanity check), `noise` (external-product noise sweep, `results/p128/e1_noise`), `rerand`
(re-randomisation check), `compress` (storage-policy study, `results/p128/e10_compress`), `fftcheck`, `packtest`.
`results/legacy_n1536/` holds the runs at the legacy k=3, N=512 shape.

## Notes
- TFHE-rs 0.6 returns wrong external products when `base_log * level = 64`; all gadgets here use at most 60 bits.
- Re-randomisation makes rotated ciphertexts pseudorandom to parties without the current key; it does not flood the
  key-switching noise, so unlinkability against holders of epoch keys is heuristic.

License: MIT.
