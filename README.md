# RotaFace

Artifact for **"RotaFace: Scalable Key Rotation for FHE-Encrypted Biometric Databases"** (IEEE ISPA 2026).

RotaFace lets a cloud server re-key a TFHE-encrypted 1:N face gallery in place, from a small per-epoch token, without
decrypting it. The repository contains the Rust implementation (on the `core_crypto` layer of TFHE-rs 0.6.4), every
experiment driver, the analysis/plotting scripts, and the raw CSV/log outputs behind each table and figure of the paper.

## Build

```bash
cd code && cargo build --release        # Rust >= 1.75; binaries in code/target/release/
python3 -m venv .venv && .venv/bin/pip install numpy scipy scikit-learn matplotlib
```
Embeddings are not redistributed; see `data/README.md`. All binaries read `RF_DATA` (default `data/lfw_emb.f32`).
Parameter set: `RF_PARAMS=p128` (k=1, N=2048, q=2^64, sigma=3.2*2^24; the paper's main set) or unset (legacy k=3, N=512).

## Main building blocks (`code/src/lib.rs`)
| function | paper |
|---|---|
| `enc_template_b`, `enc_query_b`, `pack`, `twist`, `coef_score` | Sec. IV-A (slot-packed matching, Lemma 1) |
| `rotate_ggsw`, `ggsw_row_noise` | layout A (GGSW at rest), Prop. 1 |
| `ksk_gen_seeded`, `NativeKsk::expand`, `native_ks`, `truncate_split` | Algorithm 1 (seeded mask-only key switch, gadget-aligned storage) |
| `pk_gen`, `rerandomise` | re-randomisation (Sec. IV-C) |
| `pad_key`, `upgrade_glwe`, `upgrade_ggsw` | in-place security upgrade (Sec. IV-D) |
| `owner_reencrypt_fft` | owner-side baseline (Table VII) |

## Reproducing the paper
Single-core timings are sensitive to host load; run them on an idle machine.

| paper item | command (from repo root, `B=code/target/release`, `export RF_PARAMS=p128`) | output |
|---|---|---|
| Table IV (parameters, security) | `sage results/e0_security/est.sage`, `est_full.sage` (set `LATTICE_ESTIMATOR`) | `results/e0_security/*.log` |
| Table II (1:N accuracy) | `RF_RERAND=1 RF_COMPACT=1 $B/ident B 16 2 4 9 0,1024 24 <prefix>` then `python results/analyze_e2.py <dir> <out.csv>` | `results/p128/e2_*metrics.csv` |
| Fig. 2 (noise model) | `$B/chain_compact <logB> <l> <T> 16 16 out.csv`; `python results/noise_model_compact.py` | `results/p128/rev/chain_*.csv` |
| Table V (decision flips) | predictions from `results/p128/flip_prediction.csv` | same |
| Table VI (ablation) | `$B/final_bench 9 100 out.csv` (time), `$B/compact 1024 16 16 out.csv` (sizes, error) | `results/p128/final/final_bench.csv`, `results/p128/e11_compact/compact.csv` |
| Table VII (costs) | `$B/final_bench 9 100 out.csv` | `results/p128/final/final_bench.csv` |
| Fig. 3(a) token choice | `$B/budget_bench out.csv` | `results/p128/final/budget_bench.csv` |
| Fig. 3(b) scaling | `RF_COMPACT=1 $B/scale B 16 2 4 9 250000 1,2,4,8,16,24,32,48 3 out.csv` | `results/p128/final/scale_final.csv` |
| Fig. 3(c) shards | `python results/e5_shard_sim.py 150.96 4 out.csv` | `results/p128/e5_shard/shard.csv` |
| Layout comparison (Sec. VI-D) | `RF_CFG=B_r4x15 $B/cost 9 100 out.csv`, `RF_CFG=A_q4x8_r4x15 ...` | `results/p128/final/layout/` |
| Sec. VI-F upgrade | `$B/upgrade B 16 2 4 15 64 16 16 out.csv` (and `A 4 8 4 15 16 8 8`) | `results/e6_upgrade/upgrade.csv` |
| all figures | `python results/make_figs.py` | `paper/figures/*.pdf` |

Other drivers: `correct` (sanity check), `noise` (external-product noise sweep, `results/p128/e1_noise`), `rerand`
(re-randomisation check), `compress` (storage-policy study, `results/p128/e10_compress`), `fftcheck`, `packtest`.
`results/legacy_n1536/` holds the runs at the legacy k=3, N=512 shape.

## Notes
- TFHE-rs 0.6 returns wrong external products when `base_log * level = 64`; all gadgets here use at most 60 bits.
- Re-randomisation gives heuristic unlinkability (no noise flooding); see the paper's security discussion.

License: MIT.
