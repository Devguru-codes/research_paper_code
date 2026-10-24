# Reviewer guide: from the paper to the experiments

This file maps every table, figure and number in *Channel attention in lightweight hyperspectral CNNs: an
equivalence-tested comparison* (`paper/snsm_rsl.pdf`) and its supplement (`paper/supplement.pdf`) to the runs that
produced them. All paths are relative to `Paper_SMCNN/`.

## How the numbers are produced

```
experiments/phase*.ipynb  --(Kaggle, Tesla T4)-->  experiments/results/*.csv  (one row per run)
experiments/results/*.csv --make_paper_tables.py--> paper/tables/*.tex  (tables + numbers.tex)
experiments/results/*.csv --make_paper_figures.py-> paper/figures/*.pdf
```

- **No number in the paper is typed by hand.** Tables are `\input` from `paper/tables/`, and every number in the text
  is a LaTeX macro (for example `\ipRandGain`) defined in `paper/tables/numbers.tex`. To see where a number comes from,
  search for its macro name in `experiments/make_paper_tables.py`.
- **Qualitative claims are asserted.** Sentences such as "SRM is the lowest on patch-disjoint KSC" or "no difference
  is significant" have an `assert` next to the macro that feeds them, so the script fails if the data stop supporting
  the text.
- **To check everything**, run the generator and rebuild the paper:

```bash
python experiments/make_paper_tables.py && python experiments/make_paper_figures.py
```

```bash
python experiments/check_notebooks.py
```

The second command confirms that every notebook carries the same copy of the data loader and training loop.

Each CSV row is one run, keyed by `(dataset, model, seed)`, plus `study` where a file holds several studies. Columns
include `OA`, `AA`, `kappa`, `n_train`, `n_test` and `per_class` (a JSON list of class accuracies). The Kaggle job of
every row is listed in `README.md` under *How `results/` was assembled*.

## Model names in the CSVs

| Paper | CSV `model` |
|---|---|
| Plain CNN | `plain` |
| SE | `se_gamma` |
| ECA | `eca_gamma` |
| CBAM | `cbam` |
| GCT | `gct` |
| SRM | `srm` |
| SNSM (γ) | `snsm_gamma` |
| SNSM (γ+β) | `snsm` |
| FC (γ+β) gate, HybridSN, SSFTT (supplement only) | `fc_gamma_beta`, `hybridsn`, `ssftt` |
| Deeper host (Phase 8) | the same names prefixed `deep_` |

## Protocols and where their runs are

| Protocol in the paper | Scenes | Seeds | CSV and filter | Notebook |
|---|---|---|---|---|
| Random 5% | IP | 42–56 | `runs.csv`, `dataset == 'IndianPines'` | Phase 1; GCT/SRM and seeds 47–56 from Phase 3 |
| Random 5% | PU, KSC, BW | 42–46 | `runs.csv` | Phase 1; GCT/SRM from Phase 3 |
| Random 5% | WHU | 42–51 | `phase8_runs.csv`, `study == 'whu_random'` | Phase 8 |
| Smallest 15 | IP | 42–56 | `phase7_scarcity_runs.csv`, `study == 'smallfix'` | Phase 7 |
| Random 50/15 | IP | 42–56 | `phase6_counts_runs.csv` | Phase 6 |
| Standard (SpectralFormer) | IP | 42–56 | `phase5_std_runs.csv` (`dataset` is `Indian`) | Phase 5 |
| Smallest 1/1/2 | PU, KSC | 42–56 | `phase7_scarcity_runs.csv`, `study == 'scarce'` | Phase 7 |
| Hardest 1/1/2 | PU, KSC | 42–56 | `phase8_runs.csv`, `study in ('hard_pu', 'hard_ksc')` | Phase 8 |
| Patch-disjoint (one cluster per class) | PU, KSC | 42–56 | `phase3_disjoint_runs.csv` | Phase 3 |
| Patch-disjoint | WHU | 42–51 | `phase8_runs.csv`, `study == 'whu_disjoint'` | Phase 8 |
| Block-disjoint (16×16 blocks) | PU, KSC | 42–56 | `phase8_runs.csv`, `study in ('blocks_pu', 'blocks_ksc')` | Phase 8 |
| Deeper host (4 blocks × 128 channels) | IP random, KSC patch-disjoint | 42–51 | `phase8_runs.csv`, `study in ('deep_ip_random', 'deep_ksc_disjoint')` | Phase 8 |
| PCA fitted on training pixels only | KSC patch-disjoint | 42–56 | `phase8_runs.csv`, `study == 'trainpca_ksc_disjoint'` | Phase 8 |
| Learning rate 3×10⁻⁴ | IP random | 42–46 | `phase8_runs.csv`, `study == 'lr3e-4_ip_random'` | Phase 8 |
| Learning rate 3×10⁻³ | IP random | 42–56 | `phase8_runs.csv`, `study == 'lr3e-3_ip_random'` | Phase 8 |
| SNSM probe ablation | IP random | 42–46 | `phase2_runs.csv`, `study == 'ablation'`, `dataset == 'IndianPines'` | Phase 2 |
| Blocks on raw bands (no PCA) | IP, PU, KSC random | 42–46 | `phase4_band_runs.csv` | Phase 4 |
| GCT gate statistics | block-disjoint PU/KSC, IP random | as above | `phase8_gct_gates.csv` | Phase 8 |
| Model size and latency | IP patch | — | `efficiency.csv` | Phase 1, Phase 3 |

The split functions are in cell 2 of the notebooks: `split_centers` (random), `split_counts` (50/15),
`split_scarcity` (smallest), `split_hard` (hardest), `split_disjoint` (patch-disjoint), `split_blocks`
(block-disjoint) and `cap_test` (50,000-pixel test cap on WHU). Training is cell 4 (`train_and_evaluate`): AdamW,
100 epochs, the final model evaluated once, with no validation- or test-based model selection.

**Not used in the paper:** `runs.csv` rows for Salinas and the Phase 1 WHU rows (seeds 42–46, superseded by the
10-seed Phase 8 runs); `phase2_runs.csv` rows with `study == 'budget'` and the KSC ablation; and the IP rows of
`phase3_disjoint_runs.csv` (five IP classes get no test pixels under that split).

## Main paper

| Item | Generated file | Data | Code |
|---|---|---|---|
| Table 1 (scenes and split sizes) | `tables/datasets.tex` | `results/_gt_*.npy` ground truths, seed 42 | `make_paper_tables.py`, "Table: datasets" |
| Table 2 (random 5%, OA/AA) | `tables/main.tex` | `runs.csv` | `main_table` |
| Table 3 (IP under four protocols) | `tables/protocol_ip.tex` | `runs.csv`, `phase7` smallfix, `phase6`, `phase5` | `block_table` |
| Table 4 (paired differences, all settings) | `tables/paired.tex` | every file of the protocol table above | `SETTINGS`, `paired`, `paired_table` |
| Table 5 (PU/KSC/WHU, random vs patch-disjoint) | `tables/protocol_dj.tex` | `runs.csv`, `phase3_disjoint_runs.csv`, `phase8` whu_* | `block_table` with `DJ_SETS` |
| Figure 1 (patch-disjoint split of PU) | `figures/disjoint_split_pu.pdf` | `_gt_PaviaU.npy`, seed 42 | `make_paper_figures.py` (asserts it matches the notebook split) |
| Figure 2 (paired differences, patch-disjoint) | `figures/disjoint_paired.pdf` | `phase3_disjoint_runs.csv` | `make_paper_figures.py` |

**Reading Table 4.** Each cell is the mean over seeds of AA(SNSM γ+β) − AA(block) on the same seed and split.
- `*`: Wilcoxon signed-rank p < 0.05 after Holm correction over the seven comparisons in that row.
- `‡`: also a discovery under Benjamini–Hochberg (FDR 5%) over all comparisons in the table.
- `†`: equivalent within ±1 AA point, meaning the 90% paired t interval lies inside (−1, 1) (TOST at α = 0.05).
- Rows with five seeds cannot reach p < 0.05, because the smallest two-sided Wilcoxon p with five pairs is 0.0625.

The statistics are implemented in `make_paper_tables.py`: `paired` (differences, intervals and tests), `holm`,
`equivalent` (TOST), `bh_count` and `bh_cut` (Benjamini–Hochberg). Each function has a self-test assert below its
definition.

## Supplement

| Item | Generated file | Data |
|---|---|---|
| Figure S1 (SNSM block diagram) | `figures/snsm_block.tex` | drawn by hand (TikZ); no data |
| Table S1 (all models, random 5%) | `tables/supp_main.tex` | `runs.csv` |
| Table S2 (95% intervals of Table 4) | `tables/supp_paired_ci.tex` | as Table 4 |
| Table S3 (every block − plain CNN) | `tables/supp_paired_plain.tex` | as Table 4 |
| Table S4 (equivalence at ±0.5, ±1, ±2) | `tables/supp_margin.tex` | as Table 4 |
| Table S5 (paired OA differences) | `tables/supp_paired_oa.tex` | as Table 4, column `OA` |
| Table S6 (scarcity, per-block AA) | `tables/supp_scarcity.tex` | `runs.csv`, `phase7_scarcity_runs.csv` |
| Table S7 (robustness, per-block AA) | `tables/supp_robust.tex` | `phase8_runs.csv` |
| Table S8 (SNSM probe ablation) | `tables/ablation.tex` | `phase2_runs.csv`, ablation |
| Table S9 (raw bands) | `tables/supp_band.tex` | `phase4_band_runs.csv` |
| Table S10 (size, MACs, latency, training time) | `tables/supp_efficiency.tex` | `efficiency.csv`, `runs.csv` |

## Key claims in the text

| Claim | Macro(s) | Computed from |
|---|---|---|
| IP random gap of 1.8–3.8 AA points; OA differs by only 0.17 | `\ipRandGain`, `\ipRandOAspread` | `runs.csv`, IP |
| The gap sits in Grass-pasture-mowed and Oats | `\pcGpmDisp`, `\pcOatsDisp`, `\pcAlfalfaSpread` | `per_class` column of `runs.csv`, IP |
| Blocks equivalent with ≥15 training pixels per class | `\spreadIPSmall`, `\spreadIPCnt`, `\spreadIPStd` | `phase7` smallfix, `phase6`, `phase5` |
| Blocks equivalent on WHU random | `\spreadWHUrand` | `phase8`, `whu_random` |
| Scarcity on PU/KSC does not favour dispersion gates | `\dropScarcePU`, `\dropScarceKSC`, `\hardMax` | `phase7` scarce, `phase8` hard_* |
| Patch-disjoint drop and spread | `\dropPU`, `\dropKSC`, `\dropWHU`, `\dropRange`, `\spreadDJ*`, `\ciwPUdj`, `\ciwKSCdj` | `phase3_disjoint_runs.csv`, `phase8` whu_disjoint |
| Block-disjoint: four blocks worse than plain on PU | `\blockPU`, `\blockKSC`, `\blockDropPU`, `\blockDropKSC`, `\blockSnsmPU`, `\spatialDrop` | `phase8`, blocks_* |
| Deeper plain CNN matches the best shallow block | `\deepPlainIP`, `\shallowBestIP`, `\deepPlainP` (Welch t-test), `\deepIPspread` | `phase8` deep_ip_random vs `runs.csv` |
| Learning-rate dependence | `\plainIP`, `\plainLrHigh`, `\lrHighMax`, `\lrLowGain`, `\bestLrRange`, `\bestLrGap` | `phase8` lr* and `runs.csv` |
| Train-only PCA leaves results unchanged | `\trainPcaRange`, `\scenePcaRange` | `phase8` trainpca vs `phase3` KSC |
| Variance carries the gap; gradient input not needed | `\ablMeanDrop`, `\ablVarGain`, `\ablVarWins` | `phase2_runs.csv`, ablation |
| Same pattern on raw bands | `\bandIPgain`, `\bandOtherSpread` | `phase4_band_runs.csv` |
| GCT gate stays near the identity | `\gctGateDev` | `phase8_gct_gates.csv` |
| Equivalence margin and seed SD | `\margin`, `\seedSdIP` | `runs.csv`, IP |
| Number of comparisons and discoveries | `\nSettings`, `\bhTotal`, `\bhSig`, `\holmSig` | all paired settings |
| SNSM latency about 3.7× the plain CNN | `\latencyRatio`, `\paramsSnsm` | `efficiency.csv` |

## Data details a reviewer may want to check

- **KSC encoding.** The distributed `KSC.mat` is unsigned 16-bit but holds small negative values. Every notebook reads
  it as signed (`img = img.view(np.int16)` in `load_dataset`). All KSC rows in `results/` come from runs after that fix;
  the jobs that replaced the earlier KSC rows are listed in `README.md`.
- **Features.** Wiener 5×5 per band, then PCA to 30 components fitted on all pixels without labels, plus the db4
  wavelet approximation, giving 60 features. `features_train_pca` in Phase 8 refits PCA on training pixels only.
- **Hardware.** Kaggle Tesla T4 for every run except KSC patch-disjoint seeds 47–56, which ran on an RTX 3050 laptop
  GPU for all eight blocks; each paired difference therefore compares blocks run on the same GPU. `env.json` in each
  Kaggle output records the device.
- **Reproducibility check.** The Phase 8 job `p8-blocks-ksc` reran GCT on IP random seeds 42–46
  (`study == 'gct_gate_ip'`). `make_paper_tables.py` asserts that these runs equal the Phase 1 GCT runs exactly.
- **Archived runs.** Earlier notebooks, some of which selected checkpoints on the test set, are not used. They are kept
  only in the git tag `archive-before-cleanup`.
