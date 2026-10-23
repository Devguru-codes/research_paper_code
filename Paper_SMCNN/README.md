# Channel attention in lightweight hyperspectral CNNs: an equivalence-tested comparison

Code, results and manuscript for the Remote Sensing Letters submission
**"Channel attention in lightweight hyperspectral CNNs: an equivalence-tested comparison"**.

Seven channel-modulation blocks (SE, ECA, CBAM, GCT, SRM, SNSM γ, SNSM γ+β) plus a plain CNN are compared on one backbone
on five scenes under seven protocols, with paired seeds, Holm-corrected Wilcoxon tests and TOST equivalence tests.

## Layout

```
Paper_SMCNN/
├── paper/                              Manuscript (Taylor & Francis Interact template)
│   ├── snsm_rsl.tex / snsm_rsl.pdf     source and compiled paper
│   ├── supplement.tex / supplement.pdf supplementary tables and the SNSM block diagram
│   ├── cover_letter.tex / .pdf         cover letter to the editor
│   ├── refs.bib                        references (DOIs checked against Crossref)
│   ├── interact.cls, tfcad.bst         T&F class and Chicago author-date style
│   ├── tables/                         generated LaTeX tables and inline numbers
│   └── figures/                        generated figures + SNSM block diagram (TikZ)
├── experiments/
│   ├── phase1_final_experiments.ipynb  random 5% splits, all models (seeds 42-46)
│   ├── phase2_ablation_budget.ipynb    SNSM descriptor ablation; 1%/10% label budget
│   ├── phase3_disjoint_gct_srm.ipynb   GCT/SRM on random splits; patch-disjoint splits; extra seeds
│   ├── phase4_band_modulation.ipynb    SNSM and other blocks applied to raw bands in wavelength order
│   ├── phase5_standard_splits.ipynb    standard fixed Indian Pines split (SpectralFormer)
│   ├── phase6_matched_counts.ipynb     random IP split with the standard split's class counts (50/15)
│   ├── phase7_class_scarcity.ipynb     only the three smallest classes changed: IP to 15 pixels, PU/KSC to 1/1/2
│   ├── phase8_robustness.ipynb         WHU-Hi, deeper host network, learning rates, train-only PCA, hardest-class scarcity
│   ├── results/                        one row per run (see below), efficiency, ground truths
│   ├── make_paper_tables.py            writes paper/tables/ from results/
│   ├── make_paper_figures.py           writes paper/figures/ from results/
│   └── check_notebooks.py              fails if the notebooks' shared code copies differ
```

`results/`:
- `runs.csv` — random 5% splits (Phase 1; Phase 3 adds GCT/SRM and IP seeds 47-56 for the eight backbone variants)
- `phase2_runs.csv` — ablation and label budget
- `phase3_disjoint_runs.csv` — patch-disjoint splits (PU and KSC: seeds 42-56; IP: seeds 42-46, not used in the paper because five IP classes get no test pixels)
- `phase4_band_runs.csv`, `phase4_band_efficiency.csv` — raw-band modulation
- `phase5_std_runs.csv` — standard Indian Pines split, seeds 42-56
- `phase6_counts_runs.csv` — random Indian Pines split with 15 training pixels for the three smallest classes and 50 for the others, seeds 42-56
- `phase7_scarcity_runs.csv` — random 5% splits with only the three smallest classes changed (`smallfix`: IP to 15 pixels; `scarce`: PU and KSC to 1/1/2), seeds 42-56
- `phase8_runs.csv` — robustness studies (column `study`): WHU-Hi random and patch-disjoint (seeds 42-51), deeper host on IP random and KSC patch-disjoint (models prefixed `deep_`, seeds 42-51), learning rate 3e-4 on IP random (seeds 42-46), learning rate 3e-3 on IP random (seeds 42-56), train-only PCA on KSC patch-disjoint and hardest-class scarcity on PU and KSC (seeds 42-56)
- `efficiency.csv` — parameters, MACs and latency on an IP patch

### How `results/` was assembled

Each CSV row is one run, keyed by (dataset, model, seed). Kaggle jobs (account `dev123123456`, Tesla T4 unless noted)
ran a notebook, sometimes restricted to some datasets or seeds; their `runs.csv` outputs were concatenated as below.

| File | Rows | Source job (notebook) |
|---|---|---|
| `runs.csv` | IP, PU, BW, Salinas, WHU-Hi: all Phase 1 models, seeds 42-46 | `phase1-shard-a`, `phase1-shard-b`, plus separate ECA and CBAM runs (Phase 1) |
| | KSC: all Phase 1 models, seeds 42-46 (replaced the pre-int16-fix rows) | `ksc-rerun-phase1` (Phase 1) |
| | GCT, SRM on IP, PU, KSC, BW, seeds 42-46 | `phase3-disjoint-gct-srm` (Phase 3) |
| | IP, the eight backbone variants, seeds 47-56 | `random-seeds-ip` (Phase 3) |
| `efficiency.csv` | Phase 1 models; KSC rows from the KSC rerun; GCT and SRM rows | same jobs as above |
| `phase2_runs.csv` | IP ablation and budget, seeds 42-46 | `phase2-ablation-budget` (Phase 2) |
| | KSC ablation and budget, seeds 42-46 (replaced the pre-int16-fix rows) | `ksc-rerun-phase2` (Phase 2) |
| `phase3_disjoint_runs.csv` | IP, PU, KSC, seeds 42-46 | `phase3-disjoint-gct-srm` (Phase 3) |
| | PU, seeds 47-56 | `disjoint-seeds-pu-p100` (Phase 3; requested P100, ran on T4) |
| | KSC, seeds 47-56 | Phase 3 notebook executed locally on an RTX 3050 laptop GPU |
| `phase4_band_runs.csv`, `phase4_band_efficiency.csv` | IP, PU, KSC, seeds 42-46 | `phase4-band-modulation` (Phase 4) |
| `phase5_std_runs.csv` | IP standard split, seeds 42-56 | `std-splits-ip` (Phase 5) |
| `phase6_counts_runs.csv` | IP random 50/15, seeds 42-56 | `matched-counts-ip` (Phase 6) |
| `phase7_scarcity_runs.csv` | IP `smallfix` and KSC `scarce`, seeds 42-56 | `scarcity-ip-ksc` (Phase 7) |
| | PU `scarce`, seeds 42-56 | `scarcity-pu` (Phase 7) |
| `phase8_runs.csv` | `whu_random` | `p8-whu-random` (Phase 8) |
| | `whu_disjoint` | `p8-whu-disjoint` (Phase 8) |
| | deeper host, learning rates, train-only PCA, `hard_ksc` | `p8-misc` (Phase 8) |
| | `hard_pu` | `p8-hard-pu` (Phase 8) |
| | `lr3e-3_ip_random`, seeds 47-56 | `p8-lr-high` (Phase 8, `PART = 'lr_high'`) |

Before any notebook is changed, run `python experiments/check_notebooks.py`: it fails if the copies of the shared
data loader and training loop differ between notebooks.

## Protocol

Every model uses the same inputs, splits and schedule:

- KSC is read as int16: the distributed file is uint16 but holds small negative values (−27…−1) that otherwise wrap to 65509…65535
- Wiener 5×5 per band → PCA (30 components, fitted on the whole scene without labels) → + db4 wavelet approximation = 60 features
- 15×15 patches
- Splits: **random** (5% of labelled pixels per class), **standard** (fixed SpectralFormer IP training/test sets, downloaded from the authors' GitHub repository inside the notebook), **patch-disjoint** (per class, the 5% of pixels nearest a random centre train; test pixels within 14 px of any training pixel are dropped)
- AdamW (lr 1e-3, weight decay 1e-4), StepLR (×0.5 every 40 epochs), batch 64, **exactly 100 epochs**
- No validation- or test-based checkpoint selection or early stopping; the final-epoch model is evaluated once
- Kaggle, NVIDIA Tesla T4; the KSC patch-disjoint seeds 47-56 ran on a local RTX 3050 (Phase 3 notebook executed locally)

## Reproduce

1. On Kaggle, import each `experiments/phase*.ipynb`, add the `devgurucodes/datasets-hsi-all` dataset as input, choose GPU T4, then *Save & Run All*. Each notebook writes its CSVs to `/kaggle/working/<phase>/`; copy them into `experiments/results/` under the names above (extra seeds are appended to the same files).
2. Rebuild the tables, figures and paper:

```bash
python experiments/make_paper_tables.py && python experiments/make_paper_figures.py
```

```bash
cd paper && latexmk -pdf snsm_rsl.tex
```

## Archive

The earlier notebooks and their outputs were removed from `main`; they remain in the git tag `archive-before-cleanup` (`git checkout archive-before-cleanup -- Paper_SMCNN/archive`). Several of those runs selected checkpoints on the test set, so their numbers are superseded by `experiments/results/`.

## License

Provided for academic research and peer review.
