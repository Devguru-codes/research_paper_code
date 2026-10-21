# Does the choice of channel attention matter for few-label hyperspectral classification?

Code, results and manuscript for the Remote Sensing Letters submission
**"Does the choice of channel attention matter for few-label hyperspectral classification? A controlled comparison under random, standard and patch-disjoint splits"**.

Seven channel-modulation blocks (SE, ECA, CBAM, GCT, SRM, SNSM γ, SNSM γ+β) plus a plain CNN are compared on one backbone
under three evaluation protocols, with paired seeds and Holm-corrected Wilcoxon tests.

## Layout

```
Paper_SMCNN/
├── paper/                              Manuscript (Taylor & Francis Interact template)
│   ├── snsm_rsl.tex / snsm_rsl.pdf     source and compiled paper
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
│   ├── results/                        one row per run (see below), efficiency, ground truths
│   ├── make_paper_tables.py            writes paper/tables/ from results/
│   └── make_paper_figures.py           writes paper/figures/ from results/
└── archive/                            earlier experiments (superseded; kept for history)
```

`results/`:
- `runs.csv` — random 5% splits (Phase 1; Phase 3 adds GCT/SRM and IP seeds 47-56 for the eight backbone variants)
- `phase2_runs.csv` — ablation and label budget
- `phase3_disjoint_runs.csv` — patch-disjoint splits (PU and KSC: seeds 42-56; IP: seeds 42-46, not used in the paper because five IP classes get no test pixels)
- `phase4_band_runs.csv`, `phase4_band_efficiency.csv` — raw-band modulation
- `phase5_std_runs.csv` — standard Indian Pines split, seeds 42-56
- `phase6_counts_runs.csv` — random Indian Pines split with 15 training pixels for the three smallest classes and 50 for the others, seeds 42-56
- `phase7_scarcity_runs.csv` — random 5% splits with only the three smallest classes changed (`smallfix`: IP to 15 pixels; `scarce`: PU and KSC to 1/1/2), seeds 42-56
- `efficiency.csv` — parameters, MACs and latency on an IP patch

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

`archive/` holds the earlier notebooks and their outputs. Several of those runs selected checkpoints on the test set, so their numbers are superseded by `experiments/results/`.

## License

Provided for academic research and peer review.
