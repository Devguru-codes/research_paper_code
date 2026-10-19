# Spectral-Neighbourhood Self-Modulation (SNSM) for Hyperspectral Image Classification

Code, results and manuscript for the Remote Sensing Letters submission
**"Spectral-neighbourhood self-modulation for lightweight hyperspectral image classification with scarce labels"**.

## Layout

```
Paper_SMCNN/
├── paper/                              Manuscript (Taylor & Francis Interact template)
│   ├── snsm_rsl.tex / snsm_rsl.pdf     source and compiled paper
│   ├── refs.bib                        42 references (DOIs checked against Crossref)
│   ├── interact.cls, tfcad.bst         T&F class and Chicago author-date style
│   ├── tables/                         generated LaTeX tables and inline numbers
│   └── figures/                        generated figures + SNSM block diagram (TikZ)
├── experiments/
│   ├── phase1_final_experiments.ipynb  main comparison at 5% labels (Tables 2-3, 6-7)
│   ├── phase2_ablation_budget.ipynb    descriptor ablation and 1%/10% label budget (Tables 4-5)
│   ├── results/                        runs.csv and phase2_runs.csv (one row per run), efficiency.csv, map
│   ├── make_paper_tables.py            writes paper/tables/ from results/
│   └── make_paper_figures.py           writes paper/figures/ from results/
└── archive/                            earlier experiments (superseded; kept for history)
```

## Protocol

Every model uses the same inputs, splits and schedule:

- Wiener 5×5 per band → PCA (30 components, fitted on the whole scene without labels) → + db4 wavelet approximation = 60 features
- 15×15 patches; 5% of labelled pixels per class for training, the rest for testing (capped at 50,000 for Salinas and WHU-Hi)
- Seeds 42–46; AdamW (lr 1e-3, weight decay 1e-4), StepLR (×0.5 every 40 epochs), batch 64, **exactly 100 epochs**
- No validation- or test-based checkpoint selection or early stopping; the final-epoch model is evaluated once
- Runs on Kaggle, NVIDIA Tesla T4

Models: plain CNN, SE, ECA, CBAM, FC γ+β, SNSM γ, SNSM γ+β (all on one backbone), HybridSN, SSFTT.
Datasets: Indian Pines, Pavia University, KSC, Botswana (all models); Salinas and WHU-Hi HanChuan (SNSM only).

## Reproduce

1. On Kaggle, import `experiments/phase1_final_experiments.ipynb`, add the `devgurucodes/datasets-hsi-all` dataset as input, choose GPU T4, then *Save & Run All*.
   The notebook writes everything to `/kaggle/working/final/`. To use two GPUs, run copies with different `CORE_DATASETS`/`LARGE_DATASETS` lists and concatenate their `runs.csv` and `efficiency.csv`.
   Run `experiments/phase2_ablation_budget.ipynb` the same way (one T4, about 15 minutes of training); it writes `/kaggle/working/phase2/runs.csv`, which goes to `experiments/results/phase2_runs.csv`.
2. Put the merged CSVs in `experiments/results/`, then rebuild the tables, figures and paper:

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
