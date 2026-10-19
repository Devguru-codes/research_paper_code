# Spectral-Neighbourhood Self-Modulation (SNSM) for Hyperspectral Image Classification

Code, results and manuscript for the Remote Sensing Letters submission
**"Spectral-neighbourhood self-modulation for lightweight hyperspectral image classification with scarce labels"**.

## Where things are

| Path | What it is |
|---|---|
| `notebooks/phase1_final_experiments.ipynb` | The one notebook behind every number in the paper |
| `code_and_results/final/` | Merged results: `runs.csv` (one row per training run), `efficiency.csv`, tables, classification map |
| `code_and_results/make_paper_tables.py` | Writes the paper's LaTeX tables and inline numbers from `final/*.csv` |
| `code_and_results/make_paper_figures.py` | Draws the paper's data figures from `final/*.csv` |
| `paper/rsl/` | Manuscript in the Taylor & Francis Interact template (`snsm_rsl.tex`, `refs.bib`, `snsm_rsl.pdf`) |

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

1. On Kaggle, import `notebooks/phase1_final_experiments.ipynb`, add the `devgurucodes/datasets-hsi-all` dataset as input, choose GPU T4, then *Save & Run All*.
   The notebook writes everything to `/kaggle/working/final/`. To use two GPUs, run copies with different `CORE_DATASETS`/`LARGE_DATASETS` lists and concatenate their `runs.csv` and `efficiency.csv`.
2. Put the merged CSVs in `code_and_results/final/`, then rebuild the tables and the paper:

```bash
python code_and_results/make_paper_tables.py && python code_and_results/make_paper_figures.py
```

```bash
cd paper/rsl && latexmk -pdf snsm_rsl.tex
```

## Older material

The other notebooks, `code_and_results/results*/`, `research.md` and the Springer manuscripts in `paper/` (`paper.tex`, `paper_springer*.tex`) are from earlier experiments, several of which selected checkpoints on the test set. They are kept for history only; their numbers are superseded by `code_and_results/final/`.

## License

Provided for academic research and peer review.
