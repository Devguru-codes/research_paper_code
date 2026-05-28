# HIA-PAL-SMCNN: Hyperspectral Image Analysis through Self-Modulating CNNs

## Research Paper Code Repository

This repository contains the complete experimental code, results, and publication-ready figures for the paper:

> **"Advanced Hyperspectral Image Analysis in Precision Agriculture and Land Cover Mapping through Optimized Self-Modulating Convolutional Neural Networks (HIA-PAL-SMCNN)"**

## Repository Structure

```
research_work/
├── README.md
├── research.md                              # Complete paper draft with all data & ground truths
├── HIA_PAL_SMCNN_Paper_v3.docx             # Formatted Word manuscript (IEEE-style)
│
├── research-paper-notebook-swati-mam.ipynb  # Main notebook (1st run, cells 1-15)
├── research-paper-notebook-swati-mam_2nd_run.ipynb  # Full notebook (2nd run, cells 1-19)
├── swati-mam-paper-transformer-one.ipynb    # Transformer baselines (SpectralFormer + SSFTT)
│
├── results/                                 # 1st run results (JSON, CSV)
├── results_2nd_run/                         # 2nd run results + publication figures
│   ├── results/                             # JSON metrics (statistical, noise, interpretability)
│   └── paper_graphs/                        # Publication-ready figures (PDF + PNG)
├── results_transformer/                     # Transformer baseline results
└── paper_graphs/                            # 1st run publication figures
```

## Datasets Used

| Dataset | Sensor | Classes | Source |
|---|---|---|---|
| Indian Pines | AVIRIS | 16 | [Kaggle](https://www.kaggle.com/datasets) |
| Pavia University | ROSIS | 9 | [Kaggle](https://www.kaggle.com/datasets) |
| Botswana | Hyperion | 14 | [Kaggle](https://www.kaggle.com/datasets) |
| Kennedy Space Center | AVIRIS | 13 | [Kaggle](https://www.kaggle.com/datasets) |
| WHU-Hi HanChuan | Headwall Nano | 16 | [Kaggle](https://www.kaggle.com/datasets) |

## Key Results

| Dataset | OA (mean ± std) | Kappa |
|---|---|---|
| Indian Pines | 99.08 ± 0.27% | 0.9895 |
| Pavia University | 99.88 ± 0.06% | 0.9985 |
| Botswana | 99.94 ± 0.10% | 0.9993 |
| KSC | 92.51 ± 2.66% | 0.9165 |
| WHU-Hi | 99.93 ± 0.02% | 0.9992 |

## How to Run

1. Upload the notebook to [Kaggle](https://www.kaggle.com/)
2. Add datasets (Indian Pines, Pavia University, Botswana, KSC, WHU-Hi)
3. Enable GPU accelerator
4. Run all cells

## License

This code is provided for academic research purposes only.
