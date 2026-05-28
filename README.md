# HIA-PAL-SMCNN: Hyperspectral Image Analysis through Self-Modulating CNNs

## Research Paper Code Repository

This repository contains the complete experimental code, results, and publication-ready figures for the paper:

> **"Advanced Hyperspectral Image Analysis in Precision Agriculture and Land Cover Mapping through Optimized Self-Modulating Convolutional Neural Networks (HIA-PAL-SMCNN)"**

## Repository Structure

```
research_work/
├── research.md                              # Complete paper draft with all data
├── research-paper-notebook-swati-mam.ipynb  # Main notebook (1st run)
├── research-paper-notebook-swati-mam_2nd_run.ipynb  # Full notebook (2nd run, cells 1-19)
├── swati-mam-paper-transformer-one.ipynb    # Transformer baselines notebook
│
├── colab_cells/                             # Modular code cells
│   ├── cell_1_imports.py                    # Dependencies
│   ├── cell_2_dataset_loader.py             # Dataset loading
│   ├── cell_3_preprocessing_afdkf_plcwt.py  # AFDKF + PLCWT preprocessing
│   ├── cell_4_model_smcnn.py                # SMCNN architecture + SSMRB
│   ├── cell_5_optimizer_sfwoa.py            # SFWOA scheduler
│   ├── cell_6_training_loop.py              # Training pipeline
│   ├── cell_7_evaluation.py                 # Evaluation metrics
│   ├── cell_8_main_execution.py             # Main experiment driver
│   ├── cell_9_baselines.py                  # CNN baselines (2D, 3D, HybridSN)
│   ├── cell_10_ablation.py                  # Ablation study
│   ├── cell_11_cross_dataset.py             # Cross-dataset transfer
│   ├── cell_12_multiscale_ssmrb.py          # Multiscale analysis
│   ├── cell_13_visualizations.py            # Grad-CAM, t-SNE, etc.
│   ├── cell_14_noise_robustness.py          # Basic noise robustness
│   ├── cell_15_summary.py                   # Results summary
│   ├── cell_16_statistical_tests.py         # Statistical significance
│   ├── cell_17_extended_noise.py            # 5-type noise robustness
│   ├── cell_18_deep_interpretability.py     # γ/β analysis, Fisher, entropy
│   └── cell_19_scheduler_comparison.py      # LR scheduler comparison
│
├── spectral_cells/                          # Transformer baseline cells
│
├── results/                                 # 1st run results
├── results_2nd_run/                         # 2nd run results (all 19 cells)
│   ├── results/                             # JSON metrics
│   └── paper_graphs/                        # Publication-ready figures
├── results_transformer/                     # Transformer baseline results
└── paper_graphs/                            # 1st run figures
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
