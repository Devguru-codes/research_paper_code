# HIA-PAL-SMCNN: Hyperspectral Image Analysis through Self-Modulating CNNs

## Research Paper Code Repository

This repository contains the complete experimental code, results, and publication-ready manuscripts for the paper:

> **"Advanced Hyperspectral Image Analysis in Precision Agriculture and Land Cover Mapping through Optimized Self-Modulating Convolutional Neural Networks (HIA-PAL-SMCNN)"**

### Manuscripts

Two versions of the manuscript are available in this repository:
- 📄 **[10-Page Submission Version (PDF)](paper/paper_springer_short.pdf)**: The condensed, publication-ready manuscript prepared for submission to Springer Nature journals.
- 📄 **[26-Page Extended Version (PDF)](paper/paper_springer.pdf)**: The full, comprehensive manuscript containing extensive ablation studies, extended failure analysis, and additional theoretical motivations.

## Repository Structure

```
Paper_SMCNN/
├── README.md                                      # This file
├── research.md                                    # Extended notes on dataset handling and models
├── HIA_PAL_SMCNN_Paper_v3.docx                    # Original drafted manuscript
├── paper/                                         # LaTeX Source & Compiled PDFs
│   ├── paper_springer_short.pdf                   # Final 10-page submission manuscript
│   ├── paper_springer.pdf                         # Full 26-page extended manuscript
│   ├── sn-jnl.cls                                 # Springer Nature LaTeX class
│   └── ...                                        # .tex source files and bibliography
├── research-paper-notebook-swati-mam.ipynb        # Main notebook (1st run)
├── research-paper-notebook-swati-mam_2nd_run.ipynb # Full notebook (2nd run, final results)
├── swati-mam-paper-transformer-one.ipynb          # Transformer baselines
├── results_2nd_run/                               # Final empirical results
│   ├── results/                                   # JSON and CSV metric logs
│   └── paper_graphs/                              # Publication-ready figures (PDF/PNG)
└── results_transformer/                           # Transformer baseline logs
```

## Reproducibility

All experiments and runtime comparisons were conducted using identical preprocessing pipelines, hardware configurations (NVIDIA RTX A4000 GPU), optimization settings (AdamW + APCS), and evaluation protocols across 5 fixed random seeds (`42`, `100`, `2024`, `777`, `888`). The explicit experimental details can be found in the methodology section of the manuscripts.

## Datasets Evaluated

| Dataset | Sensor | Classes | Target Domain |
|---|---|---|---|
| Indian Pines | AVIRIS | 16 | Agriculture |
| Pavia University | ROSIS | 9 | Urban |
| Botswana | Hyperion | 14 | Swamp/Agriculture |
| Kennedy Space Center | AVIRIS | 13 | Coastal Wetlands |
| WHU-Hi HanChuan | Headwall Nano | 16 | Agriculture |

## How to Run

To replicate the experiments under severe data constraints (5% training split):
1. Import the primary notebook (`research-paper-notebook-swati-mam_2nd_run.ipynb`) into your environment (e.g., [Kaggle](https://www.kaggle.com/)).
2. Download the datasets from the public Hyperspectral Remote Sensing Scenes repository.
3. Enable GPU acceleration (minimum 16GB VRAM recommended).
4. Execute the notebook sequentially to reproduce the `results_2nd_run` data and figures.

## License

This code is provided for academic research and peer review purposes.
