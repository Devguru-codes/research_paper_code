# =============================================================================
# Cell 1: Setup & Dependencies
# =============================================================================
# Transformer Baseline Notebook for HIA-PAL-SMCNN paper
# Runs SpectralFormer and SSFTT under IDENTICAL conditions as the main paper
# Same datasets, same splits, same seeds, same metrics

!pip install -q einops timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import random
import scipy.io as sio
import urllib.request
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

RESULTS_DIR = '/kaggle/working/transformer_results/'
os.makedirs(RESULTS_DIR, exist_ok=True)
