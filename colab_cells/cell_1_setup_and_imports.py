# =============================================================================
# Cell 1: Setup and Imports (Kaggle T4x2)
# =============================================================================
# GPU: Use Kaggle → Settings → Accelerator → GPU T4 x2
# NOTE: P100 (sm_60) is NO LONGER supported by Kaggle's PyTorch.
#       T4 (sm_75) works perfectly and has 2×15GB VRAM.

import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Install missing packages (Kaggle has most pre-installed)
for pkg in ['thop', 'PyWavelets']:
    try:
        __import__(pkg if pkg != 'PyWavelets' else 'pywt')
    except ImportError:
        install(pkg)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import scipy.io as sio
from scipy import signal
import pywt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             classification_report, cohen_kappa_score,
                             precision_score, recall_score, f1_score)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for stability
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import urllib.request
import time
import random
import json
import gc
from thop import profile as thop_profile

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Using device: {device}")

# Create persistent output directories
GRAPH_DIR = '/kaggle/working/paper_graphs/'
MODEL_DIR = '/kaggle/working/saved_models/'
RESULTS_DIR = '/kaggle/working/results/'
for d in [GRAPH_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("All output will be saved to /kaggle/working/ (persists even after disconnect).")
print("Setup complete!")
