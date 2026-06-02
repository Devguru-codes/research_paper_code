import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 1. Dataset Loading & Preprocessing
# ==============================================================================
DATASET_INFO = {
    'IndianPines': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
        'data_key': 'indian_pines_corrected',
        'gt_key':   'indian_pines_gt'
    },
    'PaviaU': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
        'data_key': 'paviaU',
        'gt_key':   'paviaU_gt'
    },
    'Salinas': {
        'local_data_path': '/kaggle/input/datasets/sreevallimanda/salinas-hyperspectral/Salinas_corrected.mat',
        'local_gt_path': '/kaggle/input/datasets/sreevallimanda/salinas-hyperspectral/Salinas_gt.mat',
        'data_key': 'salinas_corrected',
        'gt_key':   'salinas_gt'
    },
    'WHU_Hi': {
        'local_data_path': '/kaggle/input/datasets/devgurucodes/whu-dataset/WHU_Hi_HanChuan.mat',
        'local_gt_path': '/kaggle/input/datasets/devgurucodes/whu-dataset/WHU_Hi_HanChuan_gt.mat',
        'data_key': 'WHU_Hi_HanChuan',
        'gt_key':   'WHU_Hi_HanChuan_gt'
    }
}

import urllib.request

def load_dataset(dataset_name, download_dir='./hsi_data'):
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Dataset {dataset_name} not found in DATASET_INFO.")
        
    info = DATASET_INFO[dataset_name]
    
    if 'data_url' in info:
        os.makedirs(download_dir, exist_ok=True)
        data_path = os.path.join(download_dir, f"{dataset_name}_data.mat")
        gt_path = os.path.join(download_dir, f"{dataset_name}_gt.mat")
        
        if not os.path.exists(data_path):
            print(f"Downloading {dataset_name} data...")
            urllib.request.urlretrieve(info['data_url'], data_path)
        if not os.path.exists(gt_path):
            print(f"Downloading {dataset_name} ground truth...")
            urllib.request.urlretrieve(info['gt_url'], gt_path)
    else:
        data_path = info['local_data_path']
        gt_path = info['local_gt_path']
        
    try:
        img = sio.loadmat(data_path)[info['data_key']]
        gt = sio.loadmat(gt_path)[info['gt_key']]
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None
        
    return img, gt

def apply_pca(X, num_components=30):
    newX = np.reshape(X, (-1, X.shape[2]))
    from sklearn.decomposition import PCA
    pca = PCA(n_components=num_components, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], num_components))
    return newX

def pad_with_zeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def create_image_cubes(X, y, window_size=5, remove_zero_labels=True):
    margin = int((window_size - 1) / 2)
    zero_padded_X = pad_with_zeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if remove_zero_labels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

# ==============================================================================
# 2. Architectures (SE, CBAM, ECA, SSMRB)
# ==============================================================================
class SENet_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SENet_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ECA_Block(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CBAM_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM_Block, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid_ca = nn.Sigmoid()
        # Spatial attention
        self.conv_sa = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_sa = nn.Sigmoid()
        
    def forward(self, x):
        # CA
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        ca_out = self.sigmoid_ca(avg_out + max_out).view(b, c, 1, 1)
        x = x * ca_out.expand_as(x)
        # SA
        avg_out_sa = torch.mean(x, dim=1, keepdim=True)
        max_out_sa, _ = torch.max(x, dim=1, keepdim=True)
        sa_in = torch.cat([avg_out_sa, max_out_sa], dim=1)
        sa_out = self.sigmoid_sa(self.conv_sa(sa_in))
        return x * sa_out

class SSMRB_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SSMRB_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc_gamma = nn.Linear(channel // reduction, channel)
        self.fc_beta = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        gamma = self.sigmoid(self.fc_gamma(y)).view(b, c, 1, 1)
        beta = self.fc_beta(y).view(b, c, 1, 1)
        return (x * gamma) + beta

# Base CNN Model
class HSI_CNN(nn.Module):
    def __init__(self, num_classes, num_bands, attention_type='None'):
        super(HSI_CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_bands, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.attention_type = attention_type
        if attention_type == 'SE':
            self.attention = SENet_Block(128)
        elif attention_type == 'CBAM':
            self.attention = CBAM_Block(128)
        elif attention_type == 'ECA':
            self.attention = ECA_Block(128)
        elif attention_type == 'SSMRB':
            self.attention = SSMRB_Block(128)
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
        # For capturing activations
        self.pre_modulation_acts = None
        self.post_modulation_acts = None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        self.pre_modulation_acts = x.detach().cpu().numpy()
        
        if self.attention_type != 'None':
            x = self.attention(x)
            
        self.post_modulation_acts = x.detach().cpu().numpy()
            
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==============================================================================
# 3. Training & Evaluation Pipeline
# ==============================================================================
def train_and_eval(model, train_loader, test_loader, device, epochs=100, patience=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)
                output = model(data)
                val_loss += criterion(output, target).item()
                
        val_loss /= len(test_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    model.load_state_dict(best_model_state)
    
    # Final Eval
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, dtype=torch.float32)
            output = model(data)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())
            
    oa = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    cm_sum = cm.sum(axis=1)
    per_class_acc = np.divide(np.diag(cm), cm_sum, out=np.zeros_like(np.diag(cm), dtype=float), where=cm_sum!=0)
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(all_targets, all_preds)
    
    return oa, aa, kappa

# ==============================================================================
# 4. Main Execution
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    datasets = ['IndianPines', 'PaviaU'] # Can add 'Salinas', 'WHU_Hi'
    models_to_test = ['SE', 'CBAM', 'ECA', 'SSMRB']
    splits = [0.05, 0.10, 0.15]
    
    results = []
    
    for ds_name in datasets:
        print(f"--- Processing {ds_name} ---")
        img, gt = load_dataset(ds_name)
        if img is None: continue
            
        img_pca = apply_pca(img, num_components=30)
        X, y = create_image_cubes(img_pca, gt, window_size=11)
        # NCHW format
        X = np.transpose(X, (0, 3, 1, 2))
        
        num_classes = int(np.max(y)) + 1
        
        # 1. Baseline Comparisons (Issue 1)
        for att in models_to_test:
            print(f"Training {att} on 5% split...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42, stratify=y)
            train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
            test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=64, shuffle=False)
            
            model = HSI_CNN(num_classes=num_classes, num_bands=30, attention_type=att).to(device)
            oa, aa, kappa = train_and_eval(model, train_loader, test_loader, device)
            
            print(f"{att} - OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}")
            results.append({'Dataset': ds_name, 'Model': att, 'Split': 0.05, 'OA': oa, 'AA': aa, 'Kappa': kappa})
            
            # Activation Extraction (Issue 2) - Only on 5% Indian Pines
            if ds_name == 'IndianPines' and att in ['SE', 'SSMRB']:
                print(f"Extracting activation distributions for {att}...")
                model.eval()
                sample_data = torch.tensor(X_test[:500]).to(device, dtype=torch.float32) # Get first 500 samples
                _ = model(sample_data)
                
                pre_mean = np.mean(model.pre_modulation_acts, axis=(0, 2, 3)) # Mean per channel
                post_mean = np.mean(model.post_modulation_acts, axis=(0, 2, 3))
                
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.hist(pre_mean, bins=30, color='blue', alpha=0.7)
                plt.title(f"Pre-Modulation Activations ({att})")
                
                plt.subplot(1, 2, 2)
                plt.hist(post_mean, bins=30, color='green', alpha=0.7)
                plt.title(f"Post-Modulation Activations ({att})")
                
                plt.tight_layout()
                plt.savefig(f'activation_dist_{att}.pdf')
                plt.close()

        # 2. Data Efficiency (Issue 7) - SSMRB only
        for split in [0.10, 0.15]:
            print(f"Training SSMRB on {split*100}% split...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0-split, random_state=42, stratify=y)
            train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
            test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=64, shuffle=False)
            
            model = HSI_CNN(num_classes=num_classes, num_bands=30, attention_type='SSMRB').to(device)
            oa, aa, kappa = train_and_eval(model, train_loader, test_loader, device)
            results.append({'Dataset': ds_name, 'Model': 'SSMRB', 'Split': split, 'OA': oa, 'AA': aa, 'Kappa': kappa})

    # Save all results
    df = pd.DataFrame(results)
    df.to_csv("kaggle_results.csv", index=False)
    print("\nExperiments complete. Results saved to kaggle_results.csv")
    print("Activation distribution PDFs saved.")
