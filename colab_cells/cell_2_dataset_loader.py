# =============================================================================
# Cell 2: Dataset Loader — 7 Datasets (Reviewer Issue 7)
# =============================================================================
# Auto-download: IndianPines, PaviaU, Salinas, Botswana, KSC
# Manual upload: Houston2013, WHU-Hi (require registration)

import os
import numpy as np
import scipy.io as sio
import urllib.request
from sklearn.decomposition import PCA

DATASET_INFO = {
    'IndianPines': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
        'data_key': 'indian_pines_corrected',
        'gt_key':   'indian_pines_gt',
        'target_names': [
            'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
            'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
            'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
            'Soybean-clean', 'Wheat', 'Woods',
            'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'
        ]
    },
    'PaviaU': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
        'data_key': 'paviaU',
        'gt_key':   'paviaU_gt',
        'target_names': [
            'Asphalt', 'Meadows', 'Gravel', 'Trees',
            'Painted metal sheets', 'Bare Soil', 'Bitumen',
            'Self-Blocking Bricks', 'Shadows'
        ]
    },
    'Salinas': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
        'data_key': 'salinas_corrected',
        'gt_key':   'salinas_gt',
        'target_names': [
            'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
            'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery',
            'Grapes_untrained', 'Soil_vinyard_develop',
            'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
            'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
            'Lettuce_romaine_7wk', 'Vinyard_untrained',
            'Vinyard_vertical_trellis'
        ]
    },
    'Botswana': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/7/72/Botswana.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/5/58/Botswana_gt.mat',
        'data_key': 'Botswana',
        'gt_key':   'Botswana_gt',
        'target_names': [
            'Water', 'Hippo grass', 'Floodplain grasses1', 'Floodplain grasses2',
            'Reeds1', 'Riparian', 'Firescar2', 'Island interior',
            'Acacia woodlands', 'Acacia shrublands', 'Acacia grasslands',
            'Short mopane', 'Mixed mopane', 'Exposed soils'
        ]
    },
    'KSC': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/a/a6/KSC_gt.mat',
        'data_key': 'KSC',
        'gt_key':   'KSC_gt',
        'target_names': [
            'Scrub', 'Willow swamp', 'Cabbage palm hammock',
            'Cabbage palm/oak hammock', 'Slash pine', 'Oak/broadleaf hammock',
            'Hardwood swamp', 'Graminoid marsh', 'Spartina marsh',
            'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water'
        ]
    },
    # ── KAGGLE INPUT DATASETS ──
    # Added via: Notebook → Input → + Add Input
    'Houston2013': {
        'data_url': 'KAGGLE_INPUT',  # /kaggle/input/hsi-houston/Houston.mat
        'gt_url':   'KAGGLE_INPUT',
        'data_key': 'Houston',
        'gt_key':   'Houston_gt',
        'target_names': [
            'Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
            'Soil', 'Water', 'Residential', 'Commercial', 'Road',
            'Highway', 'Railway', 'Parking Lot 1', 'Parking Lot 2',
            'Tennis Court', 'Running Track'
        ]
    },
    'WHU_Hi': {
        'data_url': 'KAGGLE_INPUT',  # /kaggle/input/whu-dataset/WHU_HI_HanChuan.mat
        'gt_url':   'KAGGLE_INPUT',
        'data_key': 'WHU_Hi_HanChuan',
        'gt_key':   'WHU_Hi_HanChuan_gt',
        'target_names': [
            'Strawberry', 'Cowpea', 'Soybean', 'Sorghum', 'Water spinach',
            'Watermelon', 'Greens', 'Trees', 'Grass', 'Red roof',
            'Gray roof', 'Plastic', 'Bare soil', 'Road', 'Bright object', 'Water'
        ]
    },
}

def _find_in_kaggle_input(pattern):
    """Search /kaggle/input/ for a file matching the glob pattern."""
    import glob
    base = '/kaggle/input'
    if not os.path.exists(base):
        return None
    matches = glob.glob(f'{base}/**/{pattern}', recursive=True)
    return matches[0] if matches else None

def download_dataset(dataset_name):
    """Downloads standard HSI datasets or finds them in Kaggle Input."""
    info = DATASET_INFO[dataset_name]
    os.makedirs('datasets', exist_ok=True)
    data_path = f"datasets/{dataset_name}.mat"
    gt_path   = f"datasets/{dataset_name}_gt.mat"

    if info['data_url'] == 'KAGGLE_INPUT':
        # Try hardcoded known paths first, then glob fallback
        known_paths = {
            'Houston2013': (
                ['/kaggle/input/datasets/devgurucodes/hsi-houston/Houston.mat',
                 '/kaggle/input/hsi-houston/Houston.mat'],
                ['/kaggle/input/datasets/devgurucodes/hsi-houston/Houston_gt.mat',
                 '/kaggle/input/hsi-houston/Houston_gt.mat']
            ),
            'WHU_Hi': (
                ['/kaggle/input/datasets/devgurucodes/whu-dataset/WHU_Hi_HanChuan.mat',
                 '/kaggle/input/whu-dataset/WHU_Hi_HanChuan.mat'],
                ['/kaggle/input/datasets/devgurucodes/whu-dataset/WHU_Hi_HanChuan_gt.mat',
                 '/kaggle/input/whu-dataset/WHU_Hi_HanChuan_gt.mat']
            ),
        }
        
        found_data = found_gt = None
        
        # Method 1: Try known paths
        data_paths, gt_paths = known_paths.get(dataset_name, ([], []))
        for p in data_paths:
            if os.path.exists(p):
                found_data = p; break
        for p in gt_paths:
            if os.path.exists(p):
                found_gt = p; break
        
        # Method 2: Glob fallback
        if not found_data or not found_gt:
            search_patterns = {
                'Houston2013': (['*ouston*.mat'], ['*ouston*gt*.mat', '*ouston*GT*.mat']),
                'WHU_Hi':      (['*WHU*Han*[!g][!t].mat', '*HanChuan.mat'], ['*HanChuan*gt*.mat', '*HanChuan*GT*.mat']),
            }
            data_pats, gt_pats = search_patterns.get(dataset_name, ([], []))
            if not found_data:
                for pat in data_pats:
                    found_data = _find_in_kaggle_input(pat)
                    if found_data: break
            if not found_gt:
                for pat in gt_pats:
                    found_gt = _find_in_kaggle_input(pat)
                    if found_gt: break
        
        if found_data and found_gt:
            print(f"  Found {dataset_name} in Kaggle Input:")
            print(f"    Data: {found_data}")
            print(f"    GT:   {found_gt}")
            return found_data, found_gt
        else:
            # Debug: list what IS in /kaggle/input/
            import glob
            all_files = glob.glob('/kaggle/input/**/*.mat', recursive=True)
            print(f"  DEBUG: .mat files in /kaggle/input/: {all_files}")
            raise FileNotFoundError(
                f"\n  {dataset_name} not found in /kaggle/input/!"
                f"\n  Add it via: Kaggle notebook → Input → + Add Input"
            )

    if not os.path.exists(data_path):
        print(f"  Downloading {dataset_name} data...")
        urllib.request.urlretrieve(info['data_url'], data_path)
    if not os.path.exists(gt_path):
        print(f"  Downloading {dataset_name} ground truth...")
        urllib.request.urlretrieve(info['gt_url'], gt_path)
    return data_path, gt_path

def load_dataset(dataset_name):
    """Loads the HSI data cube and ground truth labels."""
    data_path, gt_path = download_dataset(dataset_name)
    info = DATASET_INFO[dataset_name]
    
    def _get_array(mat_file, expected_key):
        """Get array from .mat file, auto-detecting key if needed."""
        data = sio.loadmat(mat_file)
        if expected_key in data:
            return data[expected_key]
        # Auto-detect: find the largest numpy array (skip metadata keys)
        arrays = {k: v for k, v in data.items() if not k.startswith('__') and hasattr(v, 'shape')}
        if arrays:
            key = max(arrays, key=lambda k: arrays[k].size)
            print(f"    Key '{expected_key}' not found. Using '{key}' instead.")
            return arrays[key]
        raise KeyError(f"No arrays found in {mat_file}. Keys: {list(data.keys())}")
    
    X = _get_array(data_path, info['data_key'])
    y = _get_array(gt_path, info['gt_key'])
    print(f"  Loaded {dataset_name}: X={X.shape}, y={y.shape}, Classes={len(np.unique(y))-1}")
    return X, y

def apply_pca(X, num_components=30):
    """PCA dimensionality reduction along spectral axis."""
    h, w, b = X.shape
    flat = X.reshape(-1, b)
    pca = PCA(n_components=num_components, whiten=True)
    reduced = pca.fit_transform(flat).reshape(h, w, num_components)
    print(f"  PCA: {b} bands -> {num_components} components (explained var: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    return reduced, pca

def pad_with_zeros(X, margin):
    """Zero-pads the spatial dimensions."""
    h, w, b = X.shape
    padded = np.zeros((h + 2*margin, w + 2*margin, b), dtype=X.dtype)
    padded[margin:margin+h, margin:margin+w, :] = X
    return padded

def create_disjoint_patches(X, y, window_size=11, train_ratio=0.15, seed=42):
    """
    Standard disjoint center-pixel sampling (no data leakage).
    Same protocol as HybridSN, SSFTT, SpectralFormer.
    """
    rng = np.random.RandomState(seed)
    margin = (window_size - 1) // 2
    padded_X = pad_with_zeros(X, margin)

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list   = [], []

    classes = np.unique(y[y > 0])

    for c in classes:
        class_indices = np.argwhere(y == c)
        rng.shuffle(class_indices)

        n_train = max(1, int(len(class_indices) * train_ratio))
        train_centers = class_indices[:n_train]
        test_centers  = class_indices[n_train:]

        for r, c_idx in train_centers:
            patch = padded_X[r:r+window_size, c_idx:c_idx+window_size, :]
            X_train_list.append(patch)
            y_train_list.append(int(c - 1))

        for r, c_idx in test_centers:
            patch = padded_X[r:r+window_size, c_idx:c_idx+window_size, :]
            X_test_list.append(patch)
            y_test_list.append(int(c - 1))

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_test  = np.array(X_test_list, dtype=np.float32)
    y_test  = np.array(y_test_list, dtype=np.int64)

    print(f"  Split (seed={seed}): {len(y_train)} train, {len(y_test)} test")
    return X_train, X_test, y_train, y_test

print("Dataset loader ready. Supports 7 datasets (5 auto-download, 2 manual upload).")
