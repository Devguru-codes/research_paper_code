# =============================================================================
# Cell 2: Data Loader — IDENTICAL to main paper
# =============================================================================
# Same datasets, same splits, same seeds, same preprocessing
# This ensures FAIR comparison (reviewer requirement)

import glob

DATASET_INFO = {
    'IndianPines': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
        'data_key': 'indian_pines_corrected',
        'gt_key':   'indian_pines_gt',
        'num_classes': 16
    },
    'PaviaU': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
        'data_key': 'paviaU',
        'gt_key':   'paviaU_gt',
        'num_classes': 9
    },
    'Botswana': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/7/72/Botswana.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/5/58/Botswana_gt.mat',
        'data_key': 'Botswana',
        'gt_key':   'Botswana_gt',
        'num_classes': 14
    },
    'KSC': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/a/a6/KSC_gt.mat',
        'data_key': 'KSC',
        'gt_key':   'KSC_gt',
        'num_classes': 13
    },
    'WHU_Hi': {
        'data_url': 'KAGGLE_INPUT',
        'gt_url':   'KAGGLE_INPUT',
        'data_key': 'WHU_Hi_HanChuan',
        'gt_key':   'WHU_Hi_HanChuan_gt',
        'num_classes': 16
    },
}

def download_dataset(name):
    info = DATASET_INFO[name]
    os.makedirs('datasets', exist_ok=True)
    data_path = f"datasets/{name}.mat"
    gt_path   = f"datasets/{name}_gt.mat"

    if info['data_url'] == 'KAGGLE_INPUT':
        known = {
            'WHU_Hi': (
                ['/kaggle/input/datasets/devgurucodes/whu-dataset/WHU_Hi_HanChuan.mat',
                 '/kaggle/input/whu-dataset/WHU_Hi_HanChuan.mat'],
                ['/kaggle/input/datasets/devgurucodes/whu-dataset/WHU_Hi_HanChuan_gt.mat',
                 '/kaggle/input/whu-dataset/WHU_Hi_HanChuan_gt.mat']
            )
        }
        dp, gp = known.get(name, ([], []))
        fd = fg = None
        for p in dp:
            if os.path.exists(p): fd = p; break
        for p in gp:
            if os.path.exists(p): fg = p; break
        if not fd or not fg:
            # glob fallback
            all_mats = glob.glob('/kaggle/input/**/*.mat', recursive=True)
            for m in all_mats:
                if 'HanChuan' in m and 'gt' not in m.lower(): fd = m
                if 'HanChuan' in m and 'gt' in m.lower(): fg = m
        if fd and fg:
            return fd, fg
        raise FileNotFoundError(f"{name} not found in /kaggle/input/")
    
    if not os.path.exists(data_path):
        print(f"  Downloading {name} data...")
        urllib.request.urlretrieve(info['data_url'], data_path)
    if not os.path.exists(gt_path):
        print(f"  Downloading {name} GT...")
        urllib.request.urlretrieve(info['gt_url'], gt_path)
    return data_path, gt_path

def load_dataset(name):
    dp, gp = download_dataset(name)
    info = DATASET_INFO[name]
    
    def _get(mat_file, key):
        data = sio.loadmat(mat_file)
        if key in data: return data[key]
        arrays = {k: v for k, v in data.items() if not k.startswith('__') and hasattr(v, 'shape')}
        k = max(arrays, key=lambda k: arrays[k].size)
        return arrays[k]
    
    X = _get(dp, info['data_key'])
    y = _get(gp, info['gt_key'])
    print(f"  Loaded {name}: X={X.shape}, y={y.shape}, classes={info['num_classes']}")
    return X, y

def normalize(X):
    """Per-band normalization to [0, 1]."""
    X = X.astype(np.float32)
    for b in range(X.shape[2]):
        band = X[:, :, b]
        mn, mx = band.min(), band.max()
        if mx > mn:
            X[:, :, b] = (band - mn) / (mx - mn)
    return X

def pad_with_zeros(X, margin):
    h, w, b = X.shape
    padded = np.zeros((h + 2*margin, w + 2*margin, b), dtype=X.dtype)
    padded[margin:margin+h, margin:margin+w, :] = X
    return padded

def create_disjoint_patches(X, y, window_size=15, train_ratio=0.15, seed=42):
    """
    IDENTICAL split to main paper — pixel-level disjoint, no data leakage.
    """
    rng = np.random.RandomState(seed)
    margin = (window_size - 1) // 2
    padded_X = pad_with_zeros(X, margin)
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    classes = np.unique(y[y > 0])
    for c in classes:
        idx = np.argwhere(y == c)
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        for r, ci in idx[:n_train]:
            X_train.append(padded_X[r:r+window_size, ci:ci+window_size, :])
            y_train.append(int(c - 1))
        for r, ci in idx[n_train:]:
            X_test.append(padded_X[r:r+window_size, ci:ci+window_size, :])
            y_test.append(int(c - 1))
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    print(f"  Split (seed={seed}): {len(y_train)} train, {len(y_test)} test")
    return X_train, X_test, y_train, y_test

print("Data loader ready (identical protocol to main paper).")
